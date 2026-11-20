"""Audit published XML geometry against a grid definition, not model performance."""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import shapely
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

BASE = 'https://camelyon-dataset.s3.us-west-2.amazonaws.com/'
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
INCOMPLETE = {f'tumor_{n:03}' for n in [10,15,18,20,25,29,33,34,44,46,51,54,55,56,67,79,85,92,95,110]}


def fetch(key, cache):
    p = cache / key
    if p.exists():
        return p.read_bytes()
    with urllib.request.urlopen(BASE + key, timeout=120) as r:
        data = r.read(25_000_001)
    if len(data) > 25_000_000:
        raise ValueError('small-file limit exceeded')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return data


def list_keys(prefix):
    q = urllib.parse.urlencode({'list-type': '2', 'prefix': prefix, 'max-keys': 1000})
    with urllib.request.urlopen(BASE + '?' + q, timeout=40) as r:
        root = ET.fromstring(r.read())
    if root.findtext('s:IsTruncated', namespaces=NS) != 'false':
        raise ValueError('listing incomplete')
    return [{'key': x.findtext('s:Key', namespaces=NS),
             'bytes': int(x.findtext('s:Size', namespaces=NS))}
            for x in root.findall('s:Contents', NS)]


def range_read(key, start, count):
    end = start + count - 1
    req = urllib.request.Request(BASE + key + f'?audit_range={start}-{end}',
                                 headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=40) as r:
        cr = r.headers.get('Content-Range', '')
        if r.status != 206 or not cr.startswith(f'bytes {start}-'):
            raise ValueError(f'invalid range response {r.status} {cr}')
        data = r.read(count + 1)
    if len(data) != count:
        raise ValueError('range size mismatch')
    return data


def tiff_metadata(key, cache):
    p = cache / (key + '.header.json')
    if p.exists():
        return json.loads(p.read_text())
    h = range_read(key, 0, 16)
    endian = '<' if h[:2] == b'II' else '>' if h[:2] == b'MM' else None
    if endian is None:
        raise ValueError('invalid TIFF byte order')
    magic = struct.unpack(endian + 'H', h[2:4])[0]
    if magic == 42:
        offset = struct.unpack(endian + 'I', h[4:8])[0]
        nsize, esize, inline = 2, 12, 4
    elif magic == 43:
        if struct.unpack(endian + 'H', h[4:6])[0] != 8:
            raise ValueError('unexpected BigTIFF offsets')
        offset = struct.unpack(endian + 'Q', h[8:16])[0]
        nsize, esize, inline = 8, 20, 8
    else:
        raise ValueError('unsupported TIFF')
    n = struct.unpack(endian + ('H' if nsize == 2 else 'Q'), range_read(key, offset, nsize))[0]
    if n > 4096:
        raise ValueError('too many TIFF tags')
    raw = range_read(key, offset + nsize, n * esize)
    result = {'source': BASE + key, 'magic': magic, 'ifd_offset': offset,
              'header_sha256': hashlib.sha256(h + raw).hexdigest()}
    for i in range(n):
        e = raw[i*esize:(i+1)*esize]
        tag, typ = struct.unpack(endian + 'HH', e[:4])
        count = struct.unpack(endian + ('I' if magic == 42 else 'Q'), e[4:8 if magic == 42 else 12])[0]
        val = e[8:] if magic == 42 else e[12:]
        if tag in (256,257):
            fmt = {3:'H',4:'I',16:'Q'}[typ]
            result['width' if tag == 256 else 'height'] = struct.unpack(endian+fmt, val[:struct.calcsize(fmt)])[0]
        if tag == 270 and typ == 2 and count < 100_000:
            description = val[:count] if count <= inline else range_read(
                key, struct.unpack(endian + ('I' if inline == 4 else 'Q'), val)[0], count)
            desc = description.decode('utf-8', errors='replace')
            result['description'] = desc
            m = re.search(r'\bMPP\s*=\s*([\d.]+)', desc)
            if m:
                result['mpp_from_description'] = float(m.group(1))
    if 'width' not in result or 'height' not in result:
        raise ValueError('missing dimensions')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2))
    return result


def geometry(raw):
    root = ET.fromstring(raw)
    groups, positive, excluded = {}, [], []
    repairs = 0
    for a in root.findall('.//Annotations/Annotation'):
        group = a.attrib.get('PartOfGroup', '')
        groups[group] = groups.get(group,0)+1
        points = [(float(c.attrib['X']),float(c.attrib['Y'])) for c in a.findall('./Coordinates/Coordinate')]
        if len(points)<3:
            raise ValueError('degenerate annotation')
        p = Polygon(points)
        if not p.is_valid:
            p = shapely.make_valid(p)
            repairs += 1
        # make_valid can retain collapsed, zero-area line fragments. They are
        # not tumor area, and mixed-dimensional overlays can fail in GEOS.
        p = unary_union(components(p))
        if group.lower() in ('tumor','_0','_1'):
            positive.append(p)
        elif group.lower() in ('exclusion','_2'):
            excluded.append(p)
        else:
            raise ValueError(f'unrecognized annotation group {group}')
    if not positive:
        raise ValueError('no tumor geometry')
    t = unary_union(positive)
    if excluded:
        t = t.difference(unary_union(excluded))
    return t,groups,repairs


def components(g):
    if g.geom_type == 'Polygon':
        return [g]
    if hasattr(g,'geoms'):
        return [p for x in g.geoms for p in components(x)]
    return []


def score(g, width, height, thresholds=(0.0,0.05,0.2), grid=4096):
    unclipped = g.area
    g = g.intersection(box(0,0,width,height))
    objs = components(g)
    x0,y0,x1,y1 = g.bounds
    tiles, fractions = [], []
    for iy in range(max(0,int(y0)//grid), min(math.ceil(height/grid),int(y1)//grid+1)):
        for ix in range(max(0,int(x0)//grid), min(math.ceil(width/grid),int(x1)//grid+1)):
            cell = box(ix*grid,iy*grid,min((ix+1)*grid,width),min((iy+1)*grid,height))
            area = g.intersection(cell).area
            if area > 0:
                tiles.append(cell)
                fractions.append(area/cell.area)
    out=[]
    for threshold in thresholds:
        selected = [c for c,f in zip(tiles,fractions) if f>0 and f>=threshold]
        region = unary_union(selected)
        per_obj = []
        for i,obj in enumerate(objs):
            represented = obj.intersection(region).area > 0 if selected else False
            per_obj.append({'object_id':i,'area_px2':obj.area,'bounds':list(obj.bounds),
                            'represented':represented})
        out.append({'threshold':threshold,'positive_tiles':len(selected),'objects':len(objs),
                    'excluded_objects':sum(not x['represented'] for x in per_obj),
                    'represented_tumor_area_fraction':g.intersection(region).area/g.area if g.area else None,
                    'object_details':per_obj})
    return out, unclipped-g.area


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='verification/grid_audit')
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cache=Path('data/audit_cache')
    labels={}
    for row in csv.DictReader(io.StringIO(fetch('CAMELYON16/evaluation/reference.csv',cache).decode())):
        labels[row['image'].replace('.tif','')]=row['class']
    stages=fetch('CAMELYON17/stages.csv',cache).decode()
    for row in csv.DictReader(io.StringIO(stages)):
        if '.tif' in row['patient']:
            labels[row['patient'].replace('.tif','')]=row['stage']
    jobs=[]
    for dataset in ('CAMELYON16','CAMELYON17'):
        entries=list_keys(dataset+'/annotations/')
        if dataset=='CAMELYON16':entries=[e for e in entries if '/tumor_' in e['key']]
        if args.limit:entries=entries[:args.limit]
        for e in entries:jobs.append((dataset,e))
    (out/'source_manifest.json').write_text(json.dumps(jobs,indent=2))
    started=time.time();results=[];errors=[]
    print('START', 'pid', os.getpid(), 'jobs', len(jobs), flush=True)
    def run(job):
        dataset,e=job;sid=Path(e['key']).stem
        raw=fetch(e['key'],cache)
        md=tiff_metadata(f'{dataset}/images/{sid}.tif',cache)
        g,groups,repairs=geometry(raw)
        scores,clipped=score(g,md['width'],md['height'])
        return {'dataset':dataset,'slide_id':sid,'label':labels.get(sid,'unknown'),
                'annotation_complete':sid not in INCOMPLETE,'xml_sha256':hashlib.sha256(raw).hexdigest(),
                'metadata':md,'groups':groups,'repairs':repairs,'clipped_area_px2':clipped,'scores':scores}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,j):j for j in jobs}
        for f in cf.as_completed(futures):
            try:
                r=f.result();results.append(r)
                print('OK',r['dataset'],r['slide_id'],r['label'],r['scores'][-1]['objects'],
                      r['scores'][-1]['excluded_objects'],flush=True)
            except Exception as ex:
                j=futures[f];errors.append({'job':j,'error':repr(ex)})
                print('ERROR',j[1]['key'],repr(ex),flush=True)
            (out/'progress.json').write_text(json.dumps({'completed':len(results),'errors':errors,'total':len(jobs)}))
    results.sort(key=lambda x:(x['dataset'],x['slide_id']))
    summary=[]
    for dataset in ('CAMELYON16','CAMELYON17'):
        for complete in (True,False):
            for label in ('all','micro','macro','itc'):
                rows=[r for r in results if r['dataset']==dataset and r['annotation_complete']==complete
                      and (label=='all' or r['label']==label)]
                if not rows:continue
                for k,threshold in enumerate((0.0,0.05,0.2)):
                    summary.append({'dataset':dataset,'complete':complete,'label':label,'threshold':threshold,
                       'slides':len(rows),'slides_no_positive_tile':sum(r['scores'][k]['positive_tiles']==0 for r in rows),
                       'objects':sum(r['scores'][k]['objects'] for r in rows),
                       'excluded_objects':sum(r['scores'][k]['excluded_objects'] for r in rows),
                       'slides_with_excluded_objects':sum(r['scores'][k]['excluded_objects']>0 for r in rows)})
    payload={'command':sys.argv,'pid':os.getpid(),'started_unix':started,'elapsed_seconds':time.time()-started,
             'python':sys.version,'shapely':shapely.__version__,'results':results,'errors':errors,'summary':summary}
    (out/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)
    print('ERRORS',len(errors),'ELAPSED',payload['elapsed_seconds'],flush=True)
    return bool(errors)


if __name__=='__main__':
    sys.exit(main())
