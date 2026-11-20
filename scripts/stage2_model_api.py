"""Independent phase-two API adapter. Import/encoding/ledger reads never call a model.

The only network operation is Gateway.request(). Unit tests inject a transport and
are marked as fixtures, never mixed with actual model runs. No historical deadline.
"""
import base64
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import signal
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid

from PIL import Image

VERSION = 'stage2_model_api_v1'


def write_json(path, value):
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(raw)
    temp.replace(path)


@dataclass(frozen=True)
class APIConfig:
    image_edge: int = 512
    jpeg_quality: int = 90
    max_tokens: int = 1600
    timeout_seconds: int = 90
    retries: int = 1
    max_text_characters: int = 200000
    max_images: int = 4

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{key} must be an integer')
        if self.image_edge not in (512, 1024) or not 1 <= self.jpeg_quality <= 100:
            raise ValueError('unsupported encoding configuration')
        if not 1 <= self.max_tokens <= 8192 or not 1 <= self.timeout_seconds <= 300:
            raise ValueError('invalid output/timeout bound')
        if self.retries not in (0, 1) or not 1 <= self.max_images <= 4 or not 1000 <= self.max_text_characters <= 1000000:
            raise ValueError('invalid retry/image/text bound')


def encode(image, config):
    """Never silently resize: evidence coordinates must match transmitted pixels."""
    if not isinstance(image, Image.Image) or max(image.size) > config.image_edge:
        raise ValueError('rendered image exceeds configured transport size')
    output = io.BytesIO()
    image.convert('RGB').save(output, format='JPEG', quality=config.jpeg_quality)
    raw = output.getvalue()
    return raw, {'sha256': hashlib.sha256(raw).hexdigest(), 'size': list(image.size), 'bytes': len(raw)}


def live_transport(body, timeout, provider):
    """Resolve saved literal credentials in memory, never copy configuration."""
    settings = json.loads((Path.home()/'.pi/agent/models.json').read_text())['providers']
    route = settings[provider]
    key = route.get('apiKey')
    if not isinstance(key, str) or not key or key.startswith('!'):
        raise ValueError('unsupported credential resolution')
    request = urllib.request.Request(route['baseUrl'].rstrip('/')+'/chat/completions',
        data=json.dumps(body, allow_nan=False).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+key})
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('live transport requires the serial main-thread runner')
    def expire(signum, frame):
        raise TimeoutError('bounded API attempt elapsed')
    previous_handler = signal.signal(signal.SIGALRM, expire)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
    timer_started = time.monotonic()
    try:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status, raw = str(response.status), response.read(8000000)
        except urllib.error.HTTPError as error:
            status, raw = str(error.code), error.read(100000)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(.001, previous_timer[0]-(time.monotonic()-timer_started)), previous_timer[1])
    try:
        result = {'status': status, 'body': json.loads(raw)}
    except (ValueError, UnicodeDecodeError):
        result = {'status': status, 'error_type': 'NonJSONResponse'}
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
    for value in settings.values():
        secret = value.get('apiKey') if isinstance(value, dict) else None
        if isinstance(secret, str) and secret:
            serialized = serialized.replace(secret, '[REDACTED]')
    return json.loads(serialized)


class Gateway:
    def __init__(self, output, config, transport=None, max_attempts=None):
        self.output, self.config = Path(output), config
        self.output.mkdir(parents=True, exist_ok=True)
        self.transport = transport or live_transport
        self.fixture = transport is not None
        if max_attempts is not None and (isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1):
            raise ValueError('max_attempts must be a positive integer')
        self.max_attempts = max_attempts
        self.db = sqlite3.connect(self.output/'calls.sqlite3')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, job TEXT, started REAL, provider TEXT, model TEXT, attempt INTEGER, status TEXT, artifact TEXT)')
        self.db.commit()

    def request(self, provider, model, text, attachments, *, job, deadline):
        if not isinstance(text, str) or len(text) > self.config.max_text_characters or len(attachments) > self.config.max_images:
            raise ValueError('request exceeds configured shape; text is never silently truncated')
        parts, metadata = [{'type': 'text', 'text': text}], []
        for view_id, image in attachments:
            raw, meta = encode(image, self.config)
            cache = self.output/'images'
            cache.mkdir(exist_ok=True)
            dest = cache/(meta['sha256']+'.jpg')
            if dest.exists() and dest.read_bytes() != raw:
                raise RuntimeError('image cache mismatch')
            if not dest.exists():
                dest.write_bytes(raw)
            metadata.append({'view_id': view_id, **meta})
            parts.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+base64.b64encode(raw).decode()}})
        body = {'model': model, 'max_tokens': self.config.max_tokens,
                'messages': [{'role': 'user', 'content': parts}]}
        # No unsupported temperature/seed fields; repetition IDs are not provider RNG seeds.
        body_hash = hashlib.sha256(json.dumps(body, allow_nan=False).encode()).hexdigest()
        attempt_ids = []
        for attempt in range(self.config.retries+1):
            if self.max_attempts is not None and self.db.execute('SELECT count(*) FROM calls').fetchone()[0] >= self.max_attempts:
                raise TimeoutError('explicit run request-attempt limit reached')
            timeout = min(self.config.timeout_seconds, deadline-time.time())
            if timeout <= 0:
                raise TimeoutError('explicit run/episode deadline reached before request')
            cid = uuid.uuid4().hex
            folder = self.output/'calls'/cid
            folder.mkdir(parents=True)
            start = time.time()
            self.db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?,?)',
                            (cid, job, start, provider, model, attempt, 'running', str(folder)))
            self.db.commit()
            attempt_ids.append(cid)
            write_json(folder/'request.json', {'id': cid, 'job_log_only': job,
                'provider': provider, 'model_requested': model, 'attempt': attempt,
                'started_unix': start, 'text': text, 'images': metadata,
                'payload_sha256': body_hash, 'api_config': asdict(self.config),
                'actual_timeout_seconds': timeout, 'unit_test_fixture': self.fixture})
            try:
                result = self.transport(body, timeout, provider)
                if not isinstance(result, dict) or 'status' not in result:
                    raise ValueError('transport result lacks status')
            except (KeyboardInterrupt, SystemExit):
                self.db.execute('UPDATE calls SET status=? WHERE id=?', ('interrupted_unknown', cid))
                self.db.commit()
                write_json(folder/'response.json', {'id': cid, 'status': 'interrupted_unknown',
                    'upstream_completion_and_cost': 'unknown', 'elapsed_seconds': time.time()-start})
                raise
            except Exception as error:
                # Do not serialize exception messages, which can contain credentials/URLs.
                result = {'status': 'transport_error', 'error_type': type(error).__name__}
            result = {**result, 'id': cid, 'status': str(result['status']),
                      'attempt_ids': list(attempt_ids), 'images': metadata,
                      'elapsed_seconds': time.time()-start, 'unit_test_fixture': self.fixture}
            write_json(folder/'response.json', result)
            self.db.execute('UPDATE calls SET status=? WHERE id=?', (result['status'], cid))
            self.db.commit()
            if result['status'] not in ('429', '500', '502', '503', '504', 'transport_error'):
                return result
        return result

    def summary(self):
        rows = self.db.execute('SELECT id, status, artifact FROM calls ORDER BY started').fetchall()
        statuses, usage, elapsed, images, pixels, byte_count = {}, [], 0, 0, 0, 0
        for cid, status, artifact in rows:
            statuses[status] = statuses.get(status, 0)+1
            folder = Path(artifact)
            request = json.loads((folder/'request.json').read_text())
            for im in request['images']:
                images += 1
                pixels += im['size'][0]*im['size'][1]
                byte_count += im['bytes']
            response = json.loads((folder/'response.json').read_text()) if (folder/'response.json').exists() else {}
            elapsed += response.get('elapsed_seconds', 0)
            u = response.get('body', {}).get('usage')
            if isinstance(u, dict):
                usage.append({'call_id': cid, 'usage': u})
        return {'requests': len(rows), 'statuses': statuses, 'usage_records': usage,
                'reported_total_tokens': sum(r['usage'].get('total_tokens', 0) or 0 for r in usage),
                'request_elapsed_seconds_sum': elapsed, 'images_with_repeats': images,
                'pixels_with_repeats': pixels, 'jpeg_bytes_with_repeats': byte_count,
                'cost': None, 'unknown_upstream_cost_possible': any(k in statuses for k in ('running', 'interrupted_unknown', 'transport_error')),
                'unit_test_fixture': self.fixture}

    def close(self):
        self.db.close()
