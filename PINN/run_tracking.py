"""Per-run durable lifecycle records and exact source/config snapshots."""
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
MODEL_VERSION = '0.2.0'


def now():
    return datetime.now(timezone.utc).isoformat()


def git_info():
    def query(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=HERE, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
    return {'commit': query('rev-parse', 'HEAD'), 'branch': query('branch', '--show-current'),
            'worktree_dirty': bool(query('status', '--porcelain', '--', '.'))}


def source_hash(files):
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(file.name.encode())
        # Normalize Git's Windows line endings for stable cross-machine identity.
        digest.update(file.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def start_run(out, cfg, data_path, mode, fraction, data_kind, invalid_policy):
    repository_state = git_info()
    if out is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        out = HERE/'runs'/f'{stamp}_{data_kind}_{mode}'
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is nonempty; choose a new run directory.')
    out.mkdir(parents=True, exist_ok=True)
    source = out/'source'
    source.mkdir()
    files = list(HERE.glob('*.py'))
    for file in files:
        shutil.copy2(file, source/file.name)
    (source/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    started = now()
    record = {'run_id': out.name, 'status': 'running', 'started_at_utc': started,
              'started_at_new_york': datetime.fromisoformat(started).astimezone(ZoneInfo('America/New_York')).isoformat(),
              'model_version': MODEL_VERSION, 'source_sha256': source_hash(files), 'git': repository_state,
              'data_kind': data_kind, 'data_path': str(Path(data_path).resolve()),
              'data_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest() if Path(data_path).is_file() else None,
              'config': cfg, 'split_mode': mode, 'train_fraction': fraction, 'invalid_policy': invalid_policy,
              'timestamp_provenance': 'Runtime wall clock; explicit UTC and America/New_York offsets.',
              'models': [], 'results': []}
    save_record(out, record)
    return out, record


def save_record(out, record):
    (out/'run_record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


def finish_run(out, record, metrics=None, error=None):
    record['finished_at_utc'] = now()
    record['duration_seconds'] = (datetime.fromisoformat(record['finished_at_utc'])-datetime.fromisoformat(record['started_at_utc'])).total_seconds()
    record['status'] = 'failed' if error else 'completed'
    record['results'] = metrics or []
    if error:
        record['error'] = f'{type(error).__name__}: {error}'
    record['artifacts'] = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'run_record.json' and '__pycache__' not in p.parts}
    save_record(out, record)
    refresh_log()


def refresh_log():
    lines = ['# Experimental run register', '',
             'Each row is one run with shared splits across five models. Per-model timings, versions, configuration, data hashes, exclusions and results are in the linked record. Legacy run start times were not recorded; their file times are explicitly approximate.', '',
             '| Run | Started (New York) | Version | Data / split | Status | PINN RMSE (nm) |',
             '|---|---|---|---|---|---:|']
    for file in sorted((HERE/'runs').glob('*/run_record.json')):
        r = json.loads(file.read_text(encoding='utf-8'))
        score = next((f"{v['RMSE_nm']:.6f}" for v in r.get('results', []) if v['model'] == 'pinn'), '-')
        lines.append(f"| [{r['run_id']}](runs/{file.parent.name}/run_record.json) | {r.get('started_at_new_york') or 'unknown (legacy)'} | {r['model_version']} | {r['data_kind']} / {r['split_mode']} | {r['status']} | {score} |")
    (HERE/'RUN_LOG.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
