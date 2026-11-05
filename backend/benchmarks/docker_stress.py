"""HTTP benchmark of the actual Compose API plus sampled API/PostgreSQL memory."""
import json
import re
import subprocess
import threading
import time
from urllib.parse import urlsplit

import httpx
import pandas as pd

from backend.app.config import settings
from backend.benchmarks.specified_data import workload


def docker(*args):
    result = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError('Docker command failed; check Docker Desktop and project permissions')
    return result.stdout.strip()


def memory_mb(value):
    match = re.fullmatch(r'([\d.]+)\s*([KMGT]?i?B)', value.strip())
    if not match:
        raise ValueError(f'Unsupported Docker memory unit: {value}')
    scales = {'B': 1, 'kB': 1000, 'KB': 1000, 'MB': 1000**2, 'GB': 1000**3,
              'TB': 1000**4, 'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3, 'TiB': 1024**4}
    return float(match[1]) * scales[match[2]] / 1024**2


class DockerMemory:
    def __init__(self, containers):
        self.containers, self.samples, self.errors = containers, [], []
        self.stop = threading.Event()

    def sample(self):
        values = [json.loads(line) for line in docker('stats', '--no-stream', '--format', '{{json .}}', *self.containers).splitlines()]
        if len(values) != len(self.containers):
            raise RuntimeError('Missing Docker memory sample')
        self.samples.append({'elapsed_clock': time.perf_counter(),
            'total_mb': sum(memory_mb(v['MemUsage'].split('/')[0]) for v in values)})

    def __enter__(self):
        self.sample()
        def run():
            while not self.stop.wait(.2):
                try:
                    self.sample()
                except Exception as exc:
                    self.errors.append(type(exc).__name__)
                    break
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=65)
        self.sample()
        if self.errors:
            raise RuntimeError('Docker memory sampling failed')


def docker_latency_experiment(grid, metadata, base_url, repeats=3, require_live=True):
    if urlsplit(base_url).hostname not in ('localhost', '127.0.0.1'):
        raise ValueError('Stress benchmark requires a local Compose API URL')
    containers = [docker('compose', 'ps', '-q', service) for service in ('api', 'db')]
    if any(not c or '\n' in c for c in containers):
        raise RuntimeError('Expected exactly one running API and PostgreSQL container')
    hostname = docker('inspect', '--format', '{{.Config.Hostname}}', containers[0])
    rows, memory_samples = [], []
    with httpx.Client(base_url=base_url.rstrip('/'), headers={'X-API-Key': settings.api_key}, timeout=900) as client:
        response = client.get('/health')
        response.raise_for_status()
        health = response.json()
        if health.get('database_backend') != 'postgresql' or health.get('runtime_hostname') != hostname:
            raise RuntimeError('API URL must identify this Compose API with PostgreSQL and current instrumentation')
        if require_live and (health['forecast_mode'] != 'chronos' or not health['llm_key_configured']):
            raise RuntimeError('Live stress evaluation requires Chronos and an LLM key in the container')
        if not require_live and (health['forecast_mode'] != 'offline' or health['llm_key_configured']):
            raise RuntimeError('Offline mode requires an offline container without an LLM key; refusing live provider calls')
        if require_live and (health['forecast_strategy'] != 'fixed' or health['llm_model'] != settings.llm_model):
            raise RuntimeError('Container must use fixed Chronos and the same LLM model as the host benchmark')
        def submit(content):
            response = client.post('/api/analyze', files={'file': ('stress.csv', content, 'text/csv')}, data={'horizon': '28'})
            response.raise_for_status()
            result = response.json()
            if 'storage_latency_ms' not in result or result.get('database_backend') != 'postgresql':
                raise RuntimeError('Missing measured PostgreSQL commit timing')
            if require_live and (result['mapping_backend'] != settings.llm_model or
                    any(item['model'] != 'Chronos-Bolt' for item in result['items'])):
                raise RuntimeError('Use FORECAST_STRATEGY=fixed and the configured live LLM for the specified stress experiment')
            return result
        submit(workload(grid, metadata, 10).to_csv(index=False).encode())  # Explicit excluded warm-up.
        for size in (10, 100, 200):
            frame = workload(grid, metadata, size)
            content = frame.to_csv(index=False).encode()
            for repetition in range(1, repeats + 1):
                print(f'Docker HTTP benchmark: {size} SKUs, repetition {repetition}/{repeats}', flush=True)
                with DockerMemory(containers) as memory:
                    start = time.perf_counter()
                    result = submit(content)
                    wall = (time.perf_counter() - start) * 1000
                timings = result['phase_latency_ms']
                phases = {'1_validation_parsing': timings['1_ingestion'],
                    '2_llm_schema_mapping': timings['2_schema_alignment'],
                    '3_postgresql_insertion_commit': result['storage_latency_ms'],
                    '4_chronos_reorder_math': timings['4_forecasting'] + timings['5_replenishment']}
                for phase, elapsed in phases.items():
                    rows.append({'n_skus': size, 'input_rows': len(frame), 'phase': phase,
                        'repetition': repetition, 'Latency_ms': elapsed, 'End_to_End_ms': wall,
                        'Normalization_ms': timings['3_normalization'], 'Serialization_ms': timings['6_json_delivery'],
                        'Peak_RAM_MB': max(s['total_mb'] for s in memory.samples), 'memory_samples': len(memory.samples),
                        'model': result['items'][0]['model'], 'mapping_backend': result['mapping_backend'],
                        'run_id': result['run_id'], 'source_skus': grid.shape[1],
                        'replicated_workload': size > grid.shape[1], 'measurement_backend': 'docker_http_postgresql'})
                memory_samples.extend(dict(s, n_skus=size, repetition=repetition) for s in memory.samples)
    raw = pd.DataFrame(rows)
    table = raw.groupby(['n_skus', 'input_rows', 'phase', 'model', 'mapping_backend', 'measurement_backend'], sort=False).agg(
        Latency_ms=('Latency_ms', 'median'), Latency_p95_ms=('Latency_ms', lambda s: s.quantile(.95)),
        Peak_RAM_MB=('Peak_RAM_MB', 'max'), End_to_End_ms=('End_to_End_ms', 'median'),
        Normalization_ms=('Normalization_ms', 'median'), Serialization_ms=('Serialization_ms', 'median'),
        repeats=('repetition', 'count'), replicated_workload=('replicated_workload', 'max')).reset_index()
    return table, raw, {'health': health, 'containers': containers, 'memory_samples': memory_samples,
        'memory_scope': 'Sampled sum of API and PostgreSQL Docker stats memory (MiB); shared total repeated across phases, not per-phase allocation',
        'timing_scope': 'Warm synchronous HTTP requests; insertion includes commit and occurs after forecast; normalization/serialization reported separately',
        'workload': 'Dense SKU-day rows; deterministic cyclic copies only for scaling, never forecasting accuracy'}
