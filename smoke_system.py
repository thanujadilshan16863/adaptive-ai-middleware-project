"""Exercise an already-running deployment without making external LLM calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.parse import urlparse

import httpx
import numpy as np
import pandas as pd

from backend.app.config import settings
from backend.app.schemas import FIELDS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--skus',type=int,default=4)
    parser.add_argument('--output',type=Path,default=Path('outputs/system_smoke.json'))
    args=parser.parse_args()
    if urlparse(args.base_url).hostname not in ('localhost','127.0.0.1','::1'):
        parser.error('This smoke script is restricted to local deployments')
    df=pd.read_csv('data/raw/automotive_transactions.csv')
    df=df[df.part_id.isin(sorted(df.part_id.unique())[:args.skus])]
    headers={'X-API-Key':settings.api_key}
    report={'base_url':args.base_url,'skus':df.part_id.nunique(),'external_llm_called':False}
    with httpx.Client(base_url=args.base_url,headers=headers,timeout=120) as client:
        response=client.get('/health');response.raise_for_status()
        report['health_before']=response.json()
        submitted=client.post('/api/jobs',files={'file':('smoke.csv',df.to_csv(index=False).encode())},
                              data={'mapping_json':json.dumps({f:f for f in FIELDS}),'horizon':'28'})
        submitted.raise_for_status()
        job_id=submitted.json()['job_id']
        print(f'Submitted {len(df)} rows / {report["skus"]} SKUs: {job_id}',flush=True)
        start=time.perf_counter()
        deadline=time.monotonic()+600
        phases=[]
        while time.monotonic()<deadline:
            response=client.get('/api/jobs/'+job_id);response.raise_for_status();job=response.json()
            if job['phase'] not in phases:
                phases.append(job['phase']);print(job['phase'],flush=True)
            if job['status'] in ('completed','failed'):
                break
            time.sleep(.5)
        if job['status']!='completed':
            raise RuntimeError(f'Job did not complete: {job}')
        response=client.get('/api/runs/'+job['run_id']);response.raise_for_status();run=response.json()
        assert len(run['items'])==report['skus']
        assert all(len(item['forecast']['mean'])>=28 for item in run['items'])
        transactions=client.get('/api/runs/'+job['run_id']+'/transactions');transactions.raise_for_status()
        assert transactions.json()['items']
        history=client.get('/api/runs');history.raise_for_status()
        assert any(row['run_id']==job['run_id'] for row in history.json()['items'])
        report.update(run_id=job['run_id'],job_seconds=time.perf_counter()-start,observed_phases=phases,
                      models=sorted(set(item['model'] for item in run['items'])),
                      adaptive_audits=sum(item['selection'] is not None for item in run['items']),
                      health_after=client.get('/health').json())
    def probe(i):
        start=time.perf_counter()
        response=httpx.get(args.base_url+'/health',timeout=30)
        response.raise_for_status()
        return (time.perf_counter()-start)*1000
    with ThreadPoolExecutor(max_workers=8) as pool:
        elapsed=list(pool.map(probe,range(40)))
    report['health_concurrency_smoke']={'requests':40,'concurrency':8,'errors':0,
                                      'p50_ms':float(np.median(elapsed)),'p95_ms':float(np.quantile(elapsed,.95)),
                                      'scope':'health endpoint only; not inference throughput or production capacity'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
