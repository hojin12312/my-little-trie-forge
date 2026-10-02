"""Resumable S/D measurement driver. Immutable attempts; one GPU process.

Real HTTP TTFT uses first nonempty content. Native phase rates reconcile
count deltas with complete batch traces. D uses independent cold states.
"""
import argparse
import concurrent.futures
import ctypes
import gzip
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from collections.abc import Mapping
import validate_measurement as validation

SOURCE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('MLTF_BENCH_OUTPUT','benchmark-output')).resolve()
TASK=ROOT.parent
DEV=SOURCE
SITE=Path(os.environ['MLTF_INSTALL_SITE']).resolve()
PYTHON=Path(os.environ.get('MLTF_PYTHON',__import__('sys').executable)).resolve()
MODEL=Path(os.environ['MLTF_MODEL_PATH']).resolve()
ADAPTER=Path(os.environ.get('MLTF_BENCH_ADAPTER','benchmark-build/all-ready')).resolve()
STATIC_LIBRARY=SOURCE/'build/engine/libsplash.a'
MODEL_ID='local/Qwen3.8-27B-q8c'
BUDGET=96*1024**3
LENGTHS=dict(zip(['2Ki','8Ki','32Ki','64Ki','128Ki','near256Ki'],[2048,8192,32768,65536,131072,261568]))
FAMILIES=['bugfix','feature','refactor','bugfix','feature']
COUNTERS=['prefill_input_tokens','prefill_wall_ms','decode_output_tokens','decode_wall_ms','drafted_tokens','accepted_draft_tokens']

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return {'unix':time.time(),'monotonic':time.monotonic()}
def host():
    out={}
    for key,cmd in [('swap',['sysctl','vm.swapusage']),('pressure',['memory_pressure','-Q']),('power',['pmset','-g','batt'])]:
        out[key]=subprocess.check_output(cmd,text=True).strip()
    out['disk_free_bytes']=shutil.disk_usage(TASK).free
    out.update(stamp());return out
def process_memory(root_pid):
    rows=[]
    for line in subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True).splitlines():
        try:pid,parent,rss=map(int,line.split());rows.append((pid,parent,rss*1024))
        except ValueError:pass
    ids={root_pid}
    for _ in range(8):
        grown=ids|{pid for pid,parent,rss in rows if parent in ids}
        if grown==ids:break
        ids=grown
    lib=ctypes.CDLL('/usr/lib/libproc.dylib');result=[]
    for pid,parent,rss in rows:
        if pid not in ids:continue
        buf=ctypes.create_string_buffer(96)
        code=lib.proc_pid_rusage(pid,0,ctypes.byref(buf))
        # SDK sys/resource.h rusage_info_v0: 16-byte UUID then 10 uint64.
        footprint=int.from_bytes(buf.raw[72:80],'little') if code==0 else None
        result.append({'pid':pid,'parent':parent,'rss_bytes':rss,'physical_footprint_bytes':footprint,'footprint_api':'proc_pid_rusage/RUSAGE_INFO_V0'})
    return {'processes':result,'rss_tree_sum_bytes':sum(r['rss_bytes'] for r in result),'rss_note':'process-tree sum may double-count shared pages; separate from Metal and per-process footprint'}
def clean_env():
    env={k:os.environ[k] for k in ['PATH','LANG','LC_ALL','TMPDIR','USER','LOGNAME'] if k in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',TRANSFORMERS_VERBOSITY='error',SPLASH_CATALOG_REFRESH='0',SPLASH_DEFAULT_REASONING_EFFORT='none')
    return env
def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
def get(port,path='/status',timeout=5):
    with urlopen(f'http://127.0.0.1:{port}{path}',timeout=timeout) as r:return json.load(r)
def bound(label,c):return max(900,math.ceil(LENGTHS[label]*c/100*1.5+240))
def fixtures(label,family,c,variant_start=0):
    if variant_start<0 or variant_start+c>4:raise ValueError('invalid fixed lane variant')
    return [ROOT/'corpus/prompts'/f'{label}-{family}-lane{i}' for i in range(variant_start,variant_start+c)]
def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGINT)
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    return p.returncode
class Server:
    def __init__(self,folder,timeout,trace=True,state=False,context=262144):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.port=free_port()
        env=clean_env();env['MLTF_DATA_ROOT']=str(self.folder/'data')
        if trace:env['SPLASH_DEV_DECODE_READY_TRACE']=str(self.folder/'batches.jsonl')
        if state:env['SPLASH_DEBUG_STATE_TRACE']=str(self.folder/'states.txt')
        cmd=[str(PYTHON),'-u',str(SITE/'server/server.py'),str(MODEL/'target'),str(MODEL/'draft'),'--tokenizer',str(MODEL/'tokenizer'),'--model',MODEL_ID,'--binary',str(SITE/'engine/splash'),'--host','127.0.0.1','--port',str(self.port),'--max-context',str(context),'--max-memory','96G','--request-timeout',str(timeout),'--default-reasoning-effort','none','--no-webui']
        self.files=[open(self.folder/'server.stdout.log','wb'),open(self.folder/'server.stderr.log','wb')]
        dump(self.folder/'launch.json',{'command':cmd,'whitelisted_env':{k:v for k,v in env.items() if k not in ['PATH','USER','LOGNAME','TMPDIR']},'trace':trace,'state_audit':state,'host_before':host(),'artifact':artifact()})
        self.p=subprocess.Popen(cmd,env=env,stdout=self.files[0],stderr=self.files[1],start_new_session=True)
        dump(self.folder/'process.json',{'pid':self.p.pid,'port':self.port,**stamp()})
        deadline=time.monotonic()+240
        while time.monotonic()<deadline:
            if self.p.poll() is not None:raise RuntimeError('server exited during startup')
            try:
                s=get(self.port)
                if s.get('ready') is True:
                    self.ready=s;dump(self.folder/'ready.json',s);break
            except Exception:pass
            time.sleep(.5)
        else:stop(self.p);raise TimeoutError('startup ready timeout')
    def close(self):
        code=stop(self.p)
        for f in self.files:f.close()
        dump(self.folder/'shutdown.json',{'exit_code':code,'host_after':host(),**stamp()})
        return code
def artifact():
    return {'native_binary_sha256':digest(SITE/'engine/splash'),'metallib_sha256':digest(SITE/'engine/splash.metallib'),
            'server_py_sha256':digest(SITE/'server/server.py'),'all_ready_binary_sha256':digest(ADAPTER),
            'all_ready_source_sha256':digest(SOURCE/'benchmarks/all_ready.mm'),'model_manifest_sha256':digest(MODEL/'manifest.json'),
            'qualified_static_library_sha256':digest(STATIC_LIBRARY),
            'harness_sha256':digest(__file__)}

def sse_request(port,prompt,timeout,path,gate=None,cap=512,body_override=None):
    body={'model':MODEL_ID,'messages':[{'role':'user','content':prompt}],'max_tokens':cap,'temperature':0,
          'top_p':1,'top_k':1,'chat_template_kwargs':{'enable_thinking':False},'reasoning_effort':'none',
          'stream':True,'stream_options':{'include_usage':True}}
    if body_override:body.update(body_override)
    if gate:gate.wait()
    start=time.monotonic();first=None;last=None;events=[];chunks=[];reasoning=[];usage=None;finish=None;error=None;http_code=None
    try:
      req=Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST')
      with urlopen(req,timeout=timeout) as r:
        http_code=r.status
        for raw in r:
          at=time.monotonic()
          if not raw.startswith(b'data: '):continue
          line=raw[6:].strip()
          if line==b'[DONE]':break
          event=json.loads(line);events.append({'at':at,'event':event})
          if event.get('usage'):usage=event['usage']
          for choice in event.get('choices',[]):
            delta=choice.get('delta',{});content=delta.get('content')
            if content:
                if first is None:first=at
                last=at;chunks.append(content)
            if delta.get('reasoning_content'):reasoning.append(delta['reasoning_content'])
            if choice.get('finish_reason'):finish=choice['finish_reason']
    except HTTPError as e:http_code=e.code;error=e.read().decode(errors='replace')
    except Exception as e:error=repr(e)
    end=time.monotonic()
    record={'submit':start,'first_content':first,'last_content':last,'terminal':end,'ttft_seconds':first-start if first else None,
            'http_status':http_code,'content':''.join(chunks),'reasoning_text':''.join(reasoning),'usage':usage,
            'finish_reason':finish,'error':error,'events':events,'payload_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()}
    dump(path,record);return record
def monitor(server,done,path):
    with open(path,'w') as out:
      while not done.is_set():
        try:
            status=get(server.port,timeout=3);record={'time':stamp(),'status':status,'process_memory':process_memory(server.p.pid)}
            out.write(json.dumps(record)+'\n');out.flush()
            if status.get('memory_governor',{}).get('system_pressure')=='critical':
                stop(server.p);return
        except Exception:pass
        done.wait(1.0)
def trace(path):
    path=Path(path)
    if path.exists():stream=path.open()
    elif Path(str(path)+'.gz').exists():stream=gzip.open(str(path)+'.gz','rt')
    else:return []
    with stream:return [json.loads(l) for l in stream if l.startswith('{')]
def preserve_compressed_raw(folder):
    folder=Path(folder).resolve()
    if not folder.is_relative_to(ROOT.resolve()):raise RuntimeError('compression outside owned evidence root')
    records=[]
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or p.suffix not in ['.jsonl','.log'] or p.stat().st_size<1024*1024:continue
        target=Path(str(p)+'.gz')
        if target.exists():raise RuntimeError('immutable compressed target already exists')
        original=digest(p);size=p.stat().st_size
        with p.open('rb') as src,target.open('wb') as dst,gzip.GzipFile(filename=p.name,fileobj=dst,mode='wb',mtime=0) as compressed:shutil.copyfileobj(src,compressed)
        with gzip.open(target,'rb') as src:
            restored=hashlib.file_digest(src,'sha256').hexdigest()
        if restored!=original:raise RuntimeError('lossless compression mismatch')
        records.append({'original_path':str(p.relative_to(folder)),'original_sha256':original,'original_bytes':size,
                        'stored_path':str(target.relative_to(folder)),'compressed_sha256':digest(target),'compressed_bytes':target.stat().st_size,
                        'original_preserved':'Exact original bytes retained losslessly in verified gzip; recover by decompression.'})
        p.unlink()
    if records:dump(folder/'RAW_COMPRESSION_INDEX.json',{'lossless_verified':True,'files':records})

def run_s(label,c,family,folder,trace_on=True,state=False,variant_start=0):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);server=None;done=threading.Event();thread=None
    started=stamp();result={'label':label,'C':c,'family':family,'track':'S','recording_on':False,'start':started}
    try:
      server=Server(folder/'server',bound(label,c),trace_on,state)
      before=get(server.port);dump(folder/'before.json',before)
      thread=threading.Thread(target=monitor,args=(server,done,folder/'status.jsonl'),daemon=True);thread.start()
      gate=threading.Barrier(c+1);prompts=fixtures(label,family,c,variant_start)
      result['fixture_ids']=[p.name for p in prompts]
      with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
        jobs=[pool.submit(sse_request,server.port,p.with_suffix('.txt').read_text(),bound(label,c)+60,folder/f'request-lane{i}.json',gate) for i,p in enumerate(prompts)]
        gate.wait();rows=[j.result() for j in jobs]
      after=get(server.port);dump(folder/'after.json',after)
      a=before['metrics'];b=after['metrics'];d=validation.counter_delta({k:a[k] for k in COUNTERS},{k:b[k] for k in COUNTERS},before['instance']['id'],after['instance']['id'])
      expected=LENGTHS[label]*c;outputs=sum((r.get('usage') or {}).get('completion_tokens',0) for r in rows)
      errors=[r['error'] for r in rows if r['error']]
      if errors:raise RuntimeError('request errors: '+str(errors))
      if any(r['http_status']!=200 or not r['ttft_seconds'] or r['reasoning_text'] for r in rows):raise RuntimeError('HTTP/TTFT/thinking mismatch')
      if any(r['usage']['prompt_tokens']!=LENGTHS[label] or r['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)!=0 for r in rows):raise RuntimeError('exact input/cold mismatch')
      if d['prefill_input_tokens']!=expected or d['decode_output_tokens']!=outputs:raise RuntimeError('phase count/usage reconciliation failed')
      if any(d[k]<0 for k in d) or min(d['prefill_wall_ms'],d['decode_wall_ms'])<=0:raise RuntimeError('invalid counter delta')
      wave=max(r['terminal'] for r in rows)-min(r['submit'] for r in rows)
      batches=trace(folder/'server/batches.jsonl');completes=[e for e in batches if e['event']=='complete'];decode=[e for e in completes if e['kind']==1]
      width_counts={str(w):sum(e['width']==w for e in decode) for w in range(1,5)}
      width_time={str(w):sum(e['command_wall_ms'] for e in decode if e['width']==w) for w in range(1,5)}
      if trace_on:
        if sum(e['confirmed_tokens'] for e in decode)!=outputs:raise RuntimeError('trace output count mismatch')
        if abs(sum(e['command_wall_ms'] for e in decode)-d['decode_wall_ms'])>max(.05,len(decode)*.0011):raise RuntimeError('trace/counter wall mismatch')
      result.update(status='PASS',phase_delta=d,ttft_seconds_per_request=[r['ttft_seconds'] for r in rows],full_wave_seconds=wave,
                    full_wave_tps=outputs/wave,prefill_native_tps=expected/(d['prefill_wall_ms']/1000),
                    natural_decode_native_tps=outputs/(d['decode_wall_ms']/1000),actual_outputs=[r['usage']['completion_tokens'] for r in rows],
                    finish_reasons=[r['finish_reason'] for r in rows],cached_tokens=0,physical_b_counts=width_counts,physical_b_wall_ms=width_time,
                    requests=rows,peak_scope='fresh-process native lifetime high-water including compile warmup; request samples separately preserved',
                    artifact=artifact(),recording_on=False)
      result['memory_before']=before.get('memory_actual');result['memory_after']=after.get('memory_actual')
      result['peak_metal_bytes']=after['memory_actual']['peak_bytes']
      sampled=trace(folder/'status.jsonl')
      result['request_sampled_peak_metal_bytes']=max((e['status']['memory_actual']['current_bytes'] for e in sampled),default=None)
      pp_events=[e for e in completes if e['kind']==0]
      pp_submits=[e for e in batches if e['event']=='submit' and e['kind']==0]
      result['prefill_elapsed_tps']=expected/((pp_events[-1]['steady_ms']-pp_submits[0]['steady_ms'])/1000) if pp_events and pp_submits else None
      result['maximum_ready_residents_observed']=max((sum(r['phase']==5 for r in e['requests']) for e in batches),default=0)
      result['sampled_rss_tree_peak_bytes']=max((e['process_memory']['rss_tree_sum_bytes'] for e in sampled),default=None)
      result['sampled_native_footprint_peak_bytes']=max((p['physical_footprint_bytes'] or 0 for e in sampled for p in e['process_memory']['processes'] if p['parent']==server.p.pid),default=None)
    except Exception as e:result.update(status='ERROR',error=repr(e))
    finally:
      done.set()
      if thread:thread.join(timeout=5)
      if server:
        try:result['exit_code']=server.close()
        except Exception as e:result['shutdown_error']=repr(e)
      result['end']=stamp();result['method_sha256']=digest(ROOT/'METHOD.json');dump(folder/'result.json',result)
    return result

def run_d(label,c,family,folder,quiet=False,audit=False,plan=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    cmd=[str(ADAPTER),str(SITE/'engine/splash.metallib'),str(MODEL),str(BUDGET),'512',*[str(p.with_suffix('.ids')) for p in fixtures(label,family,c)]]
    if quiet:cmd.append('--quiet')
    if audit:cmd.append('--audit-state')
    if plan:cmd.append('--plan-only')
    dump(folder/'launch.json',{'command':cmd,'artifact':artifact(),'host_before':host(),'start':stamp()})
    with open(folder/'native.jsonl','wb') as out,open(folder/'native.stderr.log','wb') as err:
      p=subprocess.Popen(cmd,env=clean_env(),stdout=out,stderr=err,start_new_session=True)
      dump(folder/'process.json',{'pid':p.pid,**stamp()})
      deadline=time.monotonic()+bound(label,c)+180
      with open(folder/'process-memory.jsonl','w') as memory:
        while p.poll() is None:
          memory.write(json.dumps({'time':stamp(),'process_memory':process_memory(p.pid)})+'\n');memory.flush()
          if time.monotonic()>deadline:stop(p);break
          time.sleep(1)
    events=trace(folder/'native.jsonl');failure=(folder/'native.stderr.log').read_text();result={'label':label,'C':c,'family':family,'track':'D','exit_code':p.returncode,'artifact':artifact(),'recording_on':False,'cached_tokens':0,'host_after':host(),'end':stamp()}
    if p.returncode:
        kind=next((s for s in ['READY_RESIDENCY_LIMITED','MEMORY_LIMITED','CONTEXT_UNSUPPORTED','SHORT_WINDOW'] if s in failure),'ERROR')
        result.update(status=kind,limitation_reason=failure.strip(),evidence_refs=[str(folder/'launch.json'),str(folder/'native.jsonl'),str(folder/'native.stderr.log')])
    elif plan:result.update(status='PLAN_ONLY',plan=next(e for e in events if e['event']=='plan'))
    else:
        selected=[e for e in events if e['event']=='decode_cycle' and e['primary']]
        outputs=[e for e in events if e['event']=='output'];totals=next(e for e in events if e['event']=='totals')
        if quiet:result.update(status='CONTROL',totals=totals,outputs=outputs)
        elif not selected or len(selected)<16 or sum(e['wall_seconds'] for e in selected)<1:result.update(status='SHORT_WINDOW',selected_cycles=len(selected))
        else:
            validation.validate_d_window(events,c)
            confirmed=sum(e['confirmed'] for e in selected);drafted=sum(e['drafted'] for e in selected);accepted=sum(e['accepted'] for e in selected)
            seconds=sum(e['wall_seconds'] for e in selected);elapsed=selected[-1]['end']-selected[0]['begin']
            pp=[e for e in events if e['event']=='prefill'];mem=[e for e in events if e['event']=='memory']
            assert all(e['B']==c for e in selected) and len([e for e in events if e['event']=='ready'])==c
            assert sum(e['input_tokens'] for e in pp)==LENGTHS[label]*c
            assert (confirmed,drafted,accepted)==(totals['primary_confirmed'],totals['primary_drafted'],totals['primary_accepted'])
            result.update(status='PASS',native_decode_tps=confirmed/seconds,elapsed_decode_tps=confirmed/elapsed,
                          observed_physical_b=c,all_ready_verified=True,selected_cycle_count=len(selected),decode_wall_seconds=seconds,
                          elapsed_decode_seconds=elapsed,confirmed_tokens=confirmed,accepted_draft_tokens=accepted,drafted_denominator_tokens=drafted,
                          acceptance_definition='shipping accepted retained draft tokens / all seven proposed tokens per active lane per full-width cycle; anchor/bonus excluded, no inactive rows, boundary completion cycle excluded',
                          confirmed_tokens_per_cycle_per_lane=confirmed/(len(selected)*c),cycles_per_second=len(selected)/seconds,
                          prefill_native_tps=LENGTHS[label]*c/sum(e['wall_seconds'] for e in pp),
                          prefill_elapsed_tps=LENGTHS[label]*c/(pp[-1]['end']-pp[0]['begin']),
                          ready_to_first_decode_ms=(selected[0]['begin']-next(e['at'] for e in events if e['event']=='ALL_READY_RELEASE'))*1000,
                          peak_metal_bytes=max(e['peak_metal_bytes'] for e in mem),peak_scope='fresh-process lifetime native tracked dense+sparse resident peak; warmup and phase samples retained',
                          outputs=outputs,actual_outputs=[len(e['tokens']) for e in outputs],finish_reasons=[e['finish_reason'] for e in outputs])
    result['method_sha256']=digest(ROOT/'METHOD.json');dump(folder/'result.json',result);return result

def jobs():
    rows=[]
    for k,label in enumerate(LENGTHS):
        order=[2,1,4,3];order=order[k%4:]+order[:k%4]
        for c in order:
            n=5 if label=='2Ki' and c in (1,4) else 3
            for repeat in range(n):
                for track in (['S','D'] if repeat%2==0 else ['D','S']):rows.append({'label':label,'C':c,'repeat':repeat,'family':FAMILIES[repeat],'track':track})
    return rows
def run_grid():
    schedule=jobs();freeze=json.loads((ROOT/'METHOD.json').read_text())
    if freeze['status']!='FROZEN_BEFORE_OFFICIAL_MEASUREMENTS':raise RuntimeError('method is not frozen')
    results=[]
    for job_index,j in enumerate(schedule):
      key=f"{j['label']}-C{j['C']}-r{j['repeat']}-{j['family']}-{j['track']}";folder=ROOT/'runs'/key
      if (folder/'result.json').exists():
        r=json.loads((folder/'result.json').read_text())
        if r.get('method_sha256')!=digest(ROOT/'METHOD.json') or r.get('artifact')!=freeze['artifact']:raise RuntimeError('resume identity mismatch: '+key)
      else:
        if shutil.disk_usage(TASK).free<20*1024**3:raise RuntimeError('disk safety floor: 20GiB')
        if folder.exists():raise RuntimeError(f'incomplete attempt requires preserved explicit review: {folder}')
        dump(ROOT/'CURRENT.json',{'run_id':key,'job':j,'completed_runs':len(results),'total_runs':len(schedule),'start':stamp()})
        r=(run_s if j['track']=='S' else run_d)(j['label'],j['C'],j['family'],folder)
        preserve_compressed_raw(folder)
        print(key,r['status'],round(r.get('native_decode_tps',r.get('natural_decode_native_tps',0)),2),flush=True)
      r['run_id']=key;results.append(r)
      dump(ROOT/'RUN_INDEX.json',{'runs':[{'run_id':x['run_id'],'status':x['status'],'track':x['track']} for x in results],
                                  'complete':len(results)==len(schedule),'total_runs':len(schedule),'updated':stamp()})
      if r['status'] in ['ERROR','SHORT_WINDOW']:
          dump(ROOT/'REQUIRES_REVIEW.json',{'run_id':key,'status':r['status']});raise RuntimeError(f"unresolved sample {key}: {r['status']}")
      if job_index+1==len(schedule) or schedule[job_index+1]['label']!=j['label']:
          import summarize
          dump(ROOT/'results.json',summarize.ledger())
    dump(ROOT/'GRID_EXECUTED.json',{'status':'ALL_PLANNED_RUNS_ACCOUNTED','runs':len(results),'updated':stamp()})
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','dry-run','grid','S','D','plan']);ap.add_argument('--label',default='2Ki');ap.add_argument('--C',type=int,default=1);ap.add_argument('--family',default='bugfix');ap.add_argument('--out',type=Path);ap.add_argument('--quiet',action='store_true');ap.add_argument('--state',action='store_true');ap.add_argument('--no-trace',action='store_true');a=ap.parse_args()
    if a.mode=='freeze':
        if (ROOT/'METHOD.json').exists():raise RuntimeError('existing method is immutable; use a fresh output root')
        corpus=json.loads((ROOT/'corpus/manifest.json').read_text())
        dump(ROOT/'METHOD.json',{'status':'FROZEN_BEFORE_OFFICIAL_MEASUREMENTS','artifact':artifact(),
             'near_max':{'input_tokens':261568,'headroom_tokens':64,'output_cap':512,'logical_limit':262144},
             'sampling':{'temperature':0,'thinking':False,'top_p':1,'top_k':1},'corpus_sha256':corpus['corpus_sha256'],
             'run_order':jobs(),'recording_on':False,'reference_method':'docs/MEASUREMENT_METHOD.md'})
        print('User-local method frozen before timing');return
    if a.mode=='dry-run':print(json.dumps({'runs':jobs(),'total':len(jobs()),'prompt_token_work':sum(LENGTHS[j['label']]*j['C'] for j in jobs())},indent=2));return
    if a.mode=='grid':run_grid();return
    if not a.out:ap.error('--out required')
    r=run_s(a.label,a.C,a.family,a.out,not a.no_trace,a.state) if a.mode=='S' else run_d(a.label,a.C,a.family,a.out,a.quiet,a.state,a.mode=='plan')
    print(json.dumps({k:v for k,v in r.items() if k not in ['requests','outputs','artifact']},indent=2))
if __name__=='__main__':main()
