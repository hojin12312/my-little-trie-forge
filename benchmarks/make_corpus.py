"""Freeze public-safe coding fixtures; exact chat-template token counts.

Original task code is Apache-2.0. Supporting dependency sources are unmodified
CPython 3.13.14 stdlib code, pinned by individual SHA256 and PSF notices.
No private runtime code, repeated filler, random token padding, or user prompts.
"""
import ast
import os
import hashlib
import json
import sys
import sysconfig
from collections.abc import Mapping
from pathlib import Path
from transformers import AutoTokenizer

CORPUS = Path(os.environ.get('MLTF_CORPUS_OUTPUT', 'benchmark-output/corpus')).resolve()
ROOT = CORPUS.parent
MODEL = Path(os.environ['MLTF_MODEL_PATH']).resolve()
LENGTHS = [2048, 8192, 32768, 65536, 131072, 261568]
LABELS = ['2Ki','8Ki','32Ki','64Ki','128Ki','near256Ki']

QUEUE = '''"""Queue/cache example: cancellation is advisory; old work may finish."""
from dataclasses import dataclass
from pathlib import Path
import json
@dataclass
class Job:
    key: str
    generation: int
    cancelled: bool = False
class Queue:
    def __init__(self, cache):
        self.cache = cache
        self.generations = {}
        self.active = {}
    def submit(self, key):
        generation = self.generations.get(key, 0) + 1
        self.generations[key] = generation
        job = Job(key, generation)
        self.active[key] = job
        return job
    def cancel(self, job):
        job.cancelled = True
        if self.active.get(job.key) is job:
            del self.active[job.key]
    def complete(self, job, value):
        self.cache.put(job.key, value)
        if self.active.get(job.key) is job:
            del self.active[job.key]
class DiskCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
    def put(self, key, value):
        path = self.root / (key + '.json')
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value), encoding='utf-8')
        temp.replace(path)
    def get(self, key):
        return json.loads((self.root / (key + '.json')).read_text())
'''
POOL = '''"""Synchronous bounded runner; retain job input ordering."""
from concurrent.futures import ThreadPoolExecutor
class Runner:
    def __init__(self, workers=2):
        self.workers = workers
    def run(self, functions):
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(fn) for fn in functions]
            return [future.result() for future in futures]
'''
PARSER = '''"""Legacy cache record parser; public return contract must stay stable."""
import json
def parse_record(text):
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('record must be an object')
    if 'key' not in data or not isinstance(data['key'], str):
        raise ValueError('key must be a string')
    if 'value' not in data:
        raise ValueError('value is required')
    return (data['key'], data['value'])
class JsonStore:
    def __init__(self, cache):
        self.cache = cache
    def ingest(self, text):
        key, value = parse_record(text)
        self.cache.put(key, value)
        return key
'''
TESTS = '''import unittest
from tempfile import TemporaryDirectory
from queue_cache import Queue, DiskCache
from runner import Runner
from records import parse_record, JsonStore
class Tests(unittest.TestCase):
    def test_cache_roundtrip(self):
        with TemporaryDirectory() as root:
            cache = DiskCache(root)
            cache.put('report', {'ok': True})
            self.assertEqual(cache.get('report'), {'ok': True})
    def test_completed_job(self):
        with TemporaryDirectory() as root:
            cache = DiskCache(root)
            queue = Queue(cache)
            job = queue.submit('report')
            queue.complete(job, 'new')
            self.assertEqual(cache.get('report'), 'new')
            self.assertNotIn('report', queue.active)
    def test_runner_order(self):
        self.assertEqual(Runner(2).run([lambda: 3, lambda: 7]), [3, 7])
    def test_record_contract(self):
        self.assertEqual(parse_record('{"key":"report","value":4}'), ('report', 4))
        for text in ['[]', '{}', '{"key":5,"value":4}', '{"key":"report"}']:
            with self.assertRaises(ValueError):
                parse_record(text)
if __name__ == '__main__':
    unittest.main()
'''
TASKS = {
 'bugfix': ('Fix stale completion after cancellation without changing the public methods. '
            'Also reject a superseded job even when it was never cancelled. '
            'Give a minimal patch to Queue.complete and one deterministic unittest. '
            'Preserve the newer result and active job.',
            'Observed: submit old report; cancel old; submit new report; complete new with "new"; '
            'complete old with "old". DiskCache.get("report") returns "old", expected "new".'),
 'feature': ('Add optional timeout_seconds to Runner.run, preserving input order and the default behavior. '
             'Treat timeout as a total deadline for the run, cancel pending futures, and do not promise '
             'running threads can be forcibly killed. Provide a small patch and a deterministic test '
             'using an Event to exercise the timeout path.',
             'Issue: Runner(2).run blocks indefinitely when a handler waits forever. '
             'Requirement: return/raise at a bounded deadline; report TimeoutError; no silent partial results.'),
 'refactor': ('Extract JsonStore parsing into a constructor-injected parser callable defaulting to '
              'parse_record. Preserve return key, cache semantics and ValueError messages. '
              'Give a small patch and a regression unittest with a custom parser; do not redesign storage.',
              'Regression report: a proposed storage adapter hardcodes JSON and prevents a CSV parser. '
              'Existing clients rely on JsonStore(cache), ingest(text) returning the key, and parse_record validation.')
}
def sha(b): return hashlib.sha256(b).hexdigest()
def ids(tokenizer,text):
    x=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,
                                    add_generation_prompt=True,enable_thinking=False)
    return list(x['input_ids'] if isinstance(x,Mapping) else x)

def main():
    CORPUS.mkdir(parents=True,exist_ok=False)
    workspace=CORPUS/'workspace';workspace.mkdir()
    for name,body in [('queue_cache.py',QUEUE),('runner.py',POOL),('records.py',PARSER),('test_queue.py',TESTS)]:
        (workspace/name).write_text(body)
    (workspace/'LICENSE').write_text('Copyright 2026 MLTF coding fixture authors\nLicensed under the Apache License, Version 2.0.\nhttps://www.apache.org/licenses/LICENSE-2.0\n')
    (workspace/'README.md').write_text('# Queue/cache coding fixture\n\nOriginal synthetic code for cancellation, deadline and parser-contract work.\nRun `python -m unittest -v` locally. No network dependencies.\n')
    stdlib=Path(sysconfig.get_path('stdlib'))
    (CORPUS/'CPYTHON_LICENSE.txt').write_bytes((stdlib/'LICENSE.txt').read_bytes())
    tokenizer=AutoTokenizer.from_pretrained(MODEL/'tokenizer',local_files_only=True,trust_remote_code=False)
    # Related implementation references, each included at most once per prompt.
    sources=[]
    directories=['asyncio','concurrent/futures','unittest','http','urllib','logging','json','collections','sqlite3','tomllib','importlib']
    names=['queue.py','threading.py','tempfile.py','pathlib.py','argparse.py','dataclasses.py','contextlib.py',
           'functools.py','weakref.py','typing.py','inspect.py','traceback.py','socket.py','socketserver.py',
           'selectors.py','subprocess.py','configparser.py','shlex.py','shutil.py','os.py','glob.py','fnmatch.py',
           'copy.py','abc.py','types.py','enum.py','textwrap.py','pprint.py','heapq.py','bisect.py','_pyio.py','_pydatetime.py']
    files=[]
    for d in directories: files.extend(sorted((stdlib/d).rglob('*.py')))
    files.extend(stdlib/n for n in names if (stdlib/n).is_file())
    blocks=[]
    for p in files:
        raw=p.read_bytes();relative=p.relative_to(stdlib).as_posix()
        text=raw.decode('utf-8');tree=ast.parse(text);lines=text.splitlines(keepends=True)
        starts=[min(n.lineno,*(d.lineno for d in getattr(n,'decorator_list',[]) or [n]))-1 for n in tree.body]
        starts=sorted(set([0]+starts+[len(lines)]))
        for a,b in zip(starts,starts[1:]):
            section=''.join(lines[a:b])
            if not section.strip(): continue
            block=f'\nDependency reference CPython 3.13.14 Lib/{relative}, lines {a+1}-{b}:\n```python\n{section}```\n'
            blocks.append((block,len(tokenizer.encode(block,add_special_tokens=False)),relative,a+1,b))
        sources.append({'path':'Lib/'+relative,'sha256':sha(raw),'bytes':len(raw),'upstream_version':'3.13.14','license':'PSF-2.0 and included notices'})
    manifest={'data_kind':'frozen_coding_corpus','temperature':0,'thinking':False,'output_cap':512,
              'contexts':dict(zip(LABELS,LENGTHS)),'sources':sources,'fixtures':[],
              'license_files':['workspace/LICENSE','CPYTHON_LICENSE.txt'],
              'supporting_source_policy':'Unmodified, distinct related CPython implementation blocks at AST boundaries; no repeated padding.'}
    for family,(instruction,error) in TASKS.items():
      for lane in range(4):
        lane_name=['cedar','maple','birch','elm'][lane]
        core='\n'.join(f'File {n}:\n```python\n{(workspace/n).read_text()}```' for n in ['queue_cache.py','runner.py','records.py','test_queue.py'])
        header=f'Coding review {family}, independent lane {lane_name}. Work on a local Python job queue and disk cache.\n'
        evidence=f'\nTask: {instruction}\nError/requirement log: {error}\nArchitecture: cancellation is advisory; generations identify attempts; storage is local JSON; runners preserve ordering.\n{core}\n'
        suffix='\nRespond in at most 450 tokens with diagnosis, a bounded code patch, and the regression-test idea. Focus on the requested change; do not copy dependency references.\n'
        for label,length in zip(LABELS,LENGTHS):
          # Keep task evidence at fixed family-specific positions, independent of timings.
          def assemble(ref):
            if family=='bugfix':return header+evidence+ref+suffix
            if family=='feature':
                split=ref.find('Dependency reference',len(ref)//2)
                if split<0:split=len(ref)
                return header+ref[:split]+evidence+ref[split:]+suffix
            return header+ref+evidence+suffix
          base_count=len(ids(tokenizer,assemble('')))
          if base_count>length-12:raise RuntimeError(f'core too big: {base_count} for {length}')
          selected=[];estimate=base_count
          for index,block in enumerate(blocks):
            if estimate+block[1]<=length-40:
                selected.append(index);estimate+=block[1]
          ref=''.join(blocks[i][0] for i in selected);text=assemble(ref)
          tokens=ids(tokenizer,text)
          while len(tokens)>length-12 and selected:
            selected.pop();ref=''.join(blocks[i][0] for i in selected);text=assemble(ref);tokens=ids(tokenizer,text)
          # Reconcile token merges, filling gaps with other complete distinct blocks.
          for i,block in enumerate(blocks):
            if i in selected or block[1]>length-len(tokens)-14:continue
            candidate=assemble(ref+block[0]);candidate_tokens=ids(tokenizer,candidate)
            if len(candidate_tokens)<=length-12:
                selected.append(i);ref+=block[0];text=candidate;tokens=candidate_tokens
            if length-len(tokens)<=40:break
          # Fill only the final small residual with a benign manifest comment.
          remainder=length-len(tokens)
          if remainder>160:raise RuntimeError(f'excessive sentinel padding {family}/{label}: {remainder}')
          pad='\n# Fixture budget sentinel: '+(' audit'*max(0,remainder-7))+'\n'
          text+=pad;tokens=ids(tokenizer,text)
          for _ in range(200):
            delta=length-len(tokens)
            if delta==0:break
            if delta>0:text+=' x'
            else:text=text[:-2]
            tokens=ids(tokenizer,text)
          if len(tokens)!=length:raise RuntimeError('exact template count adjustment failed')
          key=f'{label}-{family}-lane{lane}';p=CORPUS/'prompts'/key;p.parent.mkdir(exist_ok=True)
          p.with_suffix('.txt').write_text(text)
          p.with_suffix('.ids').write_text(' '.join(map(str,tokens))+'\n')
          manifest['fixtures'].append({'fixture_id':key,'context_label':label,'input_tokens':length,'family':family,'lane':lane,
              'prompt_sha256':sha(text.encode()),'input_ids_sha256':sha(p.with_suffix('.ids').read_bytes()),
              'sentinel_tokens':remainder,'selected_dependency_blocks':[{'source':blocks[i][2],'first_line':blocks[i][3],'last_line':blocks[i][4]} for i in selected]})
          print(key,length,'sentinel',remainder,flush=True)
    manifest['corpus_sha256']=sha(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode())
    (CORPUS/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('CORPUS_FROZEN',manifest['corpus_sha256'],flush=True)
if __name__=='__main__':main()
