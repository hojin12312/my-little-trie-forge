"""Rebuild public-safe cell statistics from immutable S/D run results."""
import json
import statistics
from pathlib import Path
import run as m
import validate_measurement as v
def stats(values):
    values=[x for x in values if x is not None]
    return {'median':statistics.median(values),'min':min(values),'max':max(values),'n':len(values)} if values else {'median':None,'min':None,'max':None,'n':0}
def ledger():
    method=json.loads((m.ROOT/'METHOD.json').read_text());corpus=json.loads((m.ROOT/'corpus/manifest.json').read_text())
    data={'schema_version':1,'data_kind':'measured_result','near_max':method['near_max'],'artifact_identity':method['artifact'],
          'method_sha256':m.digest(m.ROOT/'METHOD.json'),'corpus_sha256':corpus['corpus_sha256'],
          'sampling':method['sampling'],'recording_on':False,'cells':[]}
    schedule=m.jobs();index=[]
    for label,length in m.LENGTHS.items():
      for c in range(1,5):
        row={'cell_id':f'{label}-C{c}','context_label':label,'input_tokens':length,'C':c,'output_cap':512,
             'sampling':{'temperature':0,'thinking':False,'output_cap':512},'cached_tokens':0,'peak_scope':'fresh_process_lifetime'}
        tracks={}
        for track,name in [('S','natural_serving'),('D','all_ready_decode')]:
          runs=[]
          for job in schedule:
            if job['label']!=label or job['C']!=c or job['track']!=track:continue
            key=f"{label}-C{c}-r{job['repeat']}-{job['family']}-{track}";p=m.ROOT/'runs'/key/'result.json'
            if not p.exists():continue
            raw=json.loads(p.read_text());r={k:x for k,x in raw.items() if k not in ['requests','outputs']}
            r.update(run_id=key,fixture_id=f"{label}-{job['family']}",raw_evidence_ref=f'runs/{key}/result.json')
            r['raw_result_sha256']=m.digest(p);r['response_sanity']={}
            if track=='D' and raw['status']=='PASS':
                events=m.trace(p.parent/'native.jsonl');v.validate_d_window(events,c)
                release=next(e for e in events if e['event']=='ALL_READY_RELEASE')
                r['observed_ready_residents']=release['R']
                memories=m.trace(p.parent/'process-memory.jsonl');pid=json.loads((p.parent/'process.json').read_text())['pid']
                r['sampled_native_footprint_peak_bytes']=max((x['physical_footprint_bytes'] or 0 for e in memories for x in e['process_memory']['processes'] if x['pid']==pid),default=None)
                r['sampled_rss_tree_peak_bytes']=max((e['process_memory']['rss_tree_sum_bytes'] for e in memories),default=None)
            if raw['status']=='PASS':
              if track=='S':
                texts=[r['content'] for r in raw['requests']]
              else:
                from transformers import AutoTokenizer
                if not hasattr(ledger,'tokenizer'):ledger.tokenizer=AutoTokenizer.from_pretrained(m.MODEL/'tokenizer',local_files_only=True,trust_remote_code=False)
                texts=[ledger.tokenizer.decode(o['tokens'],skip_special_tokens=True) for o in raw['outputs']]
              r['response_sanity']={'nonempty':all(t.strip() for t in texts),'has_code_or_diagnosis':all(('```' in t or any(w in t.lower() for w in ['cache','timeout','parser','jsonstore','runner','job'])) for t in texts),
                                    'repeated_text_flag':any(len(t)>100 and len(set(t.split()))<10 for t in texts),'cap_hit':any(x==512 for x in raw['actual_outputs']),
                                    'patch_apply_success_claim':False,'tests_executed_in_matrix':False}
              v.require(r['response_sanity']['nonempty'] and r['response_sanity']['has_code_or_diagnosis'] and not r['response_sanity']['repeated_text_flag'],'quality review required for '+key)
            runs.append(r);index.append({'run_id':key,'sha256':m.digest(p),'status':r['status'],'raw_evidence_ref':r['raw_evidence_ref']})
          required=5 if label=='2Ki' and c in (1,4) else 3
          status='NOT_RUN' if not runs else 'IN_PROGRESS'
          if len(runs)==required:
            states={r['status'] for r in runs}
            status='PASS' if states=={'PASS'} else next(iter(states)) if len(states)==1 else 'MIXED_LIMITATIONS'
          t={'status':status,'runs':runs,'n':len(runs),'required_n':required}
          passed=[r for r in runs if r['status']=='PASS']
          if passed:
            t['prefill_native_tps']=stats([r['prefill_native_tps'] for r in passed])
            t['prefill_elapsed_tps']=stats([r.get('prefill_elapsed_tps') for r in passed])
            t['peak_metal_bytes']=max(r['peak_metal_bytes'] for r in passed)
            t['peak_metal_range_bytes']=[min(r['peak_metal_bytes'] for r in passed),t['peak_metal_bytes']]
            t['actual_outputs']=[r['actual_outputs'] for r in passed]
            if track=='S':
              t['ttft_seconds']=stats([x for r in passed for x in r['ttft_seconds_per_request']])
              t['round_cohort_ttft']=[{'run_id':r['run_id'],'per_request':r['ttft_seconds_per_request'],'median':statistics.median(r['ttft_seconds_per_request']),'max':max(r['ttft_seconds_per_request'])} for r in passed]
              t['full_wave_tps']=stats([r['full_wave_tps'] for r in passed]);t['natural_decode_native_tps']=stats([r['natural_decode_native_tps'] for r in passed])
              t['physical_b_counts']={str(w):sum(r['physical_b_counts'].get(str(w),0) for r in passed) for w in range(1,5)}
              t['physical_b_wall_ms']={str(w):sum(r['physical_b_wall_ms'].get(str(w),0) for r in passed) for w in range(1,5)}
              t['max_ready_residents']=max(r['maximum_ready_residents_observed'] for r in passed)
              t['rss_tree_sampled_peak_bytes']=max((r.get('sampled_rss_tree_peak_bytes') or 0 for r in passed),default=None)
              t['native_footprint_sampled_peak_bytes']=max((r.get('sampled_native_footprint_peak_bytes') or 0 for r in passed),default=None)
            else:
              t['native_decode_tps']=stats([r['native_decode_tps'] for r in passed]);t['elapsed_decode_tps']=stats([r['elapsed_decode_tps'] for r in passed])
              t['accepted_draft_tokens']=sum(r['accepted_draft_tokens'] for r in passed);t['drafted_denominator_tokens']=sum(r['drafted_denominator_tokens'] for r in passed)
              t['acceptance_pct']=100*v.acceptance(t['accepted_draft_tokens'],t['drafted_denominator_tokens'])
              t['confirmed_tokens_per_cycle_per_lane']=sum(r['confirmed_tokens'] for r in passed)/sum(r['selected_cycle_count']*c for r in passed)
              t['observed_physical_b']=c;t['all_ready_verified']=True;t['acceptance_definition']=passed[0]['acceptance_definition']
              t['observed_ready_residents']=min(r['observed_ready_residents'] for r in passed)
              t['rss_tree_sampled_peak_bytes']=max((r.get('sampled_rss_tree_peak_bytes') or 0 for r in passed),default=None)
              t['native_footprint_sampled_peak_bytes']=max((r.get('sampled_native_footprint_peak_bytes') or 0 for r in passed),default=None)
          limited=[r for r in runs if r['status'] in ['MEMORY_LIMITED','READY_RESIDENCY_LIMITED','CONTEXT_UNSUPPORTED']]
          if limited:t['limitation_reason']='; '.join(sorted({r['limitation_reason'] for r in limited}));t['evidence_refs']=[r['raw_evidence_ref'] for r in limited]
          tracks[name]=t
        row.update(tracks);row['sample_count']=min(t['n'] for t in tracks.values());data['cells'].append(row)
    v.coverage(data['cells']);data['accounted_cells']=sum(all(r[name]['status'] in ['PASS','MEMORY_LIMITED','READY_RESIDENCY_LIMITED','CONTEXT_UNSUPPORTED'] for name in ['natural_serving','all_ready_decode']) for r in data['cells'])
    data['raw_run_index']=index;return data
def main():
    data=ledger();m.dump(m.ROOT/'results.json',data)
    print('accounted',data['accounted_cells'],'/24; samples',len(data['raw_run_index']))
if __name__=='__main__':main()
