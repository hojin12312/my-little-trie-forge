"""Raw count/window checks shared by pilot, extraction, and final rendering."""
import math
CONTEXTS={'2Ki':2048,'8Ki':8192,'32Ki':32768,'64Ki':65536,'128Ki':131072,'near256Ki':261568}
EXPECTED={f'{l}-C{c}' for l in CONTEXTS for c in range(1,5)}
def require(ok,message):
    if not ok:raise ValueError(message)
def coverage(rows):
    keys=[r['cell_id'] for r in rows]
    require(len(keys)==len(set(keys)),'duplicate cell')
    require(set(keys)==EXPECTED,'missing or unexpected cell')
def budget(input_tokens,cap,headroom,actual):
    require(input_tokens==actual,'exact input mismatch')
    require(cap==512 and input_tokens+cap+headroom<=262144,'input/output/reserve budget')
def counter_delta(before,after,instance_before,instance_after):
    require(instance_before==instance_after,'mixed native instance')
    delta={k:after[k]-v for k,v in before.items()}
    require(all(math.isfinite(v) and v>=0 for v in delta.values()),'counter reset or negative delta')
    return delta
def acceptance(accepted,drafted,bonus=0,inactive=0):
    require(all(isinstance(x,int) and x>=0 for x in [accepted,drafted,bonus,inactive]),'invalid draft counter')
    require(accepted<=drafted,'accepted exceeds proposal count')
    require(inactive==0,'inactive or padded row in denominator')
    return None if drafted==0 else accepted/drafted
def first_content(events,submitted):
    for e in events:
      for choice in e['event'].get('choices',[]):
        if choice.get('delta',{}).get('content'):
          require(e['at']>=submitted,'negative HTTP interval');return e['at']-submitted
    return None
def validate_d_window(events,c):
    release=[e for e in events if e['event']=='ALL_READY_RELEASE']
    require(len(release)==1,'all-ready release missing')
    require(release[0]['B']==c and release[0]['R']==c,'submitted C is not all-ready B/R')
    require(len([e for e in events if e['event']=='ready'])==c,'independent lane preparation missing')
    cycles=[e for e in events if e['event']=='decode_cycle']
    selected=[e for e in cycles if e['primary']]
    require(len(selected)>=16,'short complete-cycle window')
    require(all(e['B']==c and not e['boundary_completion'] for e in selected),'shape transition or completion in window')
    require([e['cycle'] for e in selected]==list(range(len(selected))),'non-contiguous or cherry-picked cycles')
    require(all(e['begin']>=release[0]['at'] and e['end']>e['begin'] for e in selected),'release/time boundary mismatch')
    require(not any(e['event']=='prefill' and e['end']>release[0]['at'] for e in events),'prefill overlaps decode window')
    wall=sum(e['wall_seconds'] for e in selected)
    require(wall>=1,'short wall-time window')
    for e in selected:
      call=e.get('call_wall_seconds',e['wall_seconds'])
      require(abs(call-(e['end']-e['begin']))<2e-8,'wall interval mismatch')
      require(0<e['wall_seconds']<=call+2e-8,'native phase exceeds call wall interval')
      require(e['accepted']<=e['drafted'],'accepted exceeds proposed count')
      require(len(e['lane_tokens'])==c,'inactive/padded lane')
      require(e['confirmed']==sum(len(r['tokens']) for r in e['lane_tokens']),'confirmed count mismatch')
      require(e['drafted']==sum(r['drafted'] for r in e['lane_tokens']),'proposal count mismatch')
      require(e['accepted']==sum(r['accepted'] for r in e['lane_tokens']),'accepted count mismatch')
    return selected
def final_input(data):
    require(data.get('data_kind')=='measured_result','dummy or pending input is not final eligible')
    coverage(data['cells'])
    for row in data['cells']:
        require(row['sampling']=={'temperature':0,'thinking':False,'output_cap':512},'incompatible historical series')
        require(row['cached_tokens']==0,'cold cache contaminated')
        require(row['peak_scope'] in ['fresh_process_lifetime','request_phase_sampled'],'unlabeled or mixed peak scope')
def phase_rates(input_tokens,output_tokens,pp_seconds,d_seconds,wave_seconds):
    require(min(pp_seconds,d_seconds,wave_seconds)>0,'invalid rate denominator')
    return {'PP_native':input_tokens/pp_seconds,'D_native':output_tokens/d_seconds,'full_wave':output_tokens/wave_seconds}
def summarize_statuses(rows):
    counts={}
    for r in rows:counts[r['status']]=counts.get(r['status'],0)+1
    return counts
def bilingual_equal(en,ko):
    require(en==ko,'bilingual numeric/unit/status mismatch')
