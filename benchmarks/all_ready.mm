// Bench-only cold independent-lane adapter. No production source is changed.
#include "engine/Types.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/MemoryGovernor.hpp"
#include "model/Runtime.hpp"
#include "model/QwenState.hpp"
#include "metal/MetalBackend.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
using namespace splash;
using namespace splash::engine;
using Clock = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
struct Lane {
  uint64_t id; uint32_t slot; uint64_t position=0;
  std::vector<uint32_t> pages,prompt,output;
  bool done=false,stop=false;
};
int main(int argc,char **argv) {
 @autoreleasepool { try {
  if(argc<6) throw std::invalid_argument("all-ready METALLIB MODEL BUDGET_BYTES OUTPUT_CAP PROMPT_IDS... [--plan-only] [--quiet] [--audit-state]");
  const uint64_t configured=std::stoull(argv[3]);
  const uint32_t cap=std::stoul(argv[4]);
  bool planOnly=false,quiet=false,audit=false;
  std::vector<Lane> lanes;
  for(int i=5;i<argc;i++) {
    const std::string a=argv[i];
    if(a=="--plan-only") {planOnly=true;continue;}
    if(a=="--quiet") {quiet=true;continue;}
    if(a=="--audit-state") {audit=true;continue;}
    Lane l{lanes.size()+1,static_cast<uint32_t>(lanes.size()),0,{},{},{},false,false};
    std::ifstream f(a);uint64_t x;
    while(f>>x) {if(x>UINT32_MAX) throw std::runtime_error("invalid token id");l.prompt.push_back(x);}
    if(l.prompt.empty()) throw std::runtime_error("empty prompt file");
    lanes.push_back(std::move(l));
  }
  if(lanes.empty()||lanes.size()>4||cap!=512) throw std::runtime_error("only B1-B4/cap512 supported");
  const uint32_t length=static_cast<uint32_t>(lanes.front().prompt.size());
  for(auto &l:lanes)if(l.prompt.size()!=length)throw std::runtime_error("lane input length mismatch");
  metal::MetalBackend backend(argv[1]);
  auto package=model::loadModelPackage(backend,std::filesystem::path(argv[2]));
  ops::ExecutionPlans operators(backend.capabilities());
  auto resources=model::plannedRuntimeMemory(backend.capabilities(),package,operators,kv::Format::Int8);
  const uint64_t hard=EngineMemoryPolicy::hardBudgetBytes(backend.capabilities().recommendedMaxWorkingSetBytes,configured);
  const uint64_t reserves=resources.pipelineReserveBytes+resources.runtimeOverheadReserveBytes;
  if(hard<=reserves)throw std::runtime_error("MEMORY_LIMITED: reserves exceed hard budget");
  const uint32_t pagesPerLane=(length+cap+model::ExecutionLimits::targetVerifyRows)/kv::kPageTokens+2;
  if(uint64_t(length)+cap+64>package.maximumContextTokens()||pagesPerLane>kv::kMaximumLogicalTokens/kv::kPageTokens)
    throw std::runtime_error("CONTEXT_UNSUPPORTED: page/speculative headroom");
  const auto layout=package.targetKvLayout(kv::Format::Int8);
  const uint32_t granule=layout.sparseMappingBatchPages();
  const uint32_t count=(pagesPerLane*lanes.size()+granule-1)/granule*granule;
  const uint64_t kvRequired=uint64_t(count)*layout.bytesPerModelPage();
  const auto loaded=backend.memoryStats();
  const uint64_t projected=loaded.allocatedBytes+loaded.sparseResidentBytes+kvRequired+
    resources.activeStateCellPlannedAllocatedBytes*lanes.size()+resources.sharedPrefillPlannedAllocatedBytes+resources.sharedDecodePlannedAllocatedBytes+reserves;
  std::printf("{\"event\":\"plan\",\"input_tokens\":%u,\"C\":%zu,\"cap\":%u,\"pages_per_lane\":%u,\"kv_pages\":%u,\"kv_bytes\":%llu,\"loaded_metal_bytes\":%llu,\"projected_required_bytes\":%llu,\"configured_bytes\":%llu,\"hard_budget_bytes\":%llu,\"reserve_bytes\":%llu,\"headroom_tokens\":64}\n",length,lanes.size(),cap,pagesPerLane,count,(unsigned long long)kvRequired,(unsigned long long)(loaded.allocatedBytes+loaded.sparseResidentBytes),(unsigned long long)projected,(unsigned long long)configured,(unsigned long long)hard,(unsigned long long)reserves);
  std::fflush(stdout);
  if(planOnly) return 0;
  if(projected>hard) throw std::runtime_error("READY_RESIDENCY_LIMITED: independent cold all-ready projection exceeds fixed budget");
  MemoryGovernor governor(backend,hard-reserves,EngineMemoryPolicy::hostAvailableReserveBytes(backend.capabilities().physicalMemoryBytes));
  kv::PageStorage pages(backend,governor.allocationAdmission(),layout,count);
  model::QwenStateStorage states(backend,governor.allocationAdmission(),package.stateLayout());
  model::RuntimeContext context{backend,governor.allocationAdmission(),package,pages,states,operators,16384,resources.pipelineReserveBytes,resources.runtimeOverheadReserveBytes};
  model::Runtime runtime(context);
  auto memory=[&](const char *phase) {
    auto m=backend.memoryStats();auto g=governor.snapshot();
    std::printf("{\"event\":\"memory\",\"phase\":\"%s\",\"at\":%.9f,\"current_metal_bytes\":%llu,\"peak_metal_bytes\":%llu,\"device_peak_bytes\":%llu,\"headroom_bytes\":%llu,\"system_pressure\":\"%s\"}\n",phase,now(),(unsigned long long)(m.allocatedBytes+m.sparseResidentBytes),(unsigned long long)m.peakResidentBytes,(unsigned long long)m.devicePeakAllocatedBytes,(unsigned long long)g.headroomBytes,memoryPressureName(g.systemPressure));std::fflush(stdout);
  };
  memory("loaded");
  (void)runtime.warmupPrefill(2048);
  for(uint32_t w=1;w<=4;w++)(void)runtime.warmupDecodeBatch(w);
  (void)runtime.warmupDraftVerifyCommit();
  while(runtime.reclaimIdleState()){}
  memory("compile_warmup");
  for(auto &l:lanes) {
    l.pages.resize(pagesPerLane);std::iota(l.pages.begin(),l.pages.end(),l.slot*pagesPerLane);
    for(auto page:l.pages)if(!pages.ensureResident(page))throw std::runtime_error("MEMORY_LIMITED: KV backing governor admission denied");
    EngineRequest r;r.id=l.id;r.prompt=l.prompt;r.maxNewTokens=cap;
    runtime.beginColdRequest(r.modelView(),l.slot);
    const uint32_t junction=(length-1)/kv::kPageTokens*kv::kPageTokens;
    const std::array<uint32_t,2> boundaries{junction,length};
    runtime.setDraftContextPlan(l.id,planDraftContext(0,length,std::nullopt,boundaries));
    for(uint32_t offset=0;offset<length;) {
      const uint32_t stopAt=offset<junction?junction:length;
      const uint32_t n=std::min(model::ExecutionLimits::prefillTokenBudget,stopAt-offset);
      BatchPlan p{WorkKind::Prefill,BatchCohort::Greedy,{{l.id,n,offset}},DecodeStage::Regular};
      ModelBatchItem item{l.id,l.slot,offset,offset,n,l.pages};item.inputTokens=std::span<const uint32_t>(l.prompt).subspan(offset,n);
      const double start=now();auto ticket=runtime.submit(p,std::span<const ModelBatchItem>(&item,1),{});auto result=ticket->wait();const double end=now();
      const double nativeWall=ticket->wallMilliseconds()/1000.0;
      if(result.size()!=1||result[0].consumedPromptTokens!=n||!result[0].failure.empty())throw std::runtime_error("prefill count/result mismatch");
      l.output.insert(l.output.end(),result[0].outputTokens.begin(),result[0].outputTokens.end());
      if(result[0].finished)throw std::runtime_error("SHORT_WINDOW: prompt ended before decode");
      if(!quiet)std::printf("{\"event\":\"prefill\",\"lane\":%u,\"offset\":%u,\"input_tokens\":%u,\"begin\":%.9f,\"end\":%.9f,\"wall_seconds\":%.9f,\"call_wall_seconds\":%.9f}\n",l.slot,offset,n,start,end,nativeWall,end-start);
      offset+=n;
    }
    l.position=length;
    const auto state=runtime.debugStateDigest(l.id);
    if(!state||!state->resident||!state->promptComplete||!state->pendingTokenValid||state->targetTokens!=length||!state->draftContextValid||state->draftContextThrough!=length)
      throw std::runtime_error("all-ready state invariant failed");
    std::printf("{\"event\":\"ready\",\"lane\":%u,\"position\":%llu,\"at\":%.9f,\"gdn_recurrent\":%llu,\"draft_keys\":%llu,\"draft_values\":%llu}\n",l.slot,(unsigned long long)l.position,now(),(unsigned long long)state->gdnRecurrent[state->activeParity],(unsigned long long)state->draftKeysWindow,(unsigned long long)state->draftValuesWindow);
    memory("prefill_lane_ready");
  }
  memory("all_ready");
  std::printf("{\"event\":\"ALL_READY_RELEASE\",\"at\":%.9f,\"B\":%zu,\"R\":%zu,\"cached_tokens\":0}\n",now(),lanes.size(),lanes.size());std::fflush(stdout);
  uint64_t cycle=0,totalConfirmed=0,totalDrafted=0,totalAccepted=0;double phaseWall=0;
  bool primary=true;
  while(std::any_of(lanes.begin(),lanes.end(),[](auto &l){return !l.done;})) {
    BatchPlan p;p.kind=WorkKind::Decode;p.cohort=BatchCohort::Greedy;
    std::vector<ModelBatchItem> items;
    for(auto &l:lanes)if(!l.done){p.items.push_back({l.id,0,0});items.push_back({l.id,l.slot,l.position,0,0,l.pages});}
    const double begin=now();auto ticket=runtime.submit(p,items,{});auto result=ticket->wait();const double end=now();
    const double nativeWall=ticket->wallMilliseconds()/1000.0;
    if(result.size()!=items.size()||runtime.telemetry().lastDecodeWidth!=items.size())throw std::runtime_error("physical width mismatch");
    uint64_t confirmed=0,drafted=0,accepted=0;bool completes=false;
    for(size_t i=0;i<result.size();i++) {
      auto &r=result[i];auto &l=lanes[items[i].stateSlot];
      if(!r.failure.empty()||r.requestId!=l.id||r.acceptedDraftTokens>r.draftedTokens)throw std::runtime_error("decode result/counter mismatch");
      l.output.insert(l.output.end(),r.outputTokens.begin(),r.outputTokens.end());
      l.position+=r.outputTokens.size()-r.outputTokensWithoutKv;
      l.done=r.finished||l.output.size()>=cap;l.stop=r.finished;
      completes|=l.done;confirmed+=r.outputTokens.size();drafted+=r.draftedTokens;accepted+=r.acceptedDraftTokens;
      if(l.output.size()>cap)throw std::runtime_error("output cap exceeded");
    }
    const bool selected=primary&&!completes&&items.size()==lanes.size();
    if(!selected)primary=false;
    if(selected){totalConfirmed+=confirmed;totalDrafted+=drafted;totalAccepted+=accepted;phaseWall+=nativeWall;}
    if(!quiet){std::printf("{\"event\":\"decode_cycle\",\"cycle\":%llu,\"begin\":%.9f,\"end\":%.9f,\"wall_seconds\":%.9f,\"call_wall_seconds\":%.9f,\"gpu_seconds\":%.9f,\"B\":%zu,\"confirmed\":%llu,\"drafted\":%llu,\"accepted\":%llu,\"primary\":%s,\"boundary_completion\":%s,\"lane_tokens\":[",(unsigned long long)cycle,begin,end,nativeWall,end-begin,runtime.telemetry().lastDecodeGpuSeconds,items.size(),(unsigned long long)confirmed,(unsigned long long)drafted,(unsigned long long)accepted,selected?"true":"false",completes?"true":"false");
      for(size_t i=0;i<result.size();i++){if(i)std::printf(",");std::printf("{\"lane\":%u,\"tokens\":[",items[i].stateSlot);for(size_t j=0;j<result[i].outputTokens.size();j++)std::printf("%s%u",j?",":"",result[i].outputTokens[j]);std::printf("],\"accepted\":%u,\"drafted\":%u}",result[i].acceptedDraftTokens,result[i].draftedTokens);}std::printf("]}\n");}
    cycle++;
    auto pressure=querySystemMemoryPressure();if(pressure&&*pressure==MemoryPressure::Critical)throw std::runtime_error("MEMORY_LIMITED: critical host pressure safe stop");
  }
  memory("decode_drained");
  std::printf("{\"event\":\"totals\",\"primary_confirmed\":%llu,\"primary_drafted\":%llu,\"primary_accepted\":%llu,\"primary_wall_seconds\":%.9f,\"cycles_including_tail\":%llu}\n",(unsigned long long)totalConfirmed,(unsigned long long)totalDrafted,(unsigned long long)totalAccepted,phaseWall,(unsigned long long)cycle);
  for(auto &l:lanes) {
    if(audit){auto d=runtime.debugStateDigest(l.id);if(!d)throw std::runtime_error("final digest unavailable");std::printf("{\"event\":\"final_state\",\"lane\":%u,\"position\":%llu,\"gdn_recurrent\":%llu,\"gdn_conv\":%llu,\"draft_keys\":%llu,\"draft_values\":%llu,\"pending\":%u}\n",l.slot,(unsigned long long)d->targetTokens,(unsigned long long)d->gdnRecurrent[d->activeParity],(unsigned long long)d->gdnConvolution[d->activeParity],(unsigned long long)d->draftKeysWindow,(unsigned long long)d->draftValuesWindow,d->pendingToken);}
    std::printf("{\"event\":\"output\",\"lane\":%u,\"finish_reason\":\"%s\",\"tokens\":[",l.slot,l.stop?"stop":"length");
    for(size_t j=0;j<l.output.size();j++)std::printf("%s%u",j?",":"",l.output[j]);std::printf("]}\n");runtime.end(l.id);
  }
  while(runtime.reclaimIdleState()){}memory("end_reclaimed");return 0;
 } catch(const std::exception &e){std::cerr<<"all-ready: "<<e.what()<<'\n';return 1;} }
}
