#pragma once

#include "model/ModelFactory.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace splash::model {

class Runtime final : public RuntimeModel {
public:
  explicit Runtime(RuntimeContext context);
  ~Runtime() override;
  void checkHealth() override;
  [[nodiscard]] bool needsHealthCheck() const noexcept override;

  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  // Direct native-oracle entry point. Production admission uses
  // begin() and installs its cache-aware plan explicitly.
  void beginColdRequest(const ModelRequest &request, uint32_t stateSlot);
  [[nodiscard]] StateAdmission
  begin(const ModelRequest &request) override;
  void suspend(uint64_t requestId) override;
  [[nodiscard]] StateAdmission
  resume(const ModelRequest &request) override;
  void restore(uint64_t requestId, uint32_t restoredPrefixLength,
                     std::shared_ptr<const CompositeState> restoredState,
                     bool restoreDraftState) override;
  void setDraftContextPlan(uint64_t requestId, DraftContextPlan plan) override;
  [[nodiscard]] std::vector<ModelStepResult>
  prefill(const BatchPlan &plan, std::span<const ModelBatchItem> items);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion) override;
  [[nodiscard]] std::vector<ModelStepResult>
  decode(const BatchPlan &plan, std::span<const ModelBatchItem> items);
  [[nodiscard]] std::shared_ptr<const CompositeState>
  snapshot(uint64_t requestId) override;
  [[nodiscard]] uint64_t reclaimIdleState() noexcept override;
  void provideMask(uint64_t requestId,
                   std::span<const uint32_t> words) override;
  void end(uint64_t requestId) override;

  [[nodiscard]] WarmupStepResult warmupPrefill(uint32_t rows) override;
  [[nodiscard]] WarmupStepResult
  warmupDecodeBatch(uint32_t width) override;
  [[nodiscard]] WarmupStepResult warmupDraftVerifyCommit() override;
  [[nodiscard]] WarmupStepResult
  warmupCompositeStateRestore() override;
  [[nodiscard]] ModelMemoryActual
  actualRuntimeMemory() const override;

  [[nodiscard]] ModelTelemetry
  telemetry() const noexcept override;
  [[nodiscard]] std::optional<StateDebugDigest>
  debugStateDigest(uint64_t requestId) const override;
  // Development-only teacher-forced evaluation: raw bf16 logits for rows
  // [rowBegin, rowBegin + rows) of this request's last prefill chunk,
  // through the production final norm and logits head, eight rows per
  // command. It overwrites decode lane 0's head buffers, so call it with no
  // command in flight and only for a request that never decodes.
  void debugPrefillLogits(uint64_t requestId, uint32_t rowBegin,
                          uint32_t rows, std::span<uint16_t> logits);
  // Development-only decode-path evaluation: copies the first `rows` rows of
  // the request lane's verify logits buffer after a decode step. Row 0 is the
  // distribution that follows the step's anchor token.
  void debugDecodeLogits(uint64_t requestId, uint32_t rows,
                         std::span<uint16_t> logits);
  // Development-only fixed-token verify evaluation: runs the target's
  // eight-row verify forward on caller-supplied tokens through the real
  // decode/verify kernels and head, then copies the lane's logits. The step
  // does not commit KV or GDN state, so callers re-prefill to advance the
  // committed position between evaluations. `tokens` must hold exactly
  // kDecodeRows ids; token i is placed at `logicalPosition` + i and row i
  // scores the distribution predicting logicalPosition + 1 + i, so the
  // committed state must cover exactly `logicalPosition` tokens (the same
  // targetTokens == logicalPosition contract a real decode step checks).
  // It overwrites decode lane 0's verify buffers, so call it with no command
  // in flight.
  void debugForcedVerify(uint64_t requestId, std::span<const uint32_t> tokens,
                         uint64_t logicalPosition,
                         std::span<const uint32_t> pageTable,
                         std::span<uint16_t> logits);

private:
  void prepareWarmupDecode(uint64_t requestId, uint32_t anchor);
  [[nodiscard]] metal::AllocationResult beginAt(const ModelRequest &request,
                                       uint32_t stateSlot);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  prefillAsync(const BatchPlan &plan, std::span<const ModelBatchItem> items,
               std::function<void()> completion);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  decodeAsync(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
