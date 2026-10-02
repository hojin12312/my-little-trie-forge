#include "model/QwenTarget.hpp"

#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace splash::model {
namespace {

template <class Layout>
QwenTargetGeometry commonGeometry(const Layout &layout) {
  QwenTargetGeometry result;
  result.maximumContextTokens = layout.maximumContextTokens;
  result.layers = layout.layers;
  result.hiddenSize = layout.hiddenSize;
  result.vocabularySize = layout.vocabularySize;
  result.packedGdnWidth = layout.packedGdnWidth;
  result.packedAttentionWidth = layout.packedFullWidth;
  result.convolutionDimension = layout.convolutionDimension;
  result.attentionWidth = layout.attentionWidth;
  result.attentionQueryHeads = layout.attentionQueryHeads;
  result.attentionKvHeads = layout.attentionKvHeads;
  result.attentionHeadDimension = layout.attentionHeadDimension;
  result.rotaryPairs = layout.rotaryPairs;
  result.rotaryTheta = layout.rotaryTheta;
  result.gdnKeyHeads = layout.gdnKeyHeads;
  result.gdnValueHeads = layout.gdnValueHeads;
  result.gdnHeadDimension = layout.gdnHeadDimension;
  result.maskToken = layout.maskToken;
  result.stopTokens = layout.stopTokens;
  result.kvLayout = layout.kvLayout();
  result.stateLayout = layout.gdnStateLayout();
  result.captureLayerCount =
      static_cast<uint32_t>(layout.hiddenCaptureLayers.size());
  std::copy(layout.hiddenCaptureLayers.begin(),
            layout.hiddenCaptureLayers.end(),
            result.captureLayerValues.begin());
  return result;
}

QwenTargetGeometry geometryFor(const Qwen3_8Layout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  result.ffnKind = QwenFfnKind::Dense;
  return result;
}

QwenTargetGeometry geometryFor(const Qwen3_6MoeLayout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.moe = {layout.hiddenSize, layout.experts, layout.expertsPerToken,
                layout.expertIntermediateSize};
  result.ffnKind = QwenFfnKind::SparseMoe;
  return result;
}

template <class Weights>
void requireWeights(const Weights &weights,
                    const QwenTargetGeometry &geometry) {
  const uint32_t attentionLayers = static_cast<uint32_t>(std::count_if(
      weights.layers.begin(), weights.layers.end(), [](const auto &layer) {
        return std::holds_alternative<QwenAttentionWeights>(layer.mixer);
      }));
  if (!geometry.valid() || weights.layers.size() != geometry.layers ||
      attentionLayers != geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument(
        "Qwen target weights do not match execution geometry");
  }
}

template <class Layer>
constexpr bool hasDenseFfn = requires(const Layer &layer) {
  layer.gateProjection;
  layer.upProjection;
  layer.downProjection;
};

template <class Mixer>
constexpr bool isGdnMixer =
    std::is_same_v<std::remove_cvref_t<Mixer>, QwenGdnWeights>;

// Q4/Q8 dispatch for dense projections. Q8 prefill runs the installed
// activation: A8 consumes the int8 codes and float2 terms the fused norms and
// one quantize pass write into the model's Q8 scratch, A16 consumes the fp32
// group sums the Q4 path already prepares. Decode is A16 for both packings and
// needs no side channel, so the graph shares its buffers.
class DenseLinear final {
public:
  DenseLinear(const ops::ExecutionPlans &operators, bool quantized8)
      : operators_(operators), quantized8_(quantized8) {}

  // True when the installed Q8 policy quantizes prefill activations, so the
  // caller prepares the Q8 scratch instead of the fp32 sums.
  [[nodiscard]] bool prefillActivation8() const noexcept {
    return quantized8_ && operators_.linearQ8().prefillActivation8();
  }

  // Prepares a projection input that is an RMS normalization output: the
  // fused norm plus the side channel the installed plan consumes.
  void addPrefillNorm(metal::CommandGraph &graph, metal::MetalBuffer input,
                      metal::MetalBuffer weight, metal::MetalBuffer output,
                      metal::MetalBuffer sums, ops::Q8Scratch scratch,
                      ops::LinearMatrix matrix, uint32_t rows) const {
    if (prefillActivation8())
      operators_.linearQ8().addPrefillNormQuantize(graph, input, weight, output,
                                                   scratch, matrix, rows);
    else
      ops::Normalization::addRmsWithQ4Sums(graph, input, weight, output, sums,
                                           matrix.inputSize, rows);
  }

  // Prepares a projection input that is not a normalization output.
  void addPrefillInput(metal::CommandGraph &graph, metal::MetalBuffer input,
                       metal::MetalBuffer sums, ops::Q8Scratch scratch,
                       ops::LinearMatrix matrix, uint32_t rows) const {
    if (prefillActivation8())
      operators_.linearQ8().addPrefillQuantize(graph, input, scratch, matrix,
                                               rows);
    else
      operators_.linear().addPrefillSums(graph, input, sums, matrix, rows);
  }

  void addPrefill(metal::CommandGraph &graph, metal::MetalBuffer activation,
                  const ops::DenseProjection &projection,
                  metal::MetalBuffer output, metal::MetalBuffer sums,
                  ops::Q8Scratch scratch, ops::LinearMatrix matrix,
                  uint32_t rows) const {
    if (quantized8_)
      operators_.linearQ8().addPrefill(graph, prefillInput(activation, scratch),
                                       projection.q8, output, sums, scratch,
                                       matrix, rows);
    else
      operators_.linear().addPrefill(graph, activation, projection.q4, output,
                                     sums, matrix, rows);
  }
  void addPrefillResidual(metal::CommandGraph &graph,
                          metal::MetalBuffer activation,
                          const ops::DenseProjection &projection,
                          metal::MetalBuffer residual,
                          metal::MetalBuffer output, metal::MetalBuffer sums,
                          ops::Q8Scratch scratch, ops::LinearMatrix matrix,
                          uint32_t rows) const {
    if (quantized8_)
      operators_.linearQ8().addPrefillResidual(
          graph, prefillInput(activation, scratch), projection.q8, residual,
          output, sums, scratch, matrix, rows);
    else
      operators_.linear().addPrefillResidual(graph, activation, projection.q4,
                                             residual, output, sums, matrix,
                                             rows);
  }
  void addPrefillUpWithGate(metal::CommandGraph &graph,
                            metal::MetalBuffer activation,
                            const ops::DenseProjection &up,
                            metal::MetalBuffer gateScratch,
                            metal::MetalBuffer output, metal::MetalBuffer sums,
                            metal::MetalBuffer downSums,
                            ops::Q8Scratch scratch, ops::Q8Scratch downScratch,
                            ops::LinearMatrix matrix, uint32_t rows) const {
    if (quantized8_)
      operators_.linearQ8().addPrefillUpWithGate(
          graph, prefillInput(activation, scratch), up.q8, gateScratch, output,
          sums, downSums, scratch, downScratch, matrix, rows);
    else
      operators_.linear().addPrefillUpWithGate(graph, activation, up.q4,
                                               gateScratch, output, sums,
                                               downSums, matrix, rows);
  }
  void addDecode(metal::CommandGraph &graph, metal::MetalBuffer input,
                 const ops::DenseProjection &projection,
                 metal::MetalBuffer output, ops::LinearMatrix matrix,
                 ops::LinearScratch scratch = {}) const {
    if (quantized8_)
      operators_.linearQ8().addDecode(graph, input, projection.q8, output,
                                      matrix, {});
    else
      operators_.linear().addDecode(graph, input, projection.q4, output, matrix,
                                    scratch);
  }
  void addDecodeBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
                      const ops::DenseProjection &projection,
                      metal::MetalBuffer output, ops::LinearMatrix matrix,
                      uint32_t lanes, ops::Q4DispatchStats &stats,
                      ops::LinearScratch scratch = {},
                      bool inputPrepared = false) const {
    if (quantized8_)
      operators_.linearQ8().addDecodeBatch(graph, input, projection.q8, output,
                                           matrix, lanes, stats, {},
                                           inputPrepared);
    else
      operators_.linear().addDecodeBatch(graph, input, projection.q4, output,
                                         matrix, lanes, stats, scratch,
                                         inputPrepared);
  }
  void addResidualBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
                        const ops::DenseProjection &projection,
                        metal::MetalBuffer residual, metal::MetalBuffer output,
                        ops::LinearMatrix matrix, uint32_t lanes,
                        ops::Q4DispatchStats &stats,
                        ops::LinearScratch scratch = {},
                        bool inputPrepared = false) const {
    if (quantized8_)
      operators_.linearQ8().addResidualBatch(graph, input, projection.q8,
                                             residual, output, matrix, lanes,
                                             stats, {scratch.partials, {}}, inputPrepared);
    else
      operators_.linear().addResidualBatch(graph, input, projection.q4,
                                           residual, output, matrix, lanes,
                                           stats, scratch, inputPrepared);
  }
  void addGateUpBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
                      const ops::DenseProjection &gate,
                      const ops::DenseProjection &up,
                      metal::MetalBuffer gateScratch,
                      metal::MetalBuffer output, ops::LinearMatrix matrix,
                      uint32_t lanes, ops::Q4DispatchStats &stats,
                      ops::LinearScratch scratch = {},
                      bool inputPrepared = false) const {
    if (quantized8_)
      operators_.linearQ8().addGateUpBatch(graph, input, gate.q8, up.q8,
                                           gateScratch, output, matrix, lanes,
                                           stats, {}, inputPrepared);
    else
      operators_.linear().addGateUpBatch(graph, input, gate.q4, up.q4,
                                         gateScratch, output, matrix, lanes,
                                         stats, scratch, inputPrepared);
  }

private:
  // The bf16 activation or the quantized codes the installed plan consumes.
  [[nodiscard]] metal::MetalBuffer prefillInput(
      metal::MetalBuffer activation, ops::Q8Scratch scratch) const {
    return prefillActivation8() ? scratch.input : activation;
  }
  const ops::ExecutionPlans &operators_;
  bool quantized8_ = false;
};

} // namespace

QwenMixerWeights readQwenMixer(WeightFile &file, metal::MetalBackend &backend,
                               const QwenMixerGeometry &geometry,
                               bool fullAttention, bool quantized8,
                               ops::Q8Parameters parameters) {
  constexpr uint64_t kFloat32Bytes = 4;
  if (fullAttention) {
    QwenAttentionWeights attention;
    attention.inputProjection = readDenseProjection(
        file, backend, geometry.packedAttentionWidth, geometry.hiddenSize,
        "attention-input", quantized8, parameters);
    const uint64_t headNormBytes = checkedWeightMultiply(
        geometry.attentionHeadDimension, kBFloat16Bytes, "head norm bytes");
    attention.queryNorm = file.section(headNormBytes, "query-norm");
    attention.keyNorm = file.section(headNormBytes, "key-norm");
    attention.outputProjection = readDenseProjection(
        file, backend, geometry.hiddenSize, geometry.attentionWidth,
        "attention-output", quantized8, parameters);
    return attention;
  }
  QwenGdnWeights gdn;
  gdn.inputProjection = readDenseProjection(
      file, backend, geometry.packedGdnWidth, geometry.hiddenSize, "gdn-input",
      quantized8, parameters);
  gdn.convolutionWeights = file.section(
      checkedWeightMultiply(
          checkedWeightMultiply(geometry.convolutionDimension, kGdnConvolutionTaps,
                                "convolution elements"),
          kBFloat16Bytes, "convolution bytes"),
      "gdn-convolution");
  gdn.decay = file.section(checkedWeightMultiply(geometry.gdnValueHeads,
                                                 kFloat32Bytes,
                                                 "GDN decay bytes"),
                           "gdn-decay");
  gdn.timeBias = file.section(
      checkedWeightMultiply(geometry.gdnValueHeads, kBFloat16Bytes,
                            "GDN time bias bytes"),
      "gdn-time-bias");
  gdn.mixerNorm = file.section(
      checkedWeightMultiply(geometry.gdnHeadDimension, kBFloat16Bytes,
                            "GDN norm bytes"),
      "gdn-norm");
  gdn.outputProjection = readDenseProjection(
      file, backend, geometry.hiddenSize, geometry.attentionWidth,
      "gdn-output", quantized8, parameters);
  return gdn;
}

QwenTarget::QwenTarget(const Qwen3_8Weights &weights,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators, kv::Format format)
    : weights_(&weights), geometry_(qwenTargetGeometry(weights)),
      backend_(backend), operators_(operators) {
  geometry_.kvLayout.format = format;
  requireWeights(weights, geometry_);
}

QwenTarget::QwenTarget(const Qwen3_6MoeWeights &weights,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators, kv::Format format)
    : weights_(&weights), geometry_(qwenTargetGeometry(weights)),
      backend_(backend), operators_(operators) {
  geometry_.kvLayout.format = format;
  requireWeights(weights, geometry_);
}

QwenTargetGeometry qwenTargetGeometry(const Qwen3_8Weights &weights) {
  QwenTargetGeometry result = geometryFor(weights.layout);
  result.denseQuantized8 = weights.logitsProjection.quantized8;
  return result;
}

QwenTargetGeometry qwenTargetGeometry(const Qwen3_6MoeWeights &weights) {
  QwenTargetGeometry result = geometryFor(weights.layout);
  result.denseQuantized8 = weights.logitsProjection.quantized8;
  return result;
}

const ops::DenseProjection &QwenTarget::vocabularyProjection() const noexcept {
  return std::visit([](const auto *weights) -> const ops::DenseProjection & {
    return weights->logitsProjection;
  }, weights_);
}

void QwenTarget::addPrefill(
    metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const kv::LayerStorage> kvLayers) const {
  std::visit(
      [&](const auto *weights) {
        addPrefillImpl(*weights, graph, std::move(buffers), sequences, rows,
                       kvLayers);
      },
      weights_);
}

template <class Weights>
void QwenTarget::addPrefillImpl(
    const Weights &weights, metal::CommandGraph &graph,
    QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const kv::LayerStorage> kvLayers) const {
  if (sequences.empty() ||
      sequences.size() > ExecutionLimits::maximumBatchWidth || !rows ||
      rows > ExecutionLimits::prefillTokenBudget ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen packed prefill batch");
  }
  for (const QwenTargetPrefillSequence &sequence : sequences) {
    if (sequence.convolutionIn.size() != geometry_.stateLayout.layers ||
        sequence.convolutionOut.size() != geometry_.stateLayout.layers ||
        sequence.recurrentIn.size() != geometry_.stateLayout.layers ||
        sequence.recurrentOut.size() != geometry_.stateLayout.layers) {
      throw std::invalid_argument("Qwen prefill state layer mismatch");
    }
  }
  const ops::LinearMatrix gdnInput{geometry_.packedGdnWidth,
                                     geometry_.hiddenSize};
  const ops::LinearMatrix attentionInput{geometry_.packedAttentionWidth,
                                           geometry_.hiddenSize};
  const ops::LinearMatrix mixerOutput{geometry_.hiddenSize,
                                        geometry_.attentionWidth};
  const DenseLinear dense(operators_, geometry_.denseQuantized8);
  const auto moePlan = [&]() -> std::optional<ops::MoePlan> {
    if constexpr (!hasDenseFfn<typename std::remove_cvref_t<decltype(weights.layers)>::value_type>)
      return operators_.moePrefill(geometry_.moe, rows);
    return std::nullopt;
  }();

  auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                 uint32_t count, uint32_t width) {
    return backend_.view(buffer, uint64_t{begin} * width * sizeof(uint16_t),
                         uint64_t{count} * width * sizeof(uint16_t));
  };
  auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                 uint32_t count, uint32_t width) {
    return backend_.view(buffer, uint64_t{begin} * width * sizeof(float),
                         uint64_t{count} * width * sizeof(float));
  };

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;
  for (uint32_t layerIndex = 0; layerIndex < geometry_.layers; ++layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];

    metal::MetalBuffer residual;
    std::visit(
        [&](const auto &mixer) {
          if constexpr (isGdnMixer<decltype(mixer)>) {
            dense.addPrefillNorm(graph, input, layer.inputNorm,
                                 buffers.normalized, buffers.projectionSums,
                                 buffers.q8Input, gdnInput, rows);
            dense.addPrefill(graph, buffers.normalized,
                           mixer.inputProjection, buffers.gdnPacked,
                           buffers.projectionSums, buffers.q8Input, gdnInput,
                           rows);
            for (const QwenTargetPrefillSequence &sequence : sequences) {
              ops::GDN::addPrefill(
                  graph,
                  {u16(buffers.gdnPacked, sequence.rowBegin, sequence.rows,
                       geometry_.packedGdnWidth),
                   mixer.convolutionWeights, sequence.convolutionIn[gdnIndex],
                   sequence.convolutionOut[gdnIndex],
                   u16(buffers.gdnQueries, sequence.rowBegin, sequence.rows,
                       geometry_.gdnKeyWidth()),
                   u16(buffers.gdnKeys, sequence.rowBegin, sequence.rows,
                       geometry_.gdnKeyWidth()),
                   u16(buffers.gdnValues, sequence.rowBegin, sequence.rows,
                       geometry_.attentionWidth),
                   mixer.decay, mixer.timeBias,
                   f32(buffers.gdnDecay, sequence.rowBegin, sequence.rows,
                       geometry_.gdnValueHeads),
                   u16(buffers.gdnBeta, sequence.rowBegin, sequence.rows,
                       geometry_.gdnValueHeads),
                   sequence.recurrentIn[gdnIndex],
                   sequence.recurrentOut[gdnIndex],
                   u16(buffers.recurrent, sequence.rowBegin, sequence.rows,
                       geometry_.attentionWidth),
                   mixer.mixerNorm,
                   u16(buffers.gdnHidden, sequence.rowBegin, sequence.rows,
                       geometry_.attentionWidth)},
                  geometry_.gdnShape(), sequence.rows);
            }
            dense.addPrefillInput(graph, buffers.gdnHidden,
                               buffers.projectionSums, buffers.q8Input,
                               mixerOutput, rows);
            dense.addPrefillResidual(
                graph, buffers.gdnHidden, mixer.outputProjection, input,
                buffers.gdnOutput, buffers.projectionSums, buffers.q8Input,
                mixerOutput, rows);
            residual = buffers.gdnOutput;
            ++gdnIndex;
          } else {
            dense.addPrefillNorm(graph, input, layer.inputNorm,
                                 buffers.normalized, buffers.projectionSums,
                                 buffers.q8Input, attentionInput, rows);
            dense.addPrefill(graph, buffers.normalized,
                           mixer.inputProjection, buffers.fullPacked,
                           buffers.projectionSums, buffers.q8Input,
                           attentionInput, rows);
            for (const QwenTargetPrefillSequence &sequence : sequences) {
              const uint64_t queryBytes =
                  uint64_t{geometry_.attentionQueryHeads} *
                  sequence.attentionStride * geometry_.attentionHeadDimension *
                  sizeof(uint16_t);
              const uint64_t kvBytes =
                  uint64_t{geometry_.attentionKvHeads} *
                  sequence.attentionStride * geometry_.attentionHeadDimension *
                  sizeof(uint16_t);
              metal::MetalBuffer queries = backend_.view(
                  buffers.fullQueries, sequence.queryOffset, queryBytes);
              metal::MetalBuffer attentionRows = backend_.view(
                  buffers.fullAttention, sequence.queryOffset, queryBytes);
              metal::MetalBuffer keys = backend_.view(
                  buffers.chunkKeys, sequence.kvOffset, kvBytes);
              metal::MetalBuffer values = backend_.view(
                  buffers.chunkValues, sequence.kvOffset, kvBytes);
              ops::PagedAttention::addPrefillProjection(
                  graph,
                  u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                      geometry_.packedAttentionWidth),
                  mixer.queryNorm, mixer.keyNorm,
                  f32(buffers.ropeCos, sequence.rowBegin, sequence.rows,
                      geometry_.rotaryPairs),
                  f32(buffers.ropeSin, sequence.rowBegin, sequence.rows,
                      geometry_.rotaryPairs),
                  queries, keys, values, sequence.rows,
                  sequence.attentionStride, sequence.attentionStride,
                  geometry_.attentionQueryHeads, geometry_.kvLayout);
              ops::PagedAttention::addPrefillStore(
                  graph, kvLayers[attentionIndex], keys, values,
                  sequence.pageTable, sequence.q8, geometry_.kvLayout);
              ops::PagedAttention::addPrefill(
                  graph, kvLayers[attentionIndex], queries, attentionRows,
                  buffers.attentionPartials, buffers.attentionStatistics,
                  sequence.pageTable, sequence.q8,
                  operators_.prefillAttention(
                      sequence.rows, geometry_.attentionQueryHeads,
                      geometry_.kvLayout, sequence.q8.committed_tokens));
              ops::PagedAttention::addPrefillGate(
                  graph,
                  u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                      geometry_.packedAttentionWidth),
                  attentionRows,
                  u16(buffers.attentionHidden, sequence.rowBegin,
                      sequence.rows, geometry_.attentionWidth),
                  sequence.rows, sequence.attentionStride,
                  sequence.attentionStride, geometry_.attentionQueryHeads,
                  geometry_.kvLayout);
            }
            dense.addPrefillInput(graph, buffers.attentionHidden,
                               buffers.projectionSums, buffers.q8Input,
                               mixerOutput, rows);
            dense.addPrefillResidual(
                graph, buffers.attentionHidden, mixer.outputProjection, input,
                buffers.attentionOutput, buffers.projectionSums,
                buffers.q8Input, mixerOutput, rows);
            residual = buffers.attentionOutput;
            ++attentionIndex;
          }
        },
        layer.mixer);

    if constexpr (hasDenseFfn<std::remove_cvref_t<decltype(layer)>>) {
      const ops::LinearMatrix up{geometry_.denseIntermediateSize,
                                   geometry_.hiddenSize};
      const ops::LinearMatrix down{geometry_.hiddenSize,
                                     geometry_.denseIntermediateSize};
      dense.addPrefillNorm(graph, residual, layer.postAttentionNorm,
                           buffers.normalized, buffers.projectionSums,
                           buffers.q8Input, up, rows);
      dense.addPrefill(graph, buffers.normalized, layer.gateProjection,
                     buffers.denseGateScratch, buffers.projectionSums,
                     buffers.q8Input, up, rows);
      dense.addPrefillUpWithGate(
          graph, buffers.normalized, layer.upProjection,
          buffers.denseGateScratch, buffers.denseIntermediate,
          buffers.projectionSums, buffers.downProjectionSums,
          buffers.q8Input, buffers.q8Down, up, rows);
      dense.addPrefillResidual(
          graph, buffers.denseIntermediate, layer.downProjection, residual,
          output, buffers.downProjectionSums, buffers.q8Down, down, rows);
    } else {
      ops::Normalization::addRmsWithQ4Sums(
          graph, residual, layer.postAttentionNorm, buffers.normalized,
          buffers.projectionSums, geometry_.hiddenSize, rows);
      ops::MoE::add(
          graph,
          {buffers.normalized, residual, output, buffers.selectedExperts,
           buffers.routingWeights, buffers.tileDescriptors, buffers.tileCount,
           buffers.groupedRoutes, buffers.routeRows, buffers.groupedInput,
           buffers.expertIntermediate, buffers.expertOutput},
          layer.ffn, *moePlan);
    }

    const auto captureLayers = geometry_.captureLayers();
    const auto captured =
        std::find(captureLayers.begin(), captureLayers.end(), layerIndex);
    if (captured != captureLayers.end()) {
      const uint32_t slot =
          static_cast<uint32_t>(captured - captureLayers.begin());
      for (const QwenTargetPrefillSequence &sequence : sequences) {
        for (uint32_t index = 0; index < sequence.captureCount; ++index) {
          const QwenTargetPrefillCapture &capture = sequence.captures[index];
          ops::DraftAttention::captureTargetHidden(
              graph, output, buffers.captured, capture.rows, slot,
              capture.sourceStart, capture.destinationStart,
              geometry_.hiddenSize, geometry_.capturedHiddenSize());
        }
      }
    }
  }
  if (gdnIndex != geometry_.stateLayout.layers ||
      attentionIndex != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }
}

void QwenTarget::addVerify(
    metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
    std::span<const kv::LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
    ops::Q4DispatchStats &stats) const {
  std::visit(
      [&](const auto *weights) {
        addVerifyImpl(*weights, graph, std::move(buffers), kvLayers, q8,
                      verify, lanes, stats);
      },
      weights_);
}

template <class Weights>
void QwenTarget::addVerifyImpl(
    const Weights &weights, metal::CommandGraph &graph,
    QwenTargetVerifyBuffers buffers,
    std::span<const kv::LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
    ops::Q4DispatchStats &stats) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      q8.size() != ExecutionLimits::maximumBatchWidth ||
      verify.size() != ExecutionLimits::maximumBatchWidth ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.gdnPacked.size() != geometry_.stateLayout.layers ||
      buffers.gdnMixed.size() != geometry_.stateLayout.layers ||
      buffers.gdnDecay.size() != geometry_.stateLayout.layers ||
      buffers.gdnBeta.size() != geometry_.stateLayout.layers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen verify batch");
  }
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = verify[lane].committed_tokens;
  const auto attentionPlan = operators_.verifyAttention(
      lanes, geometry_.attentionQueryHeads, geometry_.kvLayout, histories);
  const ops::LinearMatrix gdnInput{geometry_.packedGdnWidth, geometry_.hiddenSize};
  const ops::LinearMatrix attentionInput{geometry_.packedAttentionWidth, geometry_.hiddenSize};
  const ops::LinearMatrix mixerOutput{geometry_.hiddenSize, geometry_.attentionWidth};
  const DenseLinear dense(operators_, geometry_.denseQuantized8);
  const auto moePlan = [&]() -> std::optional<ops::MoePlan> {
    if constexpr (!hasDenseFfn<typename std::remove_cvref_t<decltype(weights.layers)>::value_type>)
      return operators_.moeDecode(geometry_.moe, lanes);
    return std::nullopt;
  }();
  constexpr uint32_t tileRows = kv::kPageTokens;

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;
  for (uint32_t layerIndex = 0; layerIndex < geometry_.layers; ++layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];
    ops::Normalization::addRms(graph, input, layer.inputNorm,
                               buffers.normalized, geometry_.hiddenSize, rows, buffers.linearScratch);

    metal::MetalBuffer residual;
    std::visit(
        [&](const auto &mixer) {
          if constexpr (isGdnMixer<decltype(mixer)>) {
            dense.addDecodeBatch(graph,
                               buffers.normalized, mixer.inputProjection,
                               buffers.gdnPacked[gdnIndex], gdnInput, lanes,
                               stats, buffers.linearScratch, true);
            ops::GDN::addDecode(
                graph,
                {buffers.gdnPacked[gdnIndex], mixer.convolutionWeights,
                 buffers.currentGdnStates, buffers.nextGdnStates,
                 buffers.gdnMixed[gdnIndex], mixer.decay, mixer.timeBias,
                 buffers.gdnDecay[gdnIndex], buffers.gdnBeta[gdnIndex],
                 buffers.recurrent, mixer.mixerNorm, buffers.gdnHidden,
                 buffers.arrived, buffers.generation, buffers.linearScratch},
                geometry_.gdnShape(), lanes, gdnIndex,
                {geometry_.stateLayout.convolutionLayerBytes(),
                 geometry_.stateLayout.recurrentLayerBytes(),
                 geometry_.stateLayout.convolutionBytes()});
            dense.addResidualBatch(
                graph, buffers.gdnHidden,
                mixer.outputProjection, input, buffers.gdnOutput, mixerOutput,
                lanes, stats, buffers.linearScratch, true);
            residual = buffers.gdnOutput;
            ++gdnIndex;
          } else {
            dense.addDecodeBatch(graph,
                               buffers.normalized, mixer.inputProjection,
                               buffers.fullPacked, attentionInput, lanes,
                               stats, buffers.linearScratch, true);
            ops::PagedAttention::addVerifyProjection(
                graph, buffers.fullPacked, mixer.queryNorm, mixer.keyNorm,
                buffers.ropeCos, buffers.ropeSin, buffers.fullQueries,
                buffers.chunkKeys[attentionIndex],
                buffers.chunkValues[attentionIndex],
                ExecutionLimits::targetVerifyRows, tileRows, tileRows,
                geometry_.attentionQueryHeads, geometry_.kvLayout, lanes);
            ops::PagedAttention::addVerify(
                graph, kvLayers[attentionIndex],
                {buffers.chunkKeys[attentionIndex],
                 buffers.chunkValues[attentionIndex], buffers.fullQueries,
                 buffers.attentionPartials, buffers.attentionStatistics,
                 buffers.fullAttention, buffers.pageTables},
                q8, verify, attentionPlan);
            ops::PagedAttention::addVerifyGate(
                graph, buffers.fullPacked, buffers.fullAttention,
                buffers.attentionHidden, ExecutionLimits::targetVerifyRows,
                tileRows, tileRows, geometry_.attentionQueryHeads,
                geometry_.kvLayout, lanes, buffers.linearScratch);
            dense.addResidualBatch(
                graph, buffers.attentionHidden,
                mixer.outputProjection, input, buffers.attentionOutput,
                mixerOutput, lanes, stats, buffers.linearScratch, true);
            residual = buffers.attentionOutput;
            ++attentionIndex;
          }
        },
        layer.mixer);

    ops::Normalization::addRms(graph, residual, layer.postAttentionNorm,
                               buffers.normalized, geometry_.hiddenSize, rows, buffers.linearScratch);
    if constexpr (hasDenseFfn<std::remove_cvref_t<decltype(layer)>>) {
      const ops::LinearMatrix up{geometry_.denseIntermediateSize, geometry_.hiddenSize};
      const ops::LinearMatrix down{geometry_.hiddenSize, geometry_.denseIntermediateSize};
      dense.addGateUpBatch(graph, buffers.normalized, layer.gateProjection,
                         layer.upProjection, buffers.denseGateScratch,
                         buffers.denseIntermediate, up, lanes, stats, buffers.linearScratch, true);
      dense.addResidualBatch(
          graph, buffers.denseIntermediate,
          layer.downProjection, residual, output, down, lanes, stats, buffers.linearScratch);
    } else {
      ops::MoE::add(
          graph,
          {buffers.normalized, residual, output, buffers.selectedExperts,
           buffers.routingWeights, buffers.tileDescriptors, buffers.tileCount,
           buffers.groupedRoutes, buffers.routeRows, buffers.groupedInput,
           buffers.expertIntermediate, buffers.expertOutput},
          layer.ffn, *moePlan);
    }

    const auto captureLayers = geometry_.captureLayers();
    const auto captured =
        std::find(captureLayers.begin(), captureLayers.end(), layerIndex);
    if (captured != captureLayers.end()) {
      ops::DraftAttention::captureTargetHidden(
          graph, output, buffers.capturedTargetHidden, rows,
          static_cast<uint32_t>(captured - captureLayers.begin()), 0, 0,
          geometry_.hiddenSize, geometry_.capturedHiddenSize());
    }
  }
  if (gdnIndex != geometry_.stateLayout.layers ||
      attentionIndex != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }

  ops::Normalization::addRms(graph, buffers.hidden[geometry_.layers & 1],
                             std::visit([](const auto *value) {
                               return value->finalNorm;
                             }, weights_),
                             buffers.finalHidden, geometry_.hiddenSize, rows, buffers.linearScratch);
  const ops::LinearMatrix head{geometry_.vocabularySize, geometry_.hiddenSize};
  dense.addDecodeBatch(graph, buffers.finalHidden,
                     vocabularyProjection(), buffers.logits, head, lanes,
                     stats, buffers.linearScratch, true);
}

void QwenTarget::addHead(metal::CommandGraph &graph,
                         metal::MetalBuffer hidden,
                         metal::MetalBuffer finalHidden,
                         metal::MetalBuffer logits,
                         uint32_t normalizedRows, ops::LinearScratch scratch) const {
  if (!normalizedRows ||
      normalizedRows > ExecutionLimits::targetVerifyRows) {
    throw std::invalid_argument("invalid Qwen head row count");
  }
  const metal::MetalBuffer norm = std::visit(
      [](const auto *weights) { return weights->finalNorm; }, weights_);
  ops::Normalization::addRms(graph, std::move(hidden), norm, finalHidden,
                             geometry_.hiddenSize, normalizedRows);
  const DenseLinear dense(operators_, geometry_.denseQuantized8);
  const ops::LinearMatrix head{geometry_.vocabularySize, geometry_.hiddenSize};
  dense.addDecode(graph, std::move(finalHidden), vocabularyProjection(),
                  std::move(logits), head, std::move(scratch));
}

void QwenTarget::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  const ops::EmbeddingTable &embedding = std::visit(
      [](const auto *weights) -> const ops::EmbeddingTable & {
        return weights->tokenEmbedding;
      },
      weights_);
  ops::Embedding::add(graph, std::move(tokens), embedding, std::move(hidden),
                      rows);
}

void QwenTarget::addStateCommit(metal::CommandGraph &graph,
                                QwenTargetCommitBuffers buffers,
                                uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid Qwen state commit batch");
  ops::GDN::addCommit(
      graph,
      {std::move(buffers.packed), std::move(buffers.mixed),
       std::move(buffers.decay), std::move(buffers.beta), buffers.currentStates,
       buffers.nextStates, std::move(buffers.retainedCounts)},
      geometry_.gdnShape(), geometry_.stateLayout.layers, lanes,
      {geometry_.stateLayout.convolutionLayerBytes(),
       geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()});
}

} // namespace splash::model
