module SimpleChains

# using ChainRules, ChainRulesCore, NNlibCPU
using UnPack,
  VectorizationBase, ArrayInterface, Polyester, SLEEFPirates, StrideArraysCore, Static
using ArrayInterface:
  size,
  axes,
  StrideIndex,
  contiguous_axis,
  stride_rank,
  static_length,
  static_first,
  static_last,
  static_step,
  indices,
  offsets
using SIMDTypes: Bit
using VectorizationBase: align, relu, stridedpointer, AbstractSIMD
using HostCPUFeatures: static_sizeof, register_size, register_count, static_sizeof
using CPUSummary: cache_linesize
using LayoutPointers: bytestrideindex, stridedpointer, zero_offsets
import ManualMemory: preserve_buffer
using IfElse: ifelse
import Random
import ChainRulesCore
import ForwardDiff

using LoopVectorization: matmul_params, CloseOpen, @turbo
# macro turbo(ex); esc(ex); end

export SimpleChain,
  TurboDense,
  Dropout,
  Activation,
  Conv,
  MaxPool,
  Flatten,
  AbsoluteLoss,
  SquaredLoss,
  LogitCrossEntropyLoss,
  relu,
  static,
  StrideArray,
  valgrad!,
  valgrad,
  NoPenalty,
  L1Penalty,
  L2Penalty,
  FrontLastPenalty

include("utils.jl")
include("simple_chain.jl")
include("activation.jl")
include("dense.jl")
include("dropout.jl")
include("conv.jl")
include("loss.jl")
include("maxpool.jl")
include("flatten.jl")
include("penalty.jl")
include("chain_rules.jl")
include("optimize.jl")

if VERSION >= v"1.7.0"
  if hasfield(Method, :recursion_relation)
    dont_limit = Returns(true)
    for m in methods(chain_valgrad!)
      m.recursion_relation = dont_limit
    end
    for m in methods(_chain)
      m.recursion_relation = dont_limit
    end
    for m in methods(output_size)
      m.recursion_relation = dont_limit
    end
  end
end

end
