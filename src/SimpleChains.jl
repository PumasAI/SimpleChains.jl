module SimpleChains

# using ChainRules, ChainRulesCore, NNlibCPU
using UnPack, VectorizationBase, ArrayInterface, Polyester, SLEEFPirates, StrideArraysCore, Static
using ArrayInterface: size, axes, StrideIndex, contiguous_axis, stride_rank, static_length, indices
using SIMDTypes: Bit
using VectorizationBase: align, static_sizeof, relu, stridedpointer, AbstractSIMD
using LayoutPointers: bytestrideindex
using IfElse: ifelse
import ForwardDiff

using LoopVectorization
# macro turbo(ex)
#   esc(ex)
# end

export SimpleChain, TurboDense, SquaredLoss, Dropout, relu, static, StrideArray, valgrad!

include("simple_chain.jl")
include("dense.jl")
include("dropout.jl")
include("squared_loss.jl")

end
