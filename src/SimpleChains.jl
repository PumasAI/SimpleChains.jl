module SimpleChains

# using ChainRules, ChainRulesCore, NNlibCPU
using UnPack, VectorizationBase, ArrayInterface, LoopVectorization, Polyester, SLEEFPirates, StrideArraysCore, Static
using ArrayInterface: size, axes
using SIMDTypes: Bit
using VectorizationBase: align, static_sizeof

include("simple_chain.jl")
include("dense_layer.jl")

end
