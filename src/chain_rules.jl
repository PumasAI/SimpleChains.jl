
if isdefined(ChainRulesCore, :NoTangent)
  const NoTangent = ChainRulesCore.NoTangent
else
  const NoTangent = ChainRulesCore.DoesNotExist
end

isloss(::AbstractLoss) = True()
isloss(_) = False()
has_loss_typed(sc::SimpleChain) = isloss(last(sc.layers))
has_loss_typed(sc::AbstractPenalty) = has_loss_typed(getchain(sc))

struct PullBackLayer{T,L,A,PBL}
  pg::Ptr{T}
  l::L
  arg::A
  p::Ptr{T}
  pu::Ptr{UInt8}
  pbl::PBL # either another `PullBackLayer`, or the last memory pointer from the forward pass (to start the reverse)
end
function pullback_layer!(pbl::PullBackLayer, lgrad)
  grad, pu3 = pullback_layer!(pbl.pbl, lgrad)
  pullback!(pbl.pg, pbl.l, grad, pbl.arg, pbl.p, pbl.pu, pu3)
end
pullback_layer!(pbl::Ptr{UInt8}, grad) = grad, pbl


#TODO: add support for not getting gradient with respect to input `x`
# struct PullBackParam{T,L,A,PBL}
#   pg::Ptr{T}
#   l::L
#   arg::A
#   p::Ptr{T}
#   pu::Ptr{UInt8}
#   pbl::PBL # either another `PullBackLayer`, or the last memory pointer from the forward pass (to start the reverse)
# end
# function pullback_layer!(pbl::PullBackParam, lgrad)
#   grad, _ = pullback_layer!(pbl.pbl, lgrad)
#   pullback_param!(pbl.pg, pbl.l, grad, pbl.arg, pbl.p, pbl.pu)
# end

# struct PullBack{PBL<:Union{PullBackLayer,PullBackParam},G,P,M}
struct PullBack{SA,PBL<:PullBackLayer,G,P,M}
  pbl::PBL
  grad::G
  params::P
  memory::M
  function PullBack{SA}(pbl::PBL, grad::G, params::P, memory::M) where {SA,PBL,G,P,M}
    new{SA,PBL,G,P,M}(pbl, grad, params, memory)
  end
end
@inline function (pb::PullBack{SA})(x) where {SA}
  @unpack pbl, grad, params, memory = pb
  GC.@preserve grad params memory begin
    lgrad, _ = pullback_layer!(pbl, x)
  end
  if SA
    NoTangent(),
    _maybe_sarray(StrideArraysCore.StrideArray(lgrad, memory)),
    _maybe_sarray(StrideArraysCore.StrideArray(grad, memory))
  else
    NoTangent(),
    StrideArraysCore.StrideArray(lgrad, memory),
    StrideArraysCore.StrideArray(grad, memory)
  end
end
@inline function (pb::PullBack)(x::StaticArrays.SArray)
  @unpack pbl, grad, params, memory = pb
  mx = StaticArrays.MArray(x);
  GC.@preserve mx grad params memory begin
    lgrad, _ = pullback_layer!(pbl, PtrArray(mx))
  end
  NoTangent(),
  _maybe_sarray(StrideArraysCore.StrideArray(lgrad, memory)),
  _maybe_sarray(StrideArraysCore.StrideArray(grad, memory))
end


@inline function chain_valgrad_pullback!(
  pg,
  arg,
  layers::Tuple{X1,X2,Vararg},
  p::Ptr,
  pu::Ptr{UInt8},
) where {X1,X2}
  l = getfield(layers, 1)
  pg2, larg, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)

  val, pbl = chain_valgrad_pullback!(pg2, larg, Base.tail(layers), p2, pu2)
  pbl_ret = PullBackLayer(pg, l, arg, p, pu, pbl)
  return val, pbl_ret
end
@inline function chain_valgrad_pullback!(
  pg,
  arg,
  layers::Tuple{X1},
  p::Ptr,
  pu::Ptr{UInt8},
) where {X1}
  l = getfield(layers, 1)
  _, val, __, pu2 = valgrad_layer!(pg, l, arg, p, pu)

  # pu2 gets fed into eventual `pullback!` call
  pbl_ret = PullBackLayer(pg, l, arg, p, pu, pu2)
  return val, pbl_ret
end

# No loss: chain closures.
function _rrule(sc, arg, params, ::False)
  valgrad_noloss(sc, arg, params)
end
function valgrad_noloss(sc, arg::AbstractArray{S}, params::StaticArrays.SVector{T}) where {T,S}
  mp = StaticArrays.MVector(params);
  @gc_preserve valgrad_noloss(sc, arg, mp)
end
function valgrad_noloss(sc, arg::AbstractArray{S}, params::AbstractVector{T}) where {T,S}
  c = getchain(sc)
  @unpack layers = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  arglen = length(parg)
  barg = preserve_buffer(arg)
  glen = _try_static(numparam(sc), static_length(params))
  goff = align(glen * static_sizeof(T))
  aoff = align(arglen * static_sizeof(S))

  num_bytes = required_bytes(Val{T}(), layers, size(parg), aoff + goff)
  memory = get_heap_memory(sc, num_bytes)

  GC.@preserve barg params memory begin
    pm = align(pointer(memory))
    parg2 = PtrArray(Ptr{S}(pm), _try_static(c.inputdim, size(parg)))
    @inbounds @simd ivdep for i in eachindex(parg)
      parg2[i] = parg[i]
    end
    pm += aoff
    g = PtrArray(Ptr{T}(pm), (glen,))
    pm += goff
    # @show pointer(g) pointer(params) pointer(memory)
    l, pbl = chain_valgrad_pullback!(pointer(g), parg2, layers, pointer(params), pm)
  end
  if arg isa StaticArrays.SArray
    _maybe_sarray(l), PullBack{true}(pbl, g, params, memory)
  else
    l, PullBack{true}(pbl, g, params, memory)
  end
end

struct ElementwisePullback{G}
  g::G
end
#TODO: add support for getting gradient with respect to `arg`
function (ep::ElementwisePullback)(l̄)
  g = ep.g
  if !isone(l̄)
    @turbo for i ∈ eachindex(g)
      g[i] *= l̄
    end
  end
  # assumes no grad w/ respect to arg
  NoTangent(), NoTangent(), g
end
# Loss: call `valgrad`.
function _rrule(sc, arg, params, ::True)
  l, g = valgrad(sc, arg, params)
  l, ElementwisePullback(g)
end
# TODO: support penalties without returning scalars
_returns_scalar(::AbstractPenalty) = True()
_returns_scalar(sc::SimpleChain) = has_loss_typed(sc)

function ChainRulesCore.rrule(sc::Chain, arg, params)
  _rrule(sc, arg, params, _returns_scalar(sc))
end
