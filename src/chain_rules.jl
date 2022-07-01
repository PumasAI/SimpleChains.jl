
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
struct PullBack{PBL<:PullBackLayer,G,P,M}
  pbl::PBL
  grad::G
  params::P
  memory::M
end
function (pb::PullBack{<:PullBackLayer})(x)
  @unpack pbl, grad, params, memory = pb
  GC.@preserve grad params memory begin
    lgrad, pu4 = pullback_layer!(pbl, x)
  end
  NoTangent(),
  StrideArraysCore.StrideArray(lgrad, memory),
  StrideArraysCore.StrideArray(grad, memory)
end
# function (pb::PullBack{<:PullBackParam})(x)
#   @unpack pbl, grad, params, memory = pb
#   GC.@preserve grad params memory pullback_layer!(pbl, x)
#   NoTangent(),
#   NoTangent(),
#   StrideArraysCore.StrideArray(grad, memory)
# end


# function chain_valgrad_pullback_entry!(
#   pg,
#   arg,
#   layers::Tuple{X1,X2,Vararg},
#   p::Ptr,
#   pu::Ptr{UInt8},
# ) where {X1,X2}
#   l = getfield(layers, 1)
#   pg2, larg, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)
#   val, pbl = chain_valgrad_pullback!(pg2, larg, Base.tail(layers), p2, pu2)
#   # if parameter_free(l)
#     pbl_ret = PullBackLayer(pg, l, arg, p, pu, pbl)
#     return val, pbl_ret
#   # else
#   #   pbl_ret = PullBackParam(pg, l, arg, p, pu, pbl)
#   #   return val, pbl_ret
#   # end
# end
function chain_valgrad_pullback!(
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
function chain_valgrad_pullback!(
  pg,
  arg,
  layers::Tuple{X1},
  p::Ptr,
  pu::Ptr{UInt8},
) where {X1}
  l = getfield(layers, 1)
  pg2, val, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)

  # pu2 gets fed into eventual `pullback!` call
  pbl_ret = PullBackLayer(pg, l, arg, p, pu, pu2)
  return val, pbl_ret
end

# No loss: chain closures.
function _rrule(sc, arg, params, ::False)
  valgrad_noloss(sc, arg, params)
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
    @inbounds @simd ivdep for i = eachindex(parg)
      parg2[i]=parg[i]
    end
    pm += aoff
    g = PtrArray(Ptr{T}(pm), (glen,))
    pm += goff
    # @show pointer(g) pointer(params) pointer(memory)
    l, pbl = chain_valgrad_pullback!(pointer(g), parg2, layers, pointer(params), pm)
  end
  l, PullBack(pbl, g, params, memory)
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

function ChainRulesCore.rrule(sc::AbstractPenalty, arg, params)
  _rrule(sc, arg, params, True())
end
function ChainRulesCore.rrule(sc::SimpleChain, arg, params)
  _rrule(sc, arg, params, has_loss_typed(sc))
end
