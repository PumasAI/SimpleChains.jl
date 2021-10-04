
isloss(::AbstractLoss) = True()
isloss(_) = False()
has_loss_typed(sc::SimpleChain) = isloss(last(sc.layers))

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

struct PullBack{PBL<:PullBackLayer, G, P, M}
  pbl::PBL
  grad::G
  params::P
  memory::M
end
function (pb::PullBack)(x)
  @unpack pbl, grad, params, memory = pb
  GC.@preserve grad  params  memory begin
    lgrad, pu4 = pullback_layer!(pbl, x)
  end
  ChainRulesCore.NoTangent(), StrideArraysCore.StrideArray(lgrad, memory), StrideArraysCore.StrideArray(grad, memory)
end


function unsafe_valgrad_pullback!(g, layers, params, memory::Vector{UInt8}, arg)
  GC.@preserve g params memory begin
    # @show pointer(g) pointer(params) pointer(memory)
    l, pbl = chain_valgrad_pullback!(pointer(g), arg, layers, pointer(params), pointer(memory))
  end
  l, PullBack(pbl, g, params, memory)
end

function chain_valgrad_pullback!(pg, arg, layers::Tuple{X1,X2,Vararg}, p::Ptr, pu::Ptr{UInt8}) where {X1,X2}
  l = getfield(layers,1)
  pg2, larg, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)

  # val, grad, pu3, pbl = chain_valgrad_pullback!(pg2, larg, Base.tail(layers), p2, pu2)
  val, pbl = chain_valgrad_pullback!(pg2, larg, Base.tail(layers), p2, pu2)
  pbl_ret = PullBackLayer(pg, l, arg, p, pu, pbl)
  return val, pbl_ret
  # lgrad, pu4 = pullback!(pg, l, grad, arg, p, pu, pu3)
  # return val, lgrad, pu4
end
function chain_valgrad_pullback!(pg, arg, layers::Tuple{X1}, p::Ptr, pu::Ptr{UInt8}) where {X1}
  l = getfield(layers,1)
  pg2, val, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)

  # val, grad, pu3, pbl = chain_valgrad!(pg2, larg, Base.tail(layers), p2, pu2)
  # pu2 gets fed into eventual `pullback!` call
  pbl_ret = PullBackLayer(pg, l, arg, p, pu, pu2)
  return val, pbl_ret
  # lgrad, pu4 = pullback!(pg, l, grad, arg, p, pu, pu3)
  # return val, lgrad, pu4
end

# No loss: chain closures.
function _rrule(sc, arg, params, ::False)
  valgrad_noloss(sc, arg, params)
end
function valgrad_noloss(sc, arg, params::AbstractVector{T}) where {T}
  c = getchain(sc)
  @unpack layers, memory = c
  off = align(resize_memory!(layers, memory, arg, length(arg)*sizeof(eltype(arg))))
  GC.@preserve memory begin
    g = PtrArray(reinterpret(Ptr{T}, pointer(memory)+off), (static_length(params),))
    l, pullback = unsafe_valgrad_pullback!(g, layers, params, memory, arg)
  end
  return StrideArraysCore.StrideArray(l, memory), pullback
end



# Loss: call `valgrad`.
function _rrule(sc, arg, params, ::True)
  l, g = valgrad(sc, arg, params)
  # assumes no grad w/ respect to arg
  pullback = let g=g
    l̄ -> begin
      if !isone(l̄)
        @turbo for i ∈ eachindex(g)
          g[i] *= l̄
        end
      end
      ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), g
    end
  end
  l, pullback
end

ChainRulesCore.rrule(sc::AbstractPenalty, arg, params) = _rrule(sc, arg, params, True())
ChainRulesCore.rrule(sc::SimpleChain, arg, params) = _rrule(sc, arg, params, has_loss_typed(sc))



