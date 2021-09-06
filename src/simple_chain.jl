
struct SimpleChain{N,L<:Tuple{Vararg{Any,N}}}
  layers::L
  memory::Vector{UInt8}
  grad::Vector{UInt8}
end
SimpleChain(l::Vararg) = SimpleChain(l, UInt8[])
SimpleChain(l::Tuple) = SimpleChain(l, UInt8[])

"""
  Base.front(c::SimpleChain)

Useful for popping off a loss layer.
"""
Base.front(c::SimpleChain) = SimpleChain(Base.front(c.layers), c.memory)
Base.vcat(c::SimpleChain, l) = SimpleChain((c.layers...,l), c.memory)


# output_size must be defined to return the total size of all outputs
output_size(::Val{T}, x::Tuple{}) where {T} = 0
output_size(::Val{T}, x::Tuple{X}) where {T,X} = output_size(Val{T}(), getfield(x,1))
function output_size(::Val{T}, x::Tuple{X1,X2,Vararg}) where {T,X1,X2}
  s = output_size(Val{T}(), getfield(x,1))
  s + output_size(Val{T}(), Base.tail(x))
end

function resize_memory!(layers, memory::Vector{UInt8}, arg::AbstractVecOrMat{T}) where {T}
  d = output_size(Val(T), layers, ArrayInterface.size(arg, StaticInt(2)))
  d > length(memory) && resize!(memory, d)
  nothing
end
function (c::SimpleChain)(arg, params)
  @unpack layers, memory = c
  resize_memory!(layers, memory, arg)
  unsafe_chain(layers, params, memory, arg)
end
function unsafe_chain(layers, params, memory::Vector{UInt8}, arg)
  GC.@preserve params memory _chain(arg, layers, pointer(params), pointer(memory))
end
_chain(arg, ::Tuple{}, p::Ptr, pu::Ptr{UInt8}) = arg
_chain(arg, l::Tuple{T}, p::Ptr, pu::Ptr{UInt8}) where {T} = getfield(getfield(l,1)(arg, p, pu), 1)
function _chain(arg, l::Tuple{T1,T2,Vararg}, p::Ptr, pu::Ptr{UInt8}) where {T1,T2}
  res, p = getfield(l,1)(arg, p)
  _chain(res, Base.tail(l), p)
end

"""
Allowed destruction:

  valgrad_layer!
Accepts return of previous layer (`B`) and returns an ouput `C`

  pullback!
Accepts adjoint of its return (`C̄`). It is allowed to destroy this.
It is also allowed to destroy the previous layer's return `B` to produce `B̄` (the `C̄` it receives).
Thus, the pullback is not allowed to depend on `C`, as it may have been destroyed in producing `C̄`.
"""
function valgrad!(g, c::SimpleChain, arg, params)
  @unpack layers, memory = c
  resize_memory!(layers, memory, arg)
  unsafe_valgrad!(g, layers, params, memory, arg)
end

function unsafe_valgrad!(g, layers, params, memory::Vector{UInt8}, arg)
  GC.@preserve g params memory begin
    chain_valgrad_entry!(pointer(g), arg, layers, pointer(params), pointer(memory))
  end
end

function chain_valgrad_entry!(pg, arg, layers::Tuple{X1,X2,Vararg}, p::Ptr, pu::Ptr{UInt8}) where {X1,X2}
  l = getfield(layers,1)
  pg2, larg, pb, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)

  val, grad, pu3 = chain_valgrad!(pg2, larg, Base.tail(layers), p2, pu2)
  pullback_param!(pg, l, grad, p, pu)
  return val
end
function chain_valgrad!(pg, arg, layers::Tuple{X1,X2,Vararg}, p::Ptr, pu::Ptr{UInt8}) where {X1,X2}
  l = getfield(layers,1)
  pg2, larg, pb, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)

  val, grad, pu3 = chain_valgrad!(pg2, larg, Base.tail(layers), p2, pu2)
  lgrad, pu4 = pullback!(pg, l, grad, arg, p, pu, pu3)
  return val, lgrad, pu4
end
function chain_valgrad!(pg, arg, layers::Tuple{X}, p::Ptr, pu::Ptr{UInt8}) where {X}
  l = getfield(layers,1)
  val, pullback, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)
  lgrad, pu3 = pullback!(pg, l, One(), arg, p, pu, pu2)
  return val, lgrad, pu3
end

