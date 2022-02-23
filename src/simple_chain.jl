
struct SimpleChain{N,L<:Tuple{Vararg{Any,N}}}
  layers::L
  memory::Vector{UInt8}
end

input_dims(_) = nothing
function _check_input_dims(x, i)
  d = input_dims(x)
  d === nothing || d == i || throw(ArgumentError("Input size of one layer did not match the next."))
end
function _input_dims(t::Tuple{L,Vararg}) where {L}
  l = first(t)
  d = input_dims(l)
  d === nothing ? _input_dims(Base.tail(t)) : d
end 
chain_input_dims(c::SimpleChain) = _input_dims(c.layers)

_verify_chain(::Tuple{}, _) = nothing
function _verify_chain(layers::Tuple{L,Vararg}, inputdim = _input_dims(layers)) where {L}
  l = first(layers)
  _check_input_dims(l, inputdim)
  d = output_size(Val(Float32), l, (inputdim,))[2][1]
  _verify_chain(Base.tail(layers), d)
end

SimpleChain(l::Vararg) = (_verify_chain(l); SimpleChain(l, UInt8[]))
SimpleChain(l::Tuple) = (_verify_chain(l); SimpleChain(l, UInt8[]))
Base.similar(c::SimpleChain) = SimpleChain(c.layers, similar(c.memory))

_show(::IO, ::Tuple{}) = nothing
function _show(io::IO, t::Tuple{T,Vararg}) where {T}
  println(io)
  show(io, first(t))
  _show(io, Base.tail(t))
end
function Base.show(io::IO, sc::SimpleChain)
  print(io, "SimpleChain with the following layers:")
  _show(io, sc.layers)
end


"""
  Base.front(c::SimpleChain)

Useful for popping off a loss layer.
"""
Base.front(c::SimpleChain) = SimpleChain(Base.front(c.layers), c.memory)
Base.vcat(c::SimpleChain, l) = SimpleChain((c.layers...,l), c.memory)


# output_size must be defined to return the total size of all outputs
output_size(::Val{T}, x::Tuple{}, _) where {T} = 0
function output_size(::Val{T}, x::Tuple{X}, s1) where {T,X}
  b, _ = output_size(Val{T}(), getfield(x,1), s1)
  b
end
function output_size(::Val{T}, x::Tuple{X1,X2,Vararg}, s1) where {T,X1,X2}
  b, s2 = output_size(Val{T}(), getfield(x,1), s1)
  b + output_size(Val{T}(), Base.tail(x), s2)
end
numparam(c::SimpleChain) = _numparam(0, c.layers)
_numparam(s, ::Tuple{}) = s
_numparam(s, layers::Tuple{L,Vararg}) where {L} = _numparam(s + numparam(getfield(layers, 1)), Base.tail(layers))
parameter_free(x) = numparam(x) == 0

@inline function resize_memory!(layers, memory::Vector{UInt8}, arg::AbstractVecOrMat{T}, additional = static(0)) where {T}
  d = output_size(Val(T), layers, ArrayInterface.size(arg)) + additional
  d2 = 2d
  d2 > length(memory) && resize!(memory, d2)
  d
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
  res, p, pu = getfield(l,1)(arg, p, pu)
  _chain(res, Base.tail(l), p, pu)
end

function init_params!(chn::SimpleChain, x::AbstractVector)
  GC.@preserve x init_params!(chn.layers, pointer(x))
  return x
end
function init_params!(layers::Tuple, p::Ptr)
  p = init_params!(first(layers), p)
  init_params!(Base.tail(layers), p)
end
init_params!(::Tuple{}, p::Ptr) = nothing
init_params(Λ::SimpleChain, ::Type{T} = Float32) where {T} = init_params!(Λ, Vector{T}(undef, numparam(Λ)))

"""
Allowed destruction:

  valgrad_layer!
Accepts return of previous layer (`B`) and returns an ouput `C`.
If an internal layer, allowed to destroy `B` (e.g. dropout layer).

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
    # @show pointer(g) pointer(params) pointer(memory)
    chain_valgrad_entry!(pointer(g), arg, layers, pointer(params), pointer(memory))
  end
end

function chain_valgrad_entry!(pg, arg, layers::Tuple{X1,X2,Vararg}, p::Ptr, pu::Ptr{UInt8}) where {X1,X2}
  l = getfield(layers,1)
  pg2, larg, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)
  if parameter_free(l)
    val = chain_valgrad_entry!(pg2, larg, Base.tail(layers), p2, pu2)
  else
    val, grad, pu3 = chain_valgrad!(pg2, larg, Base.tail(layers), p2, pu2)
    pullback_param!(pg, l, grad, arg, p, pu)
  end
  return val
end
function chain_valgrad!(pg, arg, layers::Tuple{X1,X2,Vararg}, p::Ptr, pu::Ptr{UInt8}) where {X1,X2}
  l = getfield(layers,1)
  pg2, larg, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)
  val, grad, pu3 = chain_valgrad!(pg2, larg, Base.tail(layers), p2, pu2)
  lgrad, pu4 = pullback!(pg, l, grad, arg, p, pu, pu3)
  return val, lgrad, pu4
end
function chain_valgrad!(pg, arg, layers::Tuple{X}, p::Ptr, pu::Ptr{UInt8}) where {X}
  l = getfield(layers,1)
  pg2, val, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)
  # val, pullback, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)
  lgrad, pu3 = pullback!(pg, l, One(), arg, p, pu, pu2)
  return val, lgrad, pu3
end
@inline getchain(sc::SimpleChain) = sc
function valgrad(sc, arg, params::AbstractVector{T}) where {T}
  c = getchain(sc)
  @unpack layers, memory = c
  off = align(resize_memory!(layers, memory, arg))
  GC.@preserve memory begin
    g = PtrArray(reinterpret(Ptr{T}, pointer(memory)+off), (static_length(params),))
    l = unsafe_valgrad!(g, layers, params, memory, arg)
    l = Base.FastMath.add_fast(l, apply_penalty!(g, getpenalty(sc), params))
  end
  return l, StrideArraysCore.StrideArray(g, memory)
end

isstochastic(_) = false

