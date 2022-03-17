
struct InputDimUnknown end
const InputDim=Union{InputDimUnknown,Integer,Tuple{Vararg{Integer}}}

struct SimpleChain{N,I<:InputDim,L<:Tuple{Vararg{Any,N}}}
  inputdim::I
  layers::L
  memory::Vector{UInt8}
end

#=
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


_verify_chain(::Tuple{}, _) = nothing
function _verify_chain(layers::Tuple{L,Vararg}, inputdim = _input_dims(layers)) where {L}
  l = first(layers)
  _check_input_dims(l, inputdim)
  d = output_size(Val(Float32), l, (inputdim,))[2][1]
  _verify_chain(Base.tail(layers), d)
end
=#
chain_input_dims(c::SimpleChain) = c.inputdim

function SimpleChain(input_dim::InputDim, l::Vararg)
  SimpleChain(input_dim, l, UInt8[])
end
function SimpleChain(input_dim::InputDim, l::Tuple)
  SimpleChain(input_dim, l, UInt8[])
end
SimpleChain(l::Vararg) = SimpleChain(InputDimUnknown(), l, UInt8[])
SimpleChain(l::Tuple) = SimpleChain(InputDimUnknown(), l, UInt8[])

function Base.similar(c::SimpleChain)
  SimpleChain(chain_input_dims(c), c.layers, similar(c.memory))
end

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
Base.front(c::SimpleChain) = SimpleChain(chain_input_dims(c), Base.front(c.layers), c.memory)
Base.vcat(c::SimpleChain, l) = SimpleChain(chain_input_dims(c), (c.layers...,l), c.memory)


# output_size must be defined to return the total size of all outputs
output_size(::Val{T}, x::Tuple{}, _) where {T} = 0
function output_size(::Val{T}, x::Tuple{X}, s1) where {T,X}
  b, _ = output_size(Val{T}(), getfield(x,1), s1)
  b
end
function output_size(::Val{T}, x::Tuple{X1,X2,Vararg}, s1) where {T,X1,X2}
  # Main._a[] = (T, x, s1)
  b, s2 = output_size(Val{T}(), getfield(x,1), s1)
  b + output_size(Val{T}(), Base.tail(x), s2)
end
function numparam(c::SimpleChain, id = nothing)
  _id = chain_input_dims(c, id)
  _numparam(0, c.layers, _id)
end
_numparam(s, ::Tuple{}, _) = s
function _numparam(s, layers::Tuple{L,Vararg}, id) where {L}
  np, od = numparam(getfield(layers, 1), id)
  _numparam(s + np, Base.tail(layers), od)
end
parameter_free(x) = numparam(x) == 0

@inline function resize_memory!(layers, memory::Vector{UInt8}, arg::AbstractVecOrMat{T}, additional = static(0)) where {T}
  d = output_size(Val(T), layers, ArrayInterface.size(arg)) + additional
  d2 = 2d
  d2 > length(memory) && resize!(memory, d2)
  d
end

matches(::InputDimUnknown, _) = true
matches(x::Integer, y::Integer) = x == y
matches(x::Tuple{Integer,Vararg}, y::Integer) = first(x) == y
matches(x::Integer, y::Tuple{Integer,Vararg}) = x == first(y)
matches(::Tuple{}, ::Tuple) = true
function matches(x::Tuple{X,Vararg}, y::Tuple{Y,Vararg}) where {X,Y}
  matches(first(x), first(y)) && matches(Base.tail(x), Base.tail(y))
end
function verify_arg(c, arg)
  if !matches(chain_input_dims(c), size(arg))
    throw(ArgumentError("Input argument: !matches(chain_input_dims(c), size(arg))"))
  end
end
function (c::SimpleChain)(arg, params)
  verify_arg(c, arg)
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

@inline function _try_static(i::Integer, j::Integer)
  @assert i == j
  return i
end
@inline function _try_static(::StaticInt{I}, j::Integer) where {I}
  @assert I == j
  return StaticInt{I}()
end
@inline function _try_static(j::Integer, ::StaticInt{I}) where {I}
  @assert I == j
  return StaticInt{I}()
end
@inline function _try_static(::StaticInt{I}, ::StaticInt{J}) where {I,J}
  throw(ArgumentError("StaticInt{$I}() != StaticInt{$J}()"))
end
@inline _try_static(::StaticInt{I}, ::StaticInt{I}) where {I} = StaticInt{I}()

@inline _try_static(::Tuple{}, ::Tuple{Vararg}) = ()
@inline function _try_static(x::Tuple{X,Vararg}, y::Tuple{Y,Vararg}) where {X,Y}
  (
    _try_static(first(x), first(y)),
    _try_static(Base.tail(x), Base.tail(y))...
   )
end
@inline function _try_static(j::Integer, i::Tuple{I,Vararg}) where {I}
    (_try_static(j, first(i)), Base.tail(i)...)
end
chain_input_dims(::SimpleChain{<:Any,InputDimUnknown}, id) = id
function chain_input_dims(::SimpleChain{<:Any,InputDimUnknown}, ::Nothing)
  throw(ArgumentError("SimpleChains without an explicitly provided InputDim require manually providing it when calling `init_params`"))
end
chain_input_dims(chn::SimpleChain, ::Nothing) = chain_input_dims(chn)
function chain_input_dims(chn::SimpleChain, id::InputDim)
  _try_static(chain_input_dims(chn), id)
end

function init_params!(chn::SimpleChain, x::AbstractVector, id = nothing)
  GC.@preserve x begin
    init_params!(chn.layers, pointer(x), chain_input_dims(chn, id))
  end
  return x
end
function init_params!(layers::Tuple, p::Ptr, id)
  p, od = init_params!(first(layers), p, id)
  init_params!(Base.tail(layers), p, od)
end
init_params!(::Tuple{}, p::Ptr, _) = nothing
function init_params(
  Λ::SimpleChain,
  id::Union{Nothing,InputDim} = nothing,
  ::Type{T} = Float32
) where {T}
  _id = chain_input_dims(Λ, id)
  init_params!(Λ, Vector{T}(undef, numparam(Λ, id)), chain_input_dims(Λ, _id))
end
function init_params(Λ::SimpleChain, ::Type{T}) where {T}
  init_params(Λ, nothing, T)
end

maybe_static_size_arg(_, arg) = arg
function maybe_static_size_arg(s::Tuple, arg::Array)
  PtrArray(pointer(arg), _try_static(s, size(arg)))
end

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
  verify_arg(c, arg)
  @unpack layers, memory = c
  resize_memory!(layers, memory, arg)
  GC.@preserve arg begin
    unsafe_valgrad!(g, layers, params, memory, maybe_static_size_arg(c.inputdim, arg))
  end
end

function unsafe_valgrad!(g, layers, params, memory::Vector{UInt8}, arg)
  GC.@preserve g params memory arg begin
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
    val, grad, _ = chain_valgrad!(pg2, larg, Base.tail(layers), p2, pu2)
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
  __, val, _, pu2 = valgrad_layer!(pg, l, arg, p, pu)
  # val, pullback, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)
  lgrad, pu3 = pullback!(pg, l, One(), arg, p, pu, pu2)
  return val, lgrad, pu3
end
@inline getchain(sc::SimpleChain) = sc
function valgrad(sc, arg, params::AbstractVector{T}) where {T}
  c = getchain(sc)
  @unpack layers, memory = c
  off = align(resize_memory!(layers, memory, arg))
  parg = maybe_static_size_arg(c.inputdim, arg)
  GC.@preserve memory arg begin
    g = PtrArray(reinterpret(Ptr{T}, pointer(memory)+off), (static_length(params),))
    l = Base.FastMath.add_fast(
      unsafe_valgrad!(g, layers, params, memory, parg),
      apply_penalty!(g, getpenalty(sc), params, size(parg))
    )
  end
  return l, StrideArraysCore.StrideArray(g, memory)
end

isstochastic(_) = false

