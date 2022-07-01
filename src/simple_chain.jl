
struct InputDimUnknown end
const InputDim = Union{InputDimUnknown,Tuple{Vararg{Integer}}}

"""
    SimpleChain([inputdim::Union{Integer,Tuple{Vararg{Integer}}, ] layers)

Construct a SimpleChain. Optional `inputdim` argument allows `SimpleChains` to check
the size of inputs. Making these `static` will allow `SimpleChains` to infer size
and loop bounds at compile time.
Batch size generally should not be included in the `inputdim`.
If `inputdim` is not specified, some methods, e.g. `init_params`, will require
passing the size as an additional argument, because the number of parameters may be
a function of the input size (e.g., for a `TurboDense` layer).

The `layers` argument holds various `SimpleChains` layers, e.g. `TurboDense`, `Conv`,
`Activation`, `Flatten`, `Dropout`, or `MaxPool`. It may optionally terminate in an
`AbstractLoss` layer.

These objects are callable, e.g.

```julia
c = SimpleChain(...);
p = SimpleChains.init_params(c);
c(X, p) # X are the independent variables, and `p` the parameter vector.
```
"""
struct SimpleChain{I<:InputDim,L<:Tuple}
  inputdim::I
  layers::L
end
"""
    AbstractPenalty

The `AbstractPenalty` interface requires supporting the following methods:

1. `getchain(::AbstractPenalty)::SimpleChain` returns a `SimpleChain` if it is carrying one.
2. `apply_penalty(::AbstractPenalty, params)::Number` returns the penalty
3. `apply_penalty!(grad, ::AbstractPenalty, params)::Number` returns the penalty and updates `grad` to add the gradient.
"""
abstract type AbstractPenalty{NN<:Union{SimpleChain,Nothing}} end

const Chain = Union{AbstractPenalty{<:SimpleChain},SimpleChain}

chain_input_dims(c::SimpleChain) = c.inputdim

SimpleChain(input_dim::Integer, lf, lm, lt::Vararg) =
  SimpleChain((input_dim,), (lf, lm, lt...))
SimpleChain(input_dim::InputDim, lf, lm, lt::Vararg) =
  SimpleChain(input_dim, (lf, lm, lt...))

SimpleChain(input_dim::Integer, l::Tuple) = SimpleChain((input_dim,), l)

SimpleChain(l::Vararg) = SimpleChain(InputDimUnknown(), l)
SimpleChain(l::Tuple) = SimpleChain(InputDimUnknown(), l)

function Base.similar(c::SimpleChain)
  SimpleChain(chain_input_dims(c), c.layers)
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

function outputdim(x::Tuple{X}, s1) where {X}
  last(layer_output_size(Val{Float32}(), getfield(x, 1), s1))
end
function outputdim(x::Tuple{X1,X2,Vararg}, s1::Tuple) where {X1,X2}
  # Main._a[] = (T, x, s1)
  _, s2 = layer_output_size(Val{Float32}(), getfield(x, 1), s1)
  outputdim(Base.tail(x), s2)
end
function outputdim(sc::SimpleChain, id = nothing)
  inputdim = chain_input_dims(sc, id)
  outputdim(sc.layers, inputdim)
end

"""
    Base.front(c::SimpleChain)

Useful for popping off a loss layer.
"""
Base.front(c::SimpleChain) = SimpleChain(chain_input_dims(c), Base.front(c.layers))
Base.vcat(c::SimpleChain, l) = SimpleChain(chain_input_dims(c), (c.layers..., l))

function numparam(c::SimpleChain, id = nothing)
  _id = chain_input_dims(c, id)
  _numparam(static(0), c.layers, _id)
end
_numparam(s, ::Tuple{}, _) = s
function _numparam(s, layers::Tuple{L,Vararg}, id) where {L}
  np, od = numparam(getfield(layers, 1), id)
  _numparam(s + np, Base.tail(layers), od)
end
parameter_free(x) = numparam(x) == 0


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

function (c::SimpleChain)(arg::AbstractArray{T}, params) where {T}
  verify_arg(c, arg)
  @unpack layers = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  num_bytes = required_forward_bytes(Val(T), layers, size(parg), static(0))
  if has_loss(c)
    GC.@preserve arg params with_memory(_chain, c, num_bytes, parg, pointer(params))
  else
    GC.@preserve arg params begin
      res, heap_memory = with_heap_memory(_chain, c, num_bytes, parg, pointer(params))
      StrideArray(res, heap_memory)
    end
  end
end
@inline _maybe_sarray(x) = x
@inline _maybe_sarray(x::AbstractArray) = _maybe_sarray(x, size(x))
@inline _maybe_sarray(x::AbstractArray, _) = x
@generated function _maybe_sarray(A::AbstractArray, s::Tuple{Vararg{StaticInt}})
  k = known(s)
  t = Expr(:tuple)
  ct = Expr(:curly, :Tuple)
  for x in k
    push!(ct.args, x)
  end
  for i = 1:prod(k)::Int
    push!(t.args, :(unsafe_load(p, $i)))
  end
  Expr(
    :block,
    Expr(:meta, :inline),
    :(p = pointer(A)),
    :(GC.@preserve A StaticArrays.SArray{$ct}($t)),
  )
end
function (c::SimpleChain)(arg::StaticArrays.SArray, params)
  marg = StaticArrays.MArray(arg)
  GC.@preserve marg _maybe_sarray(c(PtrArray(marg), params))
end
function (c::SimpleChain)(memory::Ptr{UInt8}, arg, params)
  verify_arg(c, arg)
  @unpack layers = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  GC.@preserve arg params _chain(c, memory, parg, pointer(params))
end

function layer_output_size(::Val{T}, l, inputdim::Tuple) where {T}
  os, od = forward_layer_output_size(Val{T}(), l, inputdim)
  os + os, od
end


@inline output_size(::Val{T}, x::Tuple{}, _) where {T} = static(0)
@inline function output_size(::Val{T}, x::Tuple{X}, s1) where {T,X}
  first(layer_output_size(Val{T}(), getfield(x, 1), s1))
end
@inline function output_size(::Val{T}, x::Tuple{X1,X2,Vararg}, s1::Tuple) where {T,X1,X2}
  b, s2 = layer_output_size(Val{T}(), getfield(x, 1), s1)
  b + output_size(Val{T}(), Base.tail(x), s2)
end
@inline forward_output_size(::Val{T}, x::Tuple{}, _) where {T} = static(0)
@inline function forward_output_size(::Val{T}, x::Tuple{X}, s1) where {T,X}
  first(forward_layer_output_size(Val{T}(), getfield(x, 1), s1))
end
@inline function forward_output_size(
  ::Val{T},
  x::Tuple{X1,X2,Vararg},
  s1::Tuple,
) where {T,X1,X2}
  b, s2 = forward_layer_output_size(Val{T}(), getfield(x, 1), s1)
  b + forward_output_size(Val{T}(), Base.tail(x), s2)
end
__chain(::Tuple{}, arg, p::Ptr, pu::Ptr{UInt8}) = arg
__chain(l::Tuple{T}, arg, p::Ptr, pu::Ptr{UInt8}) where {T} =
  getfield(getfield(l, 1)(arg, p, pu), 1)
function __chain(l::Tuple{T1,T2,Vararg}, arg, p::Ptr, pu::Ptr{UInt8}) where {T1,T2}
  res, p, pu = getfield(l, 1)(arg, p, pu)
  __chain(Base.tail(l), res, p, pu)
end
function _chain(c::Chain, pu::Ptr{UInt8}, arg, p)
  @unpack layers = c
  __chain(layers, arg, p, pu)
end
@inline function _try_static(i::Base.Integer, j::Base.Integer)
  @assert i == j
  return i
end
@inline function _try_static(::StaticInt{I}, j::Base.Integer) where {I}
  @assert I == j
  return StaticInt{I}()
end
@inline function _try_static(j::Base.Integer, ::StaticInt{I}) where {I}
  @assert I == j
  return StaticInt{I}()
end
@inline function _try_static(::StaticInt{I}, ::StaticInt{J}) where {I,J}
  throw(ArgumentError("StaticInt{$I}() != StaticInt{$J}()"))
end
@inline _try_static(::StaticInt{I}, ::StaticInt{I}) where {I} = StaticInt{I}()

@inline _try_static(::Tuple{}, x::Tuple{Vararg}) = x
@inline function _try_static(x::Tuple{X,Vararg}, y::Tuple{Y,Vararg}) where {X,Y}
  (_try_static(first(x), first(y)), _try_static(Base.tail(x), Base.tail(y))...)
end
@inline function _try_static(j::Integer, i::Tuple{I,Vararg}) where {I}
  (_try_static(j, first(i)), Base.tail(i)...)
end
chain_input_dims(::SimpleChain{InputDimUnknown}, inputdim::Tuple{Vararg{Integer}}) =
  inputdim
function chain_input_dims(::SimpleChain{InputDimUnknown}, ::Nothing)
  throw(
    ArgumentError(
      "SimpleChains without an explicitly provided InputDim require manually providing it when calling `init_params`",
    ),
  )
end
chain_input_dims(chn::SimpleChain, ::Nothing) = chain_input_dims(chn)
function chain_input_dims(chn::SimpleChain, inputdim::Tuple{Vararg{Integer}})
  _try_static(chain_input_dims(chn), inputdim)
end


"""
    SimpleChains.init_params!(chn, p, id = nothing)

Randomly initializes parameter vector `p` with input dim `id`. Input dim does not need to be specified if these were provided to the chain object itself.
See the documentation of the individual layers to see how they are initialized, but it is generally via (Xavier) Glorot uniform or normal distributions.
"""
function init_params!(chn::SimpleChain, x::AbstractVector, id = nothing)
  GC.@preserve x init_params!(chn.layers, pointer(x), chain_input_dims(chn, id))
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
  ::Type{T} = Float32,
) where {T}
  _id = chain_input_dims(Λ, id)
  init_params!(Λ, StrideArray{T}(undef, numparam(Λ, id)), chain_input_dims(Λ, _id))
end
"""
    SimpleChains.init_params(chn[, id = nothing][, ::Type{T} = Float32])

Creates a parameter vector of element type `T` with size matching that by `id` (argument not required if provided to the `chain` object itself).
See the documentation of the individual layers to see how they are initialized, but it is generally via (Xavier) Glorot uniform or normal distributions.
"""
function init_params(Λ::SimpleChain, ::Type{T}) where {T}
  init_params(Λ, nothing, T)
end

@inline maybe_static_size_arg(_, arg) = arg
@inline function maybe_static_size_arg(s::Tuple, arg::Array)
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
function valgrad!(memory::Ptr{UInt8}, g, c::SimpleChain, arg, params)
  verify_arg(c, arg)
  @unpack layers = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  GC.@preserve arg unsafe_valgrad!(c, memory, g, params, parg)
end
function valgrad!(g, c::SimpleChain, arg::AbstractArray{T}, params) where {T}
  verify_arg(c, arg)
  @assert has_loss(c)
  @unpack layers = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  num_bytes = required_bytes(Val{T}(), layers, size(parg), static(0))
  GC.@preserve arg with_memory(unsafe_valgrad!, c, num_bytes, g, params, parg)
end

function unsafe_valgrad!(c::Chain, pu::Ptr{UInt8}, g, params, arg)
  @unpack layers = c
  GC.@preserve g params begin
    # @show pointer(g) pointer(params) pointer(memory)
    chain_valgrad_entry!(pointer(g), arg, layers, pointer(params), pu)
  end
end
# fallback valgrad_layer for functions not implementing fusion w/ indexing

function subset_batch(Xp::AbstractArray{T,N}, perm, pu) where {T,N}
  Xsz = Base.front(size(Xp))
  lastdim = length(perm)
  Xtsz = (Xsz..., lastdim)
  Xtmp = PtrArray(Ptr{T}(pu), Xtsz)
  Xlen = tsprod(Xsz)
  pXtmp = pointer(Xtmp)
  szeltx = sizeof(T)
  pu += align(szeltx * Xlen * lastdim)
  pX = pointer(Xp)
  Xpb = preserve_buffer(Xp)
  GC.@preserve Xpb begin
    for i in CloseOpen(lastdim)
      @inbounds j = perm[i] # `perm` and `j` are zero-based
      Base.unsafe_copyto!(pXtmp, pX + Xlen * szeltx * j, Int(Xlen))
      pXtmp += Int(Xlen) * szeltx
    end
  end
  Xtmp, pu
end

function chain_valgrad_entry!(
  pg,
  arg,
  layers::Tuple{X1,X2,Vararg},
  p::Ptr,
  pu::Ptr{UInt8},
) where {X1,X2}
  l = getfield(layers, 1)
  pg2, larg, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)
  if parameter_free(l)
    val = chain_valgrad_entry!(pg2, larg, Base.tail(layers), p2, pu2)
  else
    val, grad, _ = chain_valgrad!(pg2, larg, Base.tail(layers), p2, pu2)
    pullback_param!(pg, l, grad, arg, p, pu)
  end
  return val
end
function chain_valgrad_entry!(
  pg,
  arg,
  layers::Tuple{X1,X2,Vararg},
  inds,
  p::Ptr,
  pu::Ptr{UInt8},
) where {X1,X2}
  arg_subset, pu = subset_batch(arg, inds, pu)
  chain_valgrad_entry!(pg, arg_subset, layers, p, pu)
end

function chain_valgrad!(
  pg,
  arg,
  layers::Tuple{X1,X2,Vararg},
  p::Ptr,
  pu::Ptr{UInt8},
) where {X1,X2}
  l = getfield(layers, 1)
  pg2, larg, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)
  val, grad, pu3 = chain_valgrad!(pg2, larg, Base.tail(layers), p2, pu2)
  lgrad, pu4 = pullback!(pg, l, grad, arg, p, pu, pu3)
  return val, lgrad, pu4
end
function chain_valgrad!(pg, arg, layers::Tuple{X}, p::Ptr, pu::Ptr{UInt8}) where {X}
  l = getfield(layers, 1)
  __, val, _, pu2 = valgrad_layer!(pg, l, arg, p, pu)
  # val, pullback, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)
  lgrad, pu3 = pullback!(pg, l, One(), arg, p, pu, pu2)
  return val, lgrad, pu3
end
@inline getchain(sc::SimpleChain) = sc
function valgrad_core(
  c::Chain,
  pu::Ptr{UInt8},
  arg,
  params::AbstractVector{T},
  glen,
) where {T}
  @unpack layers = c
  g = PtrArray(Ptr{T}(pu), (glen,))
Base.FastMath.add_fast(
    unsafe_valgrad!(c, pu + align(glen * static_sizeof(T)), g, params, arg),
    apply_penalty!(g, getpenalty(c), params, size(arg)),
  )
end
function valgrad_core_sarray(
  c::Chain,
  pu::Ptr{UInt8},
  arg,
  params::AbstractVector{T},
  ::StaticInt{L},
) where {T,L}
  @unpack layers = c
  g = PtrArray(Ptr{T}(pu), (static(L),))
  l = Base.FastMath.add_fast(
    unsafe_valgrad!(c, pu + align(static(L) * static_sizeof(T)), g, params, arg),
    apply_penalty!(g, getpenalty(c), params, size(arg)),
  )
  return l, _maybe_sarray(g, (static(L),))
end
function valgrad(sc::Chain, arg, params::AbstractVector{T}) where {T}
  c = getchain(sc)
  @unpack layers = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  glen = _try_static(numparam(sc), static_length(params))
  num_bytes = required_bytes(Val{T}(), layers, size(parg), glen * static_sizeof(T))
  l, heap_memory = with_heap_memory(valgrad_core, sc, num_bytes, parg, params, glen)
  gv = StrideArraysCore.StrideArray(
    PtrArray(align(Ptr{T}(pointer(heap_memory))), (glen,)),
    heap_memory,
  )
  return l, gv
end
function valgrad(sc::Chain, arg::StaticArrays.SArray, params::AbstractVector{T}) where {T}
  c = getchain(sc)
  @unpack layers = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  glen = _try_static(numparam(sc), static_length(params))
  num_bytes = required_bytes(Val{T}(), layers, size(parg), glen * static_sizeof(T))
  if glen isa StaticInt
    return with_memory(valgrad_core_sarray, sc, num_bytes, parg, params, glen)
  else
    l, heap_memory = with_heap_memory(valgrad_core, sc, num_bytes, parg, params, glen)
    gv = StrideArraysCore.StrideArray(
      PtrArray(Ptr{T}(pointer(heap_memory)), (glen,)),
      heap_memory,
    )
    return l, gv
  end
end

isstochastic(_) = false
