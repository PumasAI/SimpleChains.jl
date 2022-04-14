
struct InputDimUnknown end
const InputDim = Union{InputDimUnknown,Tuple{Vararg{Integer}}}

"""
    SimpleChain([inputdim::Union{Integer,Tuple{Vararg{Integer}}, ] layers)

Construct a SimpleChain. Optional `input dims` argument allows `SimpleChains` to check
the size of inputs. Making these `static` will allow `SimpleChains` to infer size
and loop bounds at compile time.
Batch size generally should not be included in the `input dim`.
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

SimpleChain(input_dim::Integer, l::Vararg) = SimpleChain((input_dim,), l, UInt8[])
SimpleChain(input_dim::Integer, l::Tuple) = SimpleChain((input_dim,), l, UInt8[])
SimpleChain(input_dim::InputDim, l::Vararg) = SimpleChain(input_dim, l, UInt8[])
SimpleChain(input_dim::InputDim, l::Tuple) = SimpleChain(input_dim, l, UInt8[])

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
Base.front(c::SimpleChain) =
  SimpleChain(chain_input_dims(c), Base.front(c.layers), c.memory)
Base.vcat(c::SimpleChain, l) = SimpleChain(chain_input_dims(c), (c.layers..., l), c.memory)

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

@inline function resize_memory!(
  layers,
  memory::Vector{UInt8},
  ::Val{T},
  sx,
  additional = static(0),
) where {T}
  d = output_size(Val(T), layers, sx) + additional
  d2 = (2d)
  if d2 > length(memory)
    empty!(memory)
    resize!(memory, d2)
  end
  d
end
@inline function resize_memory!(
  layers,
  memory::Vector{UInt8},
  ::Val{T},
  sx,
  additional,
  additional_per_thread,
  nthread,
) where {T}
  base_mem_per_thread = 2output_size(Val(T), layers, sx) + additional_per_thread
  mem_total = additional + base_mem_per_thread * nthread
  if mem_total > length(memory)
    empty!(memory)
    resize!(memory, mem_total)
  end
  base_mem_per_thread
end
@inline function resize_memory!(
  layers,
  memory::Vector{UInt8},
  arg::AbstractArray{T},
  additional = static(0),
) where {T}
  resize_memory!(layers, memory, Val(T), size(arg), additional)
end
@inline function resize_memory!(
  layers,
  memory::Vector{UInt8},
  arg::AbstractArray{T},
  additional,
  additional_per_thread,
  nthread,
) where {T}
  resize_memory!(
    layers,
    memory,
    Val(T),
    size(arg),
    additional,
    additional_per_thread,
    nthread,
  )
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
  parg = maybe_static_size_arg(c.inputdim, arg)
  resize_memory!(layers, memory, parg)
  GC.@preserve arg unsafe_chain(layers, params, memory, parg)
end
@inline function unsafe_chain(layers, params, memory::Vector{UInt8}, arg)
  GC.@preserve params memory _chain(arg, layers, pointer(params), pointer(memory))
end

@inline output_size(::Val{T}, x::Tuple{}, _) where {T} = 0
@inline function (output_size(::Val{T}, x::Tuple{X}, s1)::Int) where {T,X}
  first(layer_output_size(Val{T}(), getfield(x, 1), s1))
end
@inline function (output_size(::Val{T}, x::Tuple{X1,X2,Vararg}, s1::Tuple)::Int) where {T,X1,X2}
  b, s2 = layer_output_size(Val{T}(), getfield(x, 1), s1)
  b + output_size(Val{T}(), Base.tail(x), s2)
end
_chain(arg, ::Tuple{}, p::Ptr, pu::Ptr{UInt8}) = arg
_chain(arg, l::Tuple{T}, p::Ptr, pu::Ptr{UInt8}) where {T} =
  getfield(getfield(l, 1)(arg, p, pu), 1)
function _chain(arg, l::Tuple{T1,T2,Vararg}, p::Ptr, pu::Ptr{UInt8}) where {T1,T2}
  res, p, pu = getfield(l, 1)(arg, p, pu)
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

@inline _try_static(::Tuple{}, x::Tuple{Vararg}) = x
@inline function _try_static(x::Tuple{X,Vararg}, y::Tuple{Y,Vararg}) where {X,Y}
  (_try_static(first(x), first(y)), _try_static(Base.tail(x), Base.tail(y))...)
end
@inline function _try_static(j::Integer, i::Tuple{I,Vararg}) where {I}
  (_try_static(j, first(i)), Base.tail(i)...)
end
chain_input_dims(::SimpleChain{<:Any,InputDimUnknown}, inputdim::Tuple{Vararg{Integer}}) =
  inputdim
function chain_input_dims(::SimpleChain{<:Any,InputDimUnknown}, ::Nothing)
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
  init_params!(Λ, Vector{T}(undef, numparam(Λ, id)), chain_input_dims(Λ, _id))
end
"""
    SimpleChains.init_params(chn[, id = nothing][, ::Type{T} = Float32])

Creates a parameter vector of element type `T` with size matching that by `id` (argument not reguired if provided to the `chain` object itself.
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
function valgrad!(g, c::SimpleChain, arg, params)
  verify_arg(c, arg)
  @unpack layers, memory = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  resize_memory!(layers, memory, parg)
  GC.@preserve arg begin
    unsafe_valgrad!(g, layers, params, memory, parg)
  end
end

function unsafe_valgrad!(g, layers, params, memory::Vector{UInt8}, arg)
  GC.@preserve g params memory arg begin
    # @show pointer(g) pointer(params) pointer(memory)
    chain_valgrad_entry!(pointer(g), arg, layers, pointer(params), pointer(memory))
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
    for i = CloseOpen(lastdim)
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
function valgrad(sc, arg, params::AbstractVector{T}) where {T}
  c = getchain(sc)
  @unpack layers, memory = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  off = align(resize_memory!(layers, memory, parg))
  GC.@preserve memory arg begin
    g = PtrArray(reinterpret(Ptr{T}, pointer(memory) + off), (static_length(params),))
    l = Base.FastMath.add_fast(
      unsafe_valgrad!(g, layers, params, memory, parg),
      apply_penalty!(g, getpenalty(sc), params, size(parg)),
    )
  end
  return l, StrideArraysCore.StrideArray(g, memory)
end

isstochastic(_) = false
