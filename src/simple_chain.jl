
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
  SimpleChain((static(input_dim),), (lf, lm, lt...))
SimpleChain(input_dim::InputDim, lf, lm, lt::Vararg) =
  SimpleChain(input_dim, (lf, lm, lt...))

SimpleChain(input_dim::Integer, l::Tuple) = SimpleChain((static(input_dim),), l)
SimpleChain(input_dim::Integer, l::Vararg) =
  SimpleChain((static(input_dim),), l)

SimpleChain(l::Vararg) = SimpleChain(InputDimUnknown(), l)
SimpleChain(l::Tuple) = SimpleChain(InputDimUnknown(), l)

Base.similar(c::SimpleChain) = SimpleChain(chain_input_dims(c), c.layers)

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
  SimpleChain(chain_input_dims(c), Base.front(c.layers))
Base.vcat(c::SimpleChain, l) =
  SimpleChain(chain_input_dims(c), (c.layers..., l))

@inline function numparam(c::SimpleChain, id = nothing)
  _id = chain_input_dims(c, id)
  _numparam(static(0), c.layers, _id)
end
if VERSION >= v"1.7.0" && hasfield(Method, :recursion_relation)
  @inline _numparam(s, ::Tuple{}, _) = s
  @inline function _numparam(s, layers::Tuple{L,Vararg}, id) where {L}
    np, od = numparam(getfield(layers, 1), id)
    _numparam(s + np, Base.tail(layers), od)
  end
else
  @generated function _numparam(
    np_0,
    layers::Tuple{Vararg{Any,N}},
    id_0
  ) where {N}
    N == 0 && return :np_0
    q = Expr(:block, Expr(:meta, :inline))
    prev_np = :np_0
    prev_id = :id_0
    for n = 1:N
      tmp = Symbol(:tmp_, n)
      next_id = Symbol(:id_, n)
      next_np = Symbol(:np_, n)
      push!(
        q.args,
        :(($tmp, $next_id) = numparam(getfield(layers, $n), $prev_id))
      )
      push!(q.args, :($next_np = $tmp + $prev_np))
      prev_np = next_np
      prev_id = next_id
    end
    return q
  end
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
    throw(
      ArgumentError("Input argument: !matches(chain_input_dims(c), size(arg))")
    )
  end
end
struct SArrayOutput{T}
  f::T
end
@inline function (f::SArrayOutput)(x::Vararg{Any,K}) where {K}
  fx = f.f(x...)
  _maybe_sarray(fx, size(fx))
end

function (c::SimpleChain)(
  arg::AbstractArray{T0},
  params::AbstractVector{T1}
) where {T0,T1}
  verify_arg(c, arg)
  @unpack layers = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  num_bytes = required_forward_bytes(
    Val(promote_type(T0, T1)),
    layers,
    size(parg),
    static(0)
  )
  if has_loss(c)
    GC.@preserve arg params begin
      lret = with_memory(_chain, c, num_bytes, parg, pointer(params))
    end
    return lret
  else
    ol = tsprod(outputdim(c, size(arg)))
    if ol isa StaticInt &&
       num_bytes isa StaticInt &&
       ol < 64 &&
       num_bytes <= MAXSTACK
      GC.@preserve arg params begin
        saret = with_stack_memory(
          SArrayOutput(_chain),
          num_bytes,
          c,
          parg,
          pointer(params)
        )
      end
      return saret
    else
      GC.@preserve arg params begin
        res, heap_memory =
          with_heap_memory(_chain, c, num_bytes, parg, pointer(params))
        sret = StrideArray(res, heap_memory)
      end
      return sret
    end
  end
end
@inline _maybe_sarray(x) = x
@inline _maybe_sarray(x::AbstractArray) = _maybe_sarray(x, size(x))
@inline _maybe_sarray(x::AbstractArray, _) = x
@inline _maybe_sarray(A::AbstractArray, s::Tuple{Vararg{StaticInt}}) =
  _to_sarray(A, s)
@generated function _marray_type(s::Tuple{Vararg{StaticInt}})
  k = known(s)
  ct = Expr(:curly, :Tuple)
  for x in k
    push!(ct.args, x)
  end
  :($StaticArrays.MArray{$ct})
end
@inline function _to_sarray(
  A::AbstractArray{T},
  s::Tuple{Vararg{StaticInt}}
) where {T}
  B = _marray_type(s){T}(undef)
  if T <: Base.HWReal
    @turbo for i in eachindex(B)
      B[i] = A[i]
    end
  elseif Base.isbitstype(T)
    GC.@preserve B A begin
      ccall(
        :memmove,
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
        pointer(B),
        pointer(A),
        (length(B) * Base.aligned_sizeof(T)) % UInt
      )
    end
  else
    copyto!(B, A)
  end
  StaticArrays.SArray(B)
end
# @generated function _maybe_sarray(A::AbstractArray, s::Tuple{Vararg{StaticInt}})
#   k = known(s)
#   t = Expr(:tuple)
#   ct = Expr(:curly, :Tuple)
#   for x in k
#     push!(ct.args, x)
#   end
#   K = prod(k)::Int;
#   for i = 1:K
#     push!(t.args, :(unsafe_load(p, $i)))
#   # for i = 0:K-1
#     # push!(t.args, :($(VectorizationBase.vload)(p, (static($i),))))
#   end
#   Expr(
#     :block,
#     Expr(:meta, :inline),
#     # :(p = $zstridedpointer(A)),
#     :(p = pointer(A)),
#     :(B = GC.@preserve A StaticArrays.SArray{$ct}($t)),
#     :($(VectorizationBase.lifetime_end!)(p, Val{$K}())),
#     :B
#   )
# end
@inline function (c::SimpleChain)(
  arg::StaticArrays.SArray,
  params::StaticArrays.SArray
)
  mparams = StaticArrays.MArray(params)
  @gc_preserve c(arg, mparams)
end
@inline function (c::SimpleChain)(
  arg::StaticArrays.SArray,
  params::AbstractVector
)
  verify_arg(c, arg)
  @unpack layers = c
  marg = StaticArrays.MArray(arg)
  GC.@preserve marg params begin
    parg = maybe_static_size_arg(c.inputdim, marg)
    num_bytes = required_forward_bytes(
      Val(Base.promote_eltype(arg, params)),
      layers,
      size(parg),
      static(0)
    )
    if has_loss(c)
      with_memory(_chain, c, num_bytes, parg, pointer(params))
    else
      with_memory(_sarray_chain, c, num_bytes, parg, pointer(params))
    end
  end
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

@static if VERSION >= v"1.7.0" && hasfield(Method, :recursion_relation)
  @inline output_size(::Val{T}, x::Tuple{}, _) where {T} = static(0)
  @inline function output_size(::Val{T}, x::Tuple{X}, s1) where {T,X}
    first(layer_output_size(Val{T}(), getfield(x, 1), s1))
  end
  @inline function output_size(
    ::Val{T},
    x::Tuple{X1,X2,Vararg},
    s1::Tuple
  ) where {T,X1,X2}
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
    s1::Tuple
  ) where {T,X1,X2}
    b, s2 = forward_layer_output_size(Val{T}(), getfield(x, 1), s1)
    b + forward_output_size(Val{T}(), Base.tail(x), s2)
  end

  @inline __chain(::Tuple{}, arg, p::Ptr, pu::Ptr{UInt8}) = arg
  @inline __chain(l::Tuple{T}, arg, p::Ptr, pu::Ptr{UInt8}) where {T} =
    getfield(getfield(l, 1)(arg, p, pu), 1)
  @inline function __chain(
    l::Tuple{T1,T2,Vararg},
    arg,
    p::Ptr,
    pu::Ptr{UInt8}
  ) where {T1,T2}
    res, p, pu = getfield(l, 1)(arg, p, pu)
    __chain(Base.tail(l), res, p, pu)
  end
else
  function _output_size_expr(VT::Expr, N::Int, f::Symbol)
    N == 0 && return static(0)
    q = Expr(:block, Expr(:meta, :inline))
    prev_s = :s_0
    prev_a = :acc_0
    for n = 1:N
      next_s = Symbol(:s_, n)
      next_a = Symbol(:acc_, n)
      tmp = n == 1 ? next_a : Symbol(:tmp_, n)
      push!(q.args, :(($tmp, $next_s) = $f($VT, getfield(x, $n), $prev_s)))
      n == 1 || push!(q.args, :($next_a = $prev_a + $tmp))
      prev_s = next_s
      prev_a = next_a
    end
    push!(q.args, prev_a)
    return q
  end
  @generated function output_size(
    ::Val{T},
    x::Tuple{Vararg{Any,N}},
    s_0
  ) where {T,N}
    _output_size_expr(:(Val{$T}()), N, :layer_output_size)
  end
  @generated function forward_output_size(
    ::Val{T},
    x::Tuple{Vararg{Any,N}},
    s_0
  ) where {T,N}
    _output_size_expr(:(Val{$T}()), N, :forward_layer_output_size)
  end

  @generated function __chain(
    l::Tuple{Vararg{Any,N}},
    res_0,
    p_0::Ptr,
    pu_0::Ptr{UInt8}
  ) where {N}
    N == 0 && return :res_0
    q = Expr(:block, Expr(:meta, :inline))
    prev_p = :p_0
    prev_pu = :pu_0
    prev_res = :res_0
    for n = 1:N
      next_p = Symbol(:p, n)
      next_pu = Symbol(:pu, n)
      next_res = Symbol(:res, n)
      push!(
        q.args,
        :(
          ($next_res, $next_p, $next_pu) =
            getfield(l, $n)($prev_res, $prev_p, $prev_pu)
        )
      )
      prev_p = next_p
      prev_pu = next_pu
      prev_res = next_res
    end
    push!(q.args, prev_res)
    return q
  end
end

@inline function _chain(c::Chain, pu::Ptr{UInt8}, arg, p)
  @unpack layers = c
  __chain(layers, arg, p, pu)
end

@inline function _sarray_chain(c::Chain, pu::Ptr{UInt8}, arg, p)
  @unpack layers = c
  _maybe_sarray(__chain(layers, arg, p, pu))
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
chain_input_dims(
  ::SimpleChain{InputDimUnknown},
  inputdim::Tuple{Vararg{Integer}}
) = inputdim
function chain_input_dims(::SimpleChain{InputDimUnknown}, ::Nothing)
  throw(
    ArgumentError(
      "SimpleChains without an explicitly provided InputDim require manually providing it when calling `init_params`"
    )
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
@inline function init_params!(
  x::AbstractVector,
  chn::SimpleChain,
  id = nothing;
  rng::AbstractRNG = local_rng()
)
  GC.@preserve x _init_params!(
    chn.layers,
    pointer(x),
    chain_input_dims(chn, id),
    rng
  )
  return x
end
function _init_params!(layers::Tuple, p::Ptr, id, rng::AbstractRNG)
  p, od = init_params!(first(layers), p, id, rng)
  _init_params!(Base.tail(layers), p, od, rng)
end
_init_params!(::Tuple{}, p::Ptr, _, ::AbstractRNG) = nothing
@inline function init_params(
  Λ::SimpleChain,
  id::Union{Nothing,InputDim} = nothing,
  ::Type{T} = Float32;
  rng::AbstractRNG = local_rng()
) where {T}
  _id = chain_input_dims(Λ, id)
  init_params!(
    StrideArray{T}(undef, numparam(Λ, id)),
    Λ,
    chain_input_dims(Λ, _id);
    rng
  )
end

"""
    SimpleChains.init_params(chn[, id = nothing][, ::Type{T} = Float32])

Creates a parameter vector of element type `T` with size matching that by `id` (argument not required if provided to the `chain` object itself).
See the documentation of the individual layers to see how they are initialized, but it is generally via (Xavier) Glorot uniform or normal distributions.
"""
function init_params(
  Λ::SimpleChain,
  ::Type{T};
  rng::AbstractRNG = local_rng()
) where {T}
  init_params(Λ, nothing, T; rng)
end

@inline function maybe_static_size_arg(s::Tuple, arg)
  if SimpleChains.ArrayInterface.device(arg) === CPUPointer()
    PtrArray(pointer(arg), _try_static(s, size(arg)))
  else
    arg
  end
end
@inline function maybe_static_size_arg(s::InputDimUnknown, arg)
  if SimpleChains.ArrayInterface.device(arg) === CPUPointer()
    PtrArray(pointer(arg), size(arg))
  else
    arg
  end
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
function valgrad!(
  g,
  c::SimpleChain,
  arg::AbstractArray{T0},
  params::AbstractVector{T1}
) where {T0,T1}
  verify_arg(c, arg)
  @assert has_loss(c)
  @unpack layers = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  num_bytes =
    required_bytes(Val{promote_type(T0, T1)}(), layers, size(parg), static(0))
  GC.@preserve arg with_memory(unsafe_valgrad!, c, num_bytes, g, params, parg)
end

function unsafe_valgrad!(c::Chain, pu::Ptr{UInt8}, g, params, arg)
  @unpack layers = c
  GC.@preserve g params begin
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
  p1::Ptr,
  pu::Ptr{UInt8}
) where {X1,X2}
  l = getfield(layers, 1)
  pg2, larg, p2, pu2 = valgrad_layer!(pg, l, arg, p1, pu)
  if parameter_free(l)
    val = chain_valgrad_entry!(pg2, larg, Base.tail(layers), p2, pu2)
  else
    val, grad, _ = chain_valgrad!(pg2, larg, Base.tail(layers), p2, pu2)
    pullback_param!(pg, l, grad, arg, p1, pu)
  end
  return val
end
function chain_valgrad_entry!(
  pg,
  arg,
  layers::Tuple{X1,X2,Vararg},
  inds,
  p::Ptr,
  pu::Ptr{UInt8}
) where {X1,X2}
  arg_subset, pu = subset_batch(arg, inds, pu)
  chain_valgrad_entry!(pg, arg_subset, layers, p, pu)
end

@static if VERSION >= v"1.7.0" && hasfield(Method, :recursion_relation)
  function chain_valgrad!(
    pg,
    arg,
    layers::Tuple{X1,X2,Vararg},
    p::Ptr,
    pu::Ptr{UInt8}
  ) where {X1,X2}
    l = getfield(layers, 1)
    pg2, larg, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)
    val, grad, pu3 = chain_valgrad!(pg2, larg, Base.tail(layers), p2, pu2)
    lgrad, pu4 = pullback!(pg, l, grad, arg, p, pu, pu3)
    return val, lgrad, pu4
  end
  function chain_valgrad!(
    pg,
    arg,
    layers::Tuple{X},
    p::Ptr,
    pu::Ptr{UInt8}
  ) where {X}
    l = getfield(layers, 1)
    __, val, _, pu2 = valgrad_layer!(pg, l, arg, p, pu)
    # val, pullback, p2, pu2 = valgrad_layer!(pg, l, arg, p, pu)
    lgrad, pu3 = pullback!(pg, l, One(), arg, p, pu, pu2)
    return val, lgrad, pu3
  end
else
  function _chain_valgrad_expr(n::Int, r::Int)
    # note n + r == N
    q = Expr(:block)
    ((n == 0) & (r == 1)) && push!(q.args, Expr(:meta, :inline))
    pg_now = Symbol(:pg_, n)
    pg_next = Symbol(:pg_, n + 1)
    arg_now = Symbol(:arg_, n)
    arg_next = Symbol(:arg_, n + 1)
    p_now = Symbol(:p_, n)
    p_next = Symbol(:p_, n + 1)
    pu_now = Symbol(:pu_, n)
    pu_next = Symbol(:pu_, n + 1)
    layer = Symbol(:layer_, n)
    push!(q.args, :($layer = getfield(layers, $(n + 1))))
    push!(
      q.args,
      :(
        ($pg_next, $arg_next, $p_next, $pu_next) =
          valgrad_layer!($pg_now, $layer, $arg_now, $p_now, $pu_now)
      )
    )
    pu_final = Symbol(:pu_, 2 * n + r)
    if r == 1
      grad_next = :(One())
      pu_pullback = pu_next
    else
      pu_pullback = Symbol(:pu_, 2n + r + 1)
      grad_next = Symbol(:grad_, n + 1)
      if r == 2
        # Many loss functions are implemented in terms of chain_valgrad, so we
        # do have to recurse here in case we ought to dispatch to a specialized method
        push!(
          q.args,
          :(
            ($(Symbol(:arg_, n + r)), $grad_next, $pu_pullback) =
              chain_valgrad!(
                $pg_next,
                $arg_next,
                (getfield(layers, $(n + r)),),
                $p_next,
                $pu_next
              )
          )
        )
      else
        # pu_pullback is pu_final of n+=1, r-=1
        push!(q.args, _chain_valgrad_expr(n + 1, r - 1))
      end
    end
    grad_now = Symbol(:grad_, n)
    push!(
      q.args,
      :(
        ($grad_now, $pu_final) = pullback!(
          $pg_now,
          $layer,
          $grad_next,
          $arg_now,
          $p_now,
          $pu_now,
          $pu_pullback
        )
      )
    )
    if n == 0 # we're done
      push!(q.args, Expr(:tuple, Symbol(:arg_, r), grad_now, pu_final))
    end
    return q
  end
  @generated function chain_valgrad!(
    pg_0,
    arg_0,
    layers::Tuple{Vararg{Any,N}},
    p_0::Ptr,
    pu_0::Ptr{UInt8}
  ) where {N}
    _chain_valgrad_expr(0, N)
  end
end

@inline getchain(sc::SimpleChain) = sc
function valgrad_core(
  c::Chain,
  pu::Ptr{UInt8},
  arg,
  params::AbstractVector{T},
  glen
) where {T}
  @unpack layers = c
  g = PtrArray(Ptr{T}(pu), (glen,))
  l = unsafe_valgrad!(c, pu + align(glen * static_sizeof(T)), g, params, arg)
  Base.FastMath.add_fast(l, apply_penalty!(g, getpenalty(c), params, size(arg)))
end
function valgrad_core_sarray(
  c::Chain,
  pu::Ptr{UInt8},
  arg,
  params::AbstractVector{T},
  ::StaticInt{L}
) where {T,L}
  @unpack layers = c
  g = PtrArray(Ptr{T}(pu), (static(L),))
  l = Base.FastMath.add_fast(
    unsafe_valgrad!(
      c,
      pu + align(static(L) * static_sizeof(T)),
      g,
      params,
      arg
    ),
    apply_penalty!(g, getpenalty(c), params, size(arg))
  )
  return l, _maybe_sarray(g, (static(L),))
end
function valgrad(sc::Chain, arg, params::AbstractVector{TP}) where {TP}
  c = getchain(sc)
  @unpack layers = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  glen = _try_static(numparam(sc, size(parg)), static_length(params))
  T = Base.promote_eltype(arg, params)
  num_bytes =
    required_bytes(Val{T}(), layers, size(parg), glen * static_sizeof(T))
  l, heap_memory =
    with_heap_memory(valgrad_core, sc, num_bytes, parg, params, glen)
  gv = StrideArraysCore.StrideArray(
    PtrArray(align(Ptr{TP}(pointer(heap_memory))), (glen,)),
    heap_memory
  )
  return l, gv
end
@inline function valgrad(
  sc::Chain,
  arg::StaticArrays.SArray,
  params::AbstractVector{TP}
) where {TP}
  c = getchain(sc)
  @unpack layers = c
  parg = maybe_static_size_arg(c.inputdim, arg)
  glen = _try_static(numparam(sc), static_length(params))
  T = Base.promote_eltype(arg, params)
  num_bytes =
    required_bytes(Val{T}(), layers, size(parg), glen * static_sizeof(T))
  if glen isa StaticInt
    return with_memory(valgrad_core_sarray, sc, num_bytes, parg, params, glen)
  else
    l, heap_memory =
      with_heap_memory(valgrad_core, sc, num_bytes, parg, params, glen)
    gv = StrideArraysCore.StrideArray(
      PtrArray(Ptr{TP}(pointer(heap_memory)), (glen,)),
      heap_memory
    )
    return l, gv
  end
end

isstochastic(_) = false
