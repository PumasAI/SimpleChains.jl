
"""
    MaxPool(dims::Tuple{Vararg{Integer}}

Calculates the maximum of pools of size `dims`.
"""
struct MaxPool{D} end
MaxPool(x::Tuple{Vararg{Integer}}) = MaxPool{map(Int, x)}()
MaxPool(x::Vararg{Integer}) = MaxPool{map(Int, x)}()

parameter_free(::MaxPool) = true
function forward_layer_output_size(
  ::Val{T},
  ::MaxPool{D},
  inputdim::Tuple
) where {T,D}
  outdim = getoutputdim(MaxPool{D}(), inputdim)
  align(sizeof(T) * Static.reduce_tup(*, outdim)), outdim
end
function layer_output_size(::Val{T}, ::MaxPool{D}, inputdim::Tuple) where {T,D}
  outdim = getoutputdim(MaxPool{D}(), inputdim)
  align(sizeof(T) * Static.reduce_tup(*, outdim)), outdim
end

_maxpooloutputdim(::Tuple{}, inputdim) = inputdim
function _maxpooloutputdim(d::Tuple{StaticInt,Vararg{StaticInt}}, inputdim)
  head = first(inputdim) ÷ first(d)
  tail = _maxpooloutputdim(Base.tail(d), Base.tail(inputdim))
  (head, tail...)
end
function getoutputdim(::MaxPool{D}, inputdim) where {D}
  _maxpooloutputdim(map(StaticInt, D), inputdim)
end
init_params!(::MaxPool{D}, p, id, ::AbstractRNG) where {D} =
  p, getoutputdim(MaxPool{D}(), id)

numparam(mp::MaxPool, inputdim) = 0, getoutputdim(mp, inputdim)

function maxpoolexpr(d::NTuple{D,Int}, trailing::Int) where {D}
  baseinds = Vector{Expr}(undef, D)
  for i = 1:D
    baseinds[i] = Expr(:call, :(*), d[i], Symbol(:i_, i))
  end
  tomax = Expr[]
  for I in CartesianIndices(d)
    itup = Tuple(I)
    ref = Expr(:ref, :A)
    for i = 1:D
      ind = baseinds[i]
      j = itup[i] - 1
      if j != 0
        ind = Expr(:call, :(+), ind, j)
      end
      push!(ref.args, ind)
    end
    for i = 1:trailing
      push!(ref.args, Symbol(:i_, i + D))
    end
    push!(tomax, ref)
  end
  return tomax
end

function maxreduce!(mp::Vector{Expr})
  while length(mp) > 1
    newlen = length(mp) ÷ 2
    for i = 1:newlen
      mp[i] = Expr(:call, :max, mp[2i-1], mp[2i])
    end
    if isodd(length(mp))
      mp[(newlen+=1)] = mp[end]
    end
    resize!(mp, newlen)
  end
  return only(mp)
end
maxreduce(mp::Vector{Expr}) = maxreduce!(copy(mp))

@generated function maxpool!(
  _B::AbstractArray{T,N},
  _A::AbstractArray{T,N},
  ::MaxPool{D}
) where {D,N,T}
  mp = maxreduce!(maxpoolexpr(D, N - length(D)))
  body = :((Base.Cartesian.@nref $N B i) = $mp)
  for n = 1:N-1
    isym = Symbol(:i_, n)
    body = :(
      for $isym in axes(B, $n)
        $body
      end
    )
  end
  isym = Symbol(:i_, N)
  body = :(@turbo warn_check_args = false for $isym in axes(B, $N)
    $body
  end)
  quote
    A = zero_offsets(_A)
    B = zero_offsets(_B)
    $body
  end
end
function replace_sym(ex::Expr)
  @assert ex.head === :ref
  ret = Expr(:ref, :C)
  for i = 2:length(ex.args)
    push!(ret.args, ex.args[i])
  end
  return ret
end
function _partial_maxpool_expr(
  N::Int,
  @nospecialize(D::NTuple{<:Any,Int}),
  overwrite::Bool
)
  mp = maxpoolexpr(D, N - length(D)::Int)
  mr = maxreduce(mp)
  body = quote
    B̄i = (Base.Cartesian.@nref $N B̄ i)
    mr = $mr
  end
  eq = Expr(:call, :(==), mp[1], :mr)
  eqsym = :eq_1
  push!(body.args, Expr(:(=), eqsym, eq))
  mul = Expr(:call, :(*), eqsym, :B̄i)
  store = overwrite ? mp[1] : replace_sym(mp[1])
  push!(body.args, Expr(:(=), store, mul))
  notfoundsym = :notfound_1
  push!(body.args, Expr(:(=), notfoundsym, Expr(:call, :(!), eqsym)))
  for i = 2:length(mp)
    eqsym = Symbol(:eq_, i)
    eqexpr = Expr(:call, :(&), Expr(:call, :(==), mp[i], :mr), notfoundsym)
    push!(body.args, Expr(:(=), eqsym, eqexpr))
    mul = Expr(:call, :(*), eqsym, :B̄i)
    store = overwrite ? mp[i] : replace_sym(mp[i])
    push!(body.args, Expr(:(=), store, mul))
    if i != length(mp)
      notfoundsym_new = Symbol(:notfound_, i)
      noteq = Expr(:call, :(!), eqsym)
      newnotfound = Expr(:call, :(&), notfoundsym, noteq)
      push!(body.args, Expr(:(=), notfoundsym_new, newnotfound))
      notfoundsym = notfoundsym_new
    end
  end
  for n = 1:N-1
    isym = Symbol(:i_, n)
    body = :(
      for $isym in axes(B̄, $n)
        $body
      end
    )
  end
  isym = Symbol(:i_, N)
  body = :(@turbo for $isym in axes(B̄, $N)
    $body
  end)
  if overwrite
    quote
      A = zero_offsets(_A)
      B̄ = zero_offsets(_B̄)
      $body
    end
  else
    quote
      C = zero_offsets(_C)
      A = zero_offsets(_A)
      B̄ = zero_offsets(_B̄)
      $body
    end
  end
end

@generated function ∂maxpool!(
  _A::AbstractArray{T,N},
  _B̄::AbstractArray{T,N},
  ::MaxPool{D}
) where {D,N,T}
  _partial_maxpool_expr(N, D, true)
end
@generated function ∂maxpool!(
  _C::AbstractArray{T,N},
  _A::AbstractArray{T,N},
  _B̄::AbstractArray{T,N},
  ::MaxPool{D}
) where {D,N,T}
  _partial_maxpool_expr(N, D, false)
end

function (::MaxPool{D})(A::AbstractArray{T}, p::Ptr, pu::Ptr{UInt8}) where {T,D}
  B = PtrArray(Ptr{T}(pu), getoutputdim(MaxPool{D}(), static_size(A)))
  maxpool!(B, A, MaxPool{D}())
  B, p, pu + align(sizeof(T) * length(B))
end
function valgrad_layer!(
  pg::Union{Nothing,Ptr},
  ::MaxPool{D},
  A,
  p,
  pu
) where {D}
  return pg, MaxPool{D}()(A, p, pu)...
end
function pullback_arg!(
  ::MaxPool{D},
  B̄,
  A,
  p::Ptr,
  pu::Ptr{UInt8},
  pu2::Ptr{UInt8}
) where {D}
  ∂maxpool!(A, B̄, MaxPool{D}())
  return A, pu2
end
function pullback_arg!(
  C̄ptr::Ptr,
  ::MaxPool{D},
  B̄,
  A,
  p::Ptr,
  pu::Ptr{UInt8},
  pu2::Ptr{UInt8}
) where {D}
  C̄ = PtrArray(C̄ptr, static_size(B̄))
  ∂maxpool!(C̄, A, B̄, MaxPool{D}())
  return A, pu2
end
