@inline bsub(::Tuple{}, ::Number) = ()
@inline bsub(x::Tuple{T}, y::Number) where {T} = (only(x) - y,)
@inline bsub(x::Tuple{T0,T1,Vararg}, y::Number) where {T0,T1} =
  (first(x) - y, bsub(Base.tail(x), y)...)

@inline badd(::Tuple{}, ::Number) = ()
@inline badd(x::Tuple{T}, y::Number) where {T} = (only(x) + y,)
@inline badd(x::Tuple{T0,T1,Vararg}, y::Number) where {T0,T1} =
  (first(x) + y, badd(Base.tail(x), y)...)

# @inline bmul(::Tuple{}, ::Number) = ()
# @inline bmul(x::Tuple{T}, y::Number) where {T} = (only(x) * y,)
# @inline bmul(x::Tuple{T0,T1,Vararg}, y::Number) where {T0,T1} =
#   (first(x) * y, bmul(Base.tail(x), y)...)

tsprod(x) = ArrayInterface.reduce_tup(*, x)
tsprod(::Tuple{}) = static(1)
tssum(x) = ArrayInterface.reduce_tup(+, x)
tssum(::Tuple{}) = static(0)

function maximum_turbo!(m, y)
  @turbo for i in eachindex(m)
    mi = typemin(eltype(y))
    for j in axes(y, 1)
      mi = max(mi, y[j, i])
    end
    m[i] = mi
  end
end


function unnormalized_logsoftmax!(z, m, y::AbstractMatrix)
  maximum_turbo!(m, y)
  @turbo for j in axes(y, 2)
    mj = m[j]
    s = zero(eltype(m))
    for i in axes(y, 1)
      yij = y[i, j]
      zij = mj == Inf ? (yij == Inf ? zero(eltype(y)) : -Inf) : yij - m[j]
      z[i, j] = zij
      s += exp(zij)
    end
    m[j] = s
  end
  @turbo for i in eachindex(m)
    m[i] = log(m[i])
  end
end
function logsoftmax!(z, m, y::AbstractMatrix)
  unnormalized_logsoftmax!(z, m, y)
  @turbo for j in axes(z, 2), i in axes(z, 1)
    z[i, j] -= m[j]
  end
end
function logsoftmax(y::AbstractMatrix)
  m = similar(y, size(y, 2))
  z = similar(y)
  logsoftmax!(z, m, y)
  return z
end

# https://github.com/FluxML/Flux.jl/blob/master/LICENSE.md
nfan() = static(1), static(1)
nfan(n) = static(1), n
nfan(n_out, n_in) = n_in, n_out
@inline function nfan(dims::Vararg{Any,K}) where {K}
  df = Base.front(dims)
  dt = last(df)
  dff = Base.front(df)
  dft = last(df)
  p = tsprod(dff)
  p * dft, p * dt
end
# https://github.com/FluxML/Flux.jl/blob/master/LICENSE.md
function glorot_uniform!(A::AbstractArray{T}, rng = local_rng()) where {T}
  scale = @fastmath sqrt(T(24) / tssum(nfan(size(A)...)))
  # (rand()-0.5)*scale === rand()*scale - 0.5scale
  rand!(rng, A, static(0), T(-0.5) * scale, scale)
end
# https://github.com/FluxML/Flux.jl/blob/master/LICENSE.md
function glorot_normal!(A::AbstractArray{T}, rng = local_rng()) where {T}
  σ = @fastmath sqrt(T(2) / tssum(nfan(size(A)...)))
  randn!(rng, A, static(0), static(0), σ)
end

function randpermzero!(r::Random.AbstractRNG, a::AbstractArray{<:Integer})
  n = length(a)
  @assert n <= one(Int64) << 52
  n == 0 && return a
  fi = firstindex(a)
  @inbounds a[fi] = 0
  mask = 3
  @inbounds for i = 1:n-1
    sp = Random.ltm52(i, mask)
    while true
      j = rand(r, sp.s)
      j > sp.sup && continue
      if i != j # a[i] is undef (and could be #undef)
        a[fi+i] = a[fi+j]
      end
      a[fi+j] = i
      i % UInt64 == mask && (mask = 2mask + 1)
      break
    end
  end
  return a
end
# function randpermzero!(r::VectorizedRNG.Xoshift, a::AbstractArray{<:Integer})
#   n = length(a)
#   @assert n <= one(Int64) << 52
#   n == 0 && return a
#   fi = firstindex(a)
#   @inbounds a[fi] = 0
#   mask = UInt64(3)
#   s = VectorizedRNG.getstate(r)
#   @inbounds for i = 1:n-1
#     sp = Random.ltm52(i + 1, mask % Int)
#     while true
#       s, u = VectorizedRNG.nextstate(s)
#       j = u & mask
#       j > sp.sup && continue
#       if i != j # a[i] is undef (and could be #undef)
#         a[fi+i] = a[fi+j]
#       end
#       a[fi+j] = i
#       i % UInt64 == mask && (mask = UInt64(2) * mask + one(UInt64))
#       break
#     end
#   end
#   VectorizedRNG.storestate!(r, s)
#   return a
# end
function randpermzero!(
  r::VectorizedRNG.Xoshift,
  a::AbstractArray{<:Integer},
)
  randpermzero!(r, a, VectorizationBase.pick_vector_width(UInt64))
end
function randpermzero!(
  r::VectorizedRNG.Xoshift,
  a::AbstractArray{<:Integer},
  ::StaticInt{W},
) where {W}
    n = length(a)
    @assert n <= one(Int64) << 52
    n == 0 && return a
    fi = firstindex(a)
    @inbounds a[fi] = 0
    mask = UInt64(3)
    s = VectorizedRNG.getstate(r, Val{1}(), StaticInt{W}())
    @inbounds for i = 1:n-1
      sp = Random.ltm52(i + 1, mask % Int)
      while true
        s, uvu = VectorizedRNG.nextstate(s, Val(1))
        u = VectorizationBase.data(uvu)[1]
        jv = u & mask
        m = VectorizationBase.data(jv <= sp.sup)
        iszero(m) && continue
        j = VectorizationBase.extractelement(jv, trailing_zeros(m))
        if i != j # a[i] is undef (and could be #undef)
          a[fi+i] = a[fi+j]
        end
        a[fi+j] = i
        i % UInt64 == mask && (mask = UInt64(2) * mask + one(UInt64))
        break
      end
    end
    VectorizedRNG.storestate!(r, s)
    return a
end
randpermzero!(a::AbstractArray{<:Integer}) = randpermzero!(local_rng(), a)
