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
  rand!(rng, A, static(0), T(-0.5)*scale, scale)
end
# https://github.com/FluxML/Flux.jl/blob/master/LICENSE.md
function glorot_normal!(A::AbstractArray{T}, rng = local_rng()) where {T}
  σ = @fastmath sqrt(T(2) / tssum(nfan(size(A)...)))
  randn!(rng, A, static(0), static(0), σ)
end


