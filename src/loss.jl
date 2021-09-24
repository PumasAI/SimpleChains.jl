abstract type AbstractLoss end

numparam(::AbstractLoss) = 0
output_size(::Val{T}, sl::AbstractLoss, s) where {T} = align(length(target(sl)) * static_sizeof(T)), static_sizeof(T)

struct SquaredLoss{Y} <: AbstractLoss
  y::Y
end
target(sl::SquaredLoss) = getfield(sl, :y)

function chain_valgrad!(pg, arg::AbstractArray{T}, layers::Tuple{SquaredLoss}, p::Ptr, pu::Ptr{UInt8}) where {T}
  y = getfield(getfield(layers, 1), :y)
  g = PtrArray(stridedpointer(Base.unsafe_convert(Ptr{T}, pu), bytestrideindex(arg)), size(arg), StrideArraysCore.val_dense_dims(arg))
  s = zero(eltype(g))
  @turbo for i ∈ eachindex(g)
    δ = arg[i] - y[i]
    g[i] = δ
    s += δ*δ
  end
  return 0.5s, g, pu + sizeof(T)*length(g)
end
function (sl::SquaredLoss)(arg, p, pu)
  y = getfield(sl, :y)
  s = zero(promote_type(eltype(arg), eltype(y)))
  @turbo for i ∈ eachindex(arg)
    δ = arg[i] - y[i]
    s += δ*δ
  end
  s, p, pu
end



struct AbsoluteLoss{Y} <: AbstractLoss
  y::Y
end
target(sl::AbsoluteLoss) = getfield(sl, :y)

function chain_valgrad!(pg, arg::AbstractArray{T}, layers::Tuple{AbsoluteLoss}, p::Ptr, pu::Ptr{UInt8}) where {T}
  y = getfield(getfield(layers, 1), :y)
  g = PtrArray(stridedpointer(Base.unsafe_convert(Ptr{T}, pu), bytestrideindex(arg)), size(arg), StrideArraysCore.val_dense_dims(arg))
  s = zero(eltype(g))
  @turbo for i ∈ eachindex(g)
    δ = arg[i] - y[i]
    pos = δ > zero(T)
    g[i] = ifelse(pos, one(T), -one(T))
    s += ifelse(pos, δ, -δ)
  end
  return s, g, pu + sizeof(T)*length(g)
end

function (sl::AbsoluteLoss)(arg, p, pu)
  y = getfield(sl, :y)
  s = zero(promote_type(eltype(arg), eltype(y)))
  @turbo for i ∈ eachindex(arg)
    s += abs(arg[i] - y[i])
  end
  s, p, pu
end


