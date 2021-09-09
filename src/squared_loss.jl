
struct SquaredLoss{Y}
  y::Y
end

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

output_size(::Val{T}, sl::SquaredLoss, s) where {T} = align(length(sl.y) * static_sizeof(T)), static_sizeof(T)

function (sl::SquaredLoss)(arg, p, pu)
  y = getfield(sl, :y)
  s = zero(promote_type(eltype(arg), eltype(y)))
  @turbo for i ∈ eachindex(arg)
    s += abs2(arg[i] - y[i])
  end
  s, p, pu
end

numparam(::SquaredLoss) = 0
