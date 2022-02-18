abstract type AbstractLoss{Y} end

has_loss(sc::SimpleChain) = last(sc.layers) isa AbstractLoss
function add_loss(sc::SimpleChain, l::AbstractLoss)
  has_loss(sc) ? SimpleChain((Base.front(sc.layers)...,l), sc.memory) : SimpleChain((sc.layers...,l), sc.memory)
end
function remove_loss(sc::SimpleChain)
  has_loss(sc) ? Base.front(sc) : sc
end

target(_) = nothing
target(sc::SimpleChain) = target(last(sc.layers))
_iterate_over_losses(::AbstractArray{<:AbstractArray}) = true
_iterate_over_losses(_) = false
iterate_over_losses(sc) = _iterate_over_losses(target(sc))

parameter_free(::AbstractLoss) = true
numparam(::AbstractLoss) = 0
function output_size(::Val{T}, sl::AbstractLoss{<:AbstractArray{<:AbstractArray}}, s) where {T}
  align(length(first(target(sl))) * static_sizeof(T)), static_sizeof(T)
end
output_size(::Val{T}, sl::AbstractLoss, s) where {T} = align(length(target(sl)) * static_sizeof(T)), static_sizeof(T)

struct SquaredLoss{Y} <: AbstractLoss{Y}
  y::Y
end
(::SquaredLoss)(y) = SquaredLoss(y)
SquaredLoss() = SquaredLoss(nothing)
target(sl::SquaredLoss) = getfield(sl, :y)
init_params!(::AbstractLoss, p) = p

squared_loss(chn::SimpleChain, y) = add_loss(chn, SquaredLoss(y))

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
function (sl::SquaredLoss{<:AbstractArray{<:Number}})(arg, p, pu)
  y = getfield(sl, :y)
  s = zero(promote_type(eltype(arg), eltype(y)))
  @turbo for i ∈ eachindex(arg)
    δ = arg[i] - y[i]
    s += δ*δ
  end
  s, p, pu
end


struct AbsoluteLoss{Y} <: AbstractLoss{Y}
  y::Y
end
(::AbsoluteLoss)(y) = AbsoluteLoss(y)
AbsoluteLoss() = AbsoluteLoss(nothing)
target(sl::AbsoluteLoss) = getfield(sl, :y)

absolute_loss(chn::SimpleChain, y) = add_loss(chn, AbsoluteLoss(y))


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

function (sl::AbsoluteLoss{<:AbstractArray{<:Number}})(arg, p, pu)
  y = getfield(sl, :y)
  s = zero(promote_type(eltype(arg), eltype(y)))
  @turbo for i ∈ eachindex(arg)
    s += abs(arg[i] - y[i])
  end
  s, p, pu
end
function (sl::AbstractLoss{<:AbstractArray{<:AbstractArray}})(arg, p, pu)
  y = getfield(sl, :y)
  s = zero(promote_type(eltype(arg), eltype(first(y))))
  for yᵢ ∈ y
    sᵢ, p, pu = sl(yᵢ)(arg, p, pu)
    @fastmath s += sᵢ
  end
  return s, p, pu
end


