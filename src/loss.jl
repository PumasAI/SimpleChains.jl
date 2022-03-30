abstract type AbstractLoss{Y} end

has_loss(sc::SimpleChain) = last(sc.layers) isa AbstractLoss
function add_loss(sc::SimpleChain, l::AbstractLoss)
  id = chain_input_dims(sc)
  if has_loss(sc)
    SimpleChain(id, (Base.front(sc.layers)...,l), sc.memory)
  else
    SimpleChain(id, (sc.layers...,l), sc.memory)
  end
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
numparam(::AbstractLoss, _) = 0, 1
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
init_params!(::AbstractLoss, p, _) = p, 1

function Base.getindex(sl::SquaredLoss, r)
  SquaredLoss(view(target(sl), r))
end

squared_loss(chn::SimpleChain, y) = add_loss(chn, SquaredLoss(y))

Base.show(io::IO, ::SquaredLoss) = print(io, "SquaredLoss")

function chain_valgrad!(_, arg::AbstractArray{T}, layers::Tuple{SquaredLoss}, p::Ptr, pu::Ptr{UInt8}) where {T}
  y = getfield(getfield(layers, 1), :y)
  # g = PtrArray(stridedpointer(Base.unsafe_convert(Ptr{T}, pu), bytestrideindex(arg)), size(arg), VectorizationBase.val_dense_dims(arg))
  s = zero(T)
  @turbo for i ∈ eachindex(arg)
    δ = arg[i] - y[i]
    arg[i] = δ
    s += δ*δ
  end
  return T(0.5)*s, arg, pu# + sizeof(T)*length(g)
end
function (sl::SquaredLoss{<:AbstractArray{<:Number}})(arg, p, pu)
  y = getfield(sl, :y)
  T = Base.promote_eltype(arg, y)
  s = zero(T)
  @turbo for i ∈ eachindex(arg)
    δ = arg[i] - y[i]
    s += δ*δ
  end
  T(0.5)*s, p, pu
end


struct AbsoluteLoss{Y} <: AbstractLoss{Y}
  y::Y
end
(::AbsoluteLoss)(y) = AbsoluteLoss(y)
AbsoluteLoss() = AbsoluteLoss(nothing)
target(sl::AbsoluteLoss) = getfield(sl, :y)

absolute_loss(chn::SimpleChain, y) = add_loss(chn, AbsoluteLoss(y))

Base.show(io::IO, ::AbsoluteLoss) = print(io, "AbsoluteLoss")

function Base.getindex(sl::AbsoluteLoss, r)
  AbsoluteLoss(view(target(sl), r))
end
function chain_valgrad!(__, arg::AbstractArray{T}, layers::Tuple{AbsoluteLoss}, _::Ptr, pu::Ptr{UInt8}) where {T}
  y = getfield(getfield(layers, 1), :y)
  # g = PtrArray(stridedpointer(Base.unsafe_convert(Ptr{T}, pu), bytestrideindex(arg)), size(arg), VectorizationBase.val_dense_dims(arg))
  s = zero(eltype(arg))
  @turbo for i ∈ eachindex(arg)
    δ = arg[i] - y[i]
    pos = δ > zero(T)
    arg[i] = ifelse(pos, one(T), -one(T))
    s += ifelse(pos, δ, -δ)
  end
  return s, arg, pu
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


struct LogitCrossEntropyLoss{Y<:AbstractVector{UInt32}}
  y::Y
end
LogitCrossEntropyLoss() = LogitCrossEntropyLoss(nothing)
target(sl::LogitCrossEntropyLoss) = getfield(sl, :y)

function (lcel::LogitCrossEntropyLoss)(arg::AbstractArray{T}, p::Ptr, pu) where {T}
  y = lcel.y
  N = length(y)
  m = PtrArray(Ptr{T}(pu), (N,))
  unnormalized_logsoftmax!(arg, m, arg)
  s = zero(T)
  @turbo for i = eachindex(y)
    s -= arg[y[i],i] - m[i]
  end
  s / N, p, pu
end
function chain_valgrad!(
  __, arg::AbstractArray{T},
  layers::Tuple{LogitCrossEntropyLoss},
  _::Ptr, pu::Ptr{UInt8}
) where {T}
  y = getfield(getfield(layers, 1), :y)
  N = length(y)
  m = PtrArray(Ptr{T}(pu), (N,))
  logsoftmax!(arg, m, arg)
  s = zero(T)
  @turbo for i = eachindex(y)
    s -= arg[y[i],i]
  end
  Ninv = inv(T(N))
  @turbo for i = eachindex(arg)
    arg[i] = exp(arg[i])*Ninv
  end
  @turbo for i = eachindex(y)
    arg[y[i],i] -= Ninv
  end
  return s*Ninv, arg, pu
end
function Base.getindex(sl::LogitCrossEntropyLoss, r)
  LogitCrossEntropyLoss(view(target(sl), r))
end
