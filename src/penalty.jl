
function (Λ::AbstractPenalty{<:SimpleChain})(arg, params)
  Base.FastMath.add_fast(
    getchain(Λ)(arg, params),
    apply_penalty(Λ, params, size(arg))
  )
end
function valgrad!(g, Λ::AbstractPenalty{<:SimpleChain}, arg, params)
  Base.FastMath.add_fast(
    valgrad!(g, getchain(Λ), arg, params),
    apply_penalty!(g, Λ, params, size(arg))
  )
end

_penalty_applied_to_sc(_::IO, ::Nothing) = nothing
function _penalty_applied_to_sc(io::IO, sc::SimpleChain)
  println(io, " applied to:")
  show(io, sc)
end
function Base.show(io::IO, p::AbstractPenalty)
  print(io, string(Base.typename(typeof(p)))[begin+9:end-1])
  λ = getλ(p)
  λ === nothing || print(io, " (λ=$λ)")
  _penalty_applied_to_sc(io, getchain(p))
end
alloc_threaded_grad(
  c::AbstractPenalty,
  id::Union{Nothing,InputDim} = nothing,
  ::Type{T} = Float32;
  numthreads = _numthreads()
) where {T} = alloc_threaded_grad(getchain(c), id, T; numthreads)

UnPack.unpack(c::AbstractPenalty{<:SimpleChain}, ::Val{:layers}) =
  getfield(getchain(c), :layers)
UnPack.unpack(c::AbstractPenalty{<:SimpleChain}, ::Val{:memory}) =
  getfield(getchain(c), :memory)

Base.front(Λ::AbstractPenalty) = Base.front(getchain(Λ))
numparam(Λ::AbstractPenalty, id = nothing) = numparam(getchain(Λ), id)

remove_loss(Λ::AbstractPenalty) = remove_loss(getchain(Λ))
_type_sym(c::Chain) = __type_sym(remove_loss(c))

function init_params(
  Λ::AbstractPenalty,
  id::Union{Nothing,InputDim} = nothing,
  ::Type{T} = Float32;
  rng::AbstractRNG = local_rng()
) where {T}
  init_params(getchain(Λ), id, T; rng)
end
function init_params(
  Λ::AbstractPenalty,
  ::Type{T};
  rng::AbstractRNG = local_rng()
) where {T}
  init_params(getchain(Λ), nothing, T; rng)
end
function init_params!(
  x,
  Λ::AbstractPenalty,
  id = nothing;
  rng::AbstractRNG = local_rng()
)
  init_params!(x, getchain(Λ), id; rng)
end

target(c::AbstractPenalty) = target(getchain(c))

struct NoPenalty{NN} <: AbstractPenalty{NN}
  chn::NN
end
getchain(p::NoPenalty) = getfield(p, :chn)
NoPenalty() = NoPenalty(nothing)
apply_penalty(::NoPenalty, _) = Static.Zero()
apply_penalty!(_, ::NoPenalty, __) = Static.Zero()
(::NoPenalty)(chn::SimpleChain) = NoPenalty(chn)
getpenalty(sc::SimpleChain) = NoPenalty(sc)
getpenalty(Λ::AbstractPenalty) = Λ
getλ(::NoPenalty) = nothing
Base.:(/)(Λ::NoPenalty, ::Number) = Λ

@inline apply_penalty(Λ, p, _) = apply_penalty(Λ, p)
@inline apply_penalty!(g, Λ, p, _) = apply_penalty!(g, Λ, p)

"""
    L1Penalty(λ)

Applies a L1 penalty of `λ` to parameters, i.e. penalizing by their absolute value.
"""
struct L1Penalty{NN,T} <: AbstractPenalty{NN}
  chn::NN
  λ::T
end
getchain(p::L1Penalty) = getfield(p, :chn)
L1Penalty(λ::Number) = L1Penalty(nothing, λ)
L1Penalty(p::AbstractPenalty, λ) = L1Penalty(getchain(p), λ)
getλ(p::L1Penalty) = getfield(p, :λ)
(p::L1Penalty)(chn::SimpleChain) = L1Penalty(chn, p.λ)
Base.:(/)(Λ::L1Penalty, x::Number) = L1Penalty(Λ.chn, Λ.λ / x)

@inline function apply_penalty(Λ::L1Penalty, p::AbstractVector{T}) where {T}
  l = zero(T)
  @turbo for i ∈ eachindex(p) # add penalty
    l += abs(p[i])
  end
  Base.FastMath.mul_fast(l, T(Λ.λ))
end
function apply_penalty!(
  g::AbstractVector{T1},
  Λ::L1Penalty,
  p::AbstractVector{T2}
) where {T1,T2}
  T = promote_type(T1, T2)
  l = zero(T)
  λ = T(Λ.λ)
  @turbo for i ∈ eachindex(g) # add penalty
    pᵢ = p[i]
    pos = pᵢ ≥ zero(T2)
    λᵢ = ifelse(pos, λ, -λ)
    l += λᵢ * pᵢ
    g[i] += λᵢ
  end
  l
end

"""
    L2Penalty(λ)

Applies a L2 penalty of `λ` to parameters, i.e. penalizing by their squares.
"""
struct L2Penalty{NN,T} <: AbstractPenalty{NN}
  chn::NN
  λ::T
end
getchain(p::L2Penalty) = getfield(p, :chn)
L2Penalty(λ) = L2Penalty(nothing, λ)
L2Penalty(p::AbstractPenalty, λ) = L2Penalty(getchain(p), λ)
getλ(p::L2Penalty) = getfield(p, :λ)
(p::L2Penalty)(chn::SimpleChain) = L2Penalty(chn, p.λ)
Base.:(/)(Λ::L2Penalty, x::Number) = L2Penalty(Λ.chn, Λ.λ / x)

@inline function apply_penalty(Λ::L2Penalty, p::AbstractVector{T}) where {T}
  l = zero(T)
  @turbo for i ∈ eachindex(p) # add penalty
    pᵢ = p[i]
    l += pᵢ * pᵢ
  end
  Base.FastMath.mul_fast(l, T(Λ.λ))
end
function apply_penalty!(
  g::AbstractVector{T1},
  Λ::L2Penalty,
  p::AbstractVector{T2}
) where {T1,T2}
  T = promote_type(T1, T2)
  l = zero(T)
  λ = T(Λ.λ)
  @turbo for i ∈ eachindex(g) # add penalty
    pᵢ = p[i]
    λpᵢ = λ * pᵢ
    l += λpᵢ * pᵢ
    g[i] += 2 * λpᵢ
  end
  l
end

"""
    FrontLastPenalty(SimpleChain, frontpen(λ₁...), lastpen(λ₂...))

Applies `frontpen` to all but the last layer, applying `lastpen` to the last layer instead.
"Last layer" here ignores the loss function, i.e. if the last element of the chain is a loss layer,
the then `lastpen` applies to the layer preceding this.
"""
struct FrontLastPenalty{
  NN<:Union{SimpleChain,Nothing},
  P1<:AbstractPenalty{Nothing},
  P2<:AbstractPenalty{Nothing}
} <: AbstractPenalty{NN}
  chn::NN
  front::P1
  last::P2
end
getchain(p::FrontLastPenalty) = getfield(p, :chn)
FrontLastPenalty(λ₁, λ₂) = FrontLastPenalty(nothing, λ₁, λ₂)
FrontLastPenalty(p::AbstractPenalty, λ₁, λ₂) =
  FrontLastPenalty(getchain(p), λ₁, λ₂)
(p::FrontLastPenalty)(chn::SimpleChain) = FrontLastPenalty(chn, p.front, p.last)
Base.:(/)(Λ::FrontLastPenalty, x::Number) =
  FrontLastPenalty(Λ.chn, Λ.front / x, Λ.last / x)

function Base.show(io::IO, p::FrontLastPenalty)
  print(io, "Penalty on all but last layer: ")
  show(io, p.front)
  print(io, "\nPenalty on last layer: ")
  show(io, p.last)
  _penalty_applied_to_sc(io, getchain(p))
end

function split_front_last(c::SimpleChain)
  l = c.layers
  split_front_last(Base.front(l), last(l))
end
split_front_last(l::Tuple, x) = (l, x)
split_front_last(l::Tuple, ::AbstractLoss) = (Base.front(l), last(l))
function front_param_lens(c::SimpleChain, id)
  f, _ = split_front_last(c)
  _numparam(0, f, id)
end

@inline function apply_penalty(Λ::FrontLastPenalty{<:SimpleChain}, param, id)
  f = front_param_lens(getchain(Λ), id)

  Base.FastMath.add_fast(
    apply_penalty(Λ.front, view(param, 1:f)),
    apply_penalty(Λ.last, view(param, f+1:length(param)))
  )
end
@inline function apply_penalty!(
  grad,
  Λ::FrontLastPenalty{<:SimpleChain},
  param,
  id
)
  f = front_param_lens(getchain(Λ), id)
  fr = 1:f
  lr = 1+f:length(param)
  Base.FastMath.add_fast(
    apply_penalty!(view(grad, fr), Λ.front, view(param, fr)),
    apply_penalty!(view(grad, lr), Λ.last, view(param, lr))
  )
end

params(sc::AbstractPenalty, p::AbstractVector, inputdim = nothing) =
  params(getchain(sc), p, inputdim)
