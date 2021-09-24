
"""
    AbstractPenalty

The `AbstractPenalty` interface requires supporting the following methods:

1. `getchain(::AbstractPenalty)::SimpleChain` returns a `SimpleChain` if it is carrying one.
2. `apply_penalty(::AbstractPenalty, params)::Number` returns the penalty
3. `apply_penalty!(grad, ::AbstractPenalty, params)::Number` returns the penalty and updates `grad` to add the gradient.
"""
abstract type AbstractPenalty{NN<:Union{SimpleChain,Nothing}} end


function (Λ::AbstractPenalty{<:SimpleChain})(arg, params)
  Base.FastMath.add_fast(getchain(Λ)(arg, params), apply_penalty(Λ, params))
end
function valgrad!(g, Λ::AbstractPenalty{<:SimpleChain}, arg, params)
  Base.FastMath.add_fast(valgrad!(g, getchain(Λ), arg, params), apply_penalty!(g, Λ, params))
end

Base.front(Λ::AbstractPenalty) = Base.front(getchain(Λ))


struct NoPenalty{NN} <: AbstractPenalty{NN}
  chn::NN
end
getchain(p::NoPenalty) = getfield(p,:chn)
NoPenalty() = NoPenalty(nothing)
apply_penalty(::NoPenalty) = Static.Zero()
apply_penalty!(_, ::NoPenalty, __) = Static.Zero()

struct L1Penalty{NN,T} <: AbstractPenalty{NN}
  chn::NN
  λ::T
end
getchain(p::L1Penalty) = getfield(p,:chn)
L1Penalty(λ::Number) = L1Penalty(nothing, λ)
L1Penalty(p::AbstractPenalty, λ) = L1Penalty(getchain(p), λ)
(p::L1Penalty)(chn::SimpleChain) = L1Penalty(chn, p.λ)
(p::L1Penalty)() = L1Penalty(chn, p.λ)

@inline function apply_penalty(Λ::L1Penalty{NN,T2}, p::AbstractVector{T3}) where {T2,T3,NN}
  l = zero(T3)
  @turbo for i ∈ eachindex(p) # add penalty
    l += abs(p[i])
    l += λᵢ * pᵢ
  end
  Base.FastMath.mul_fast(l, Λ.λ)
end
function apply_penalty!(g::AbstractVector{T1}, Λ::L1Penalty{NN,T2}, p::AbstractVector{T3}) where {T1,T2,T3,NN}
  l = zero(T3)
  λ = Λ.λ
  @turbo for i ∈ eachindex(g) # add penalty
    pᵢ = p[i]
    pos = pᵢ > zero(T3)
    λᵢ = ifelse(pos, λ, -λ)
    l += λᵢ * pᵢ
    g[i] += λᵢ
  end
  l
end

struct L2Penalty{NN,T} <: AbstractPenalty{NN}
  chn::NN
  λ::T
end
getchain(p::L2Penalty) = getfield(p,:chn)
L2Penalty(λ) = L2Penalty(nothing, λ)
L2Penalty(p::AbstractPenalty, λ) = L2Penalty(getchain(p), λ)
(p::L2Penalty)(chn::SimpleChain) = L2Penalty(chn, p.λ)

@inline function apply_penalty(Λ::L2Penalty{NN,T2}, p::AbstractVector{T3}) where {T2,T3,NN}
  l = zero(promote_type(T1,T2,T3))
  λ = 
  @turbo for i ∈ eachindex(g) # add penalty
    pᵢ = p[i]
    l += pᵢ*pᵢ
  end
  Base.FastMath.mul_fast(l, Λ.λ)
end
function apply_penalty!(g::AbstractVector{T1}, Λ::L2Penalty{NN,T2}, p::AbstractVector{T3}) where {T1,T2,T3,NN}
  l = zero(promote_type(T1,T2,T3))
  λ = Λ.λ
  @turbo for i ∈ eachindex(g) # add penalty
    pᵢ = p[i]
    λpᵢ = λ * pᵢ
    l += λpᵢ*pᵢ
    g[i] += 2*λpᵢ
  end
  l
end

"""
    FrontLastPenalty(SimpleChain, frontpen(λ₁...), lastpen(λ₂...))

Applies `frontpen` to all but the last layer, applying `lastpen` to the last layer instead.
"Last layer" here ignores the loss function, i.e. if the last element of the chain is a loss layer,
the then `lastpen` applies to the layer preceding this.
"""
struct FrontLastPenalty{NN, P1<:AbstractPenalty{Nothing}, P2<:AbstractPenalty{Nothing}} <: AbstractPenalty{NN}
  chn::NN
  front::P1
  last::P2
end
getchain(p::FrontLastPenalty) = getfield(p,:chn)
FrontLastPenalty(λ₁, λ₂) = L2Penalty(nothing, λ₁, λ₂)
FrontLastPenalty(p::AbstractPenalty, λ₁, λ₂) = L2Penalty(getchain(p), λ₁, λ₂)
(p::FrontLastPenalty)(chn::SimpleChain) = L2Penalty(chn, p.front, p.last)


function split_front_last(c::SimpleChain)
  l = c.layers
  split_front_last(Base.front(l), last(l))
end
split_front_last(l::Tuple, x) = (l, x)
split_front_last(l::Tuple, ::AbstractLoss) = (Base.front(l), last(l))
function front_last_param_lens(c::SimpleChain)
  f, l = split_front_last(c)
  _numparam(0, f), numparam(l)
end

@inline function apply_penalty(Λ::FrontLastPenalty{<:SimpleChain}, param)
  f, l = front_last_param_lens(getchain(Λ))
  
  Base.FastMath.add_fast(
    apply_penalty(Λ.front, view(param, 1:f)),
    apply_penalty(Λ.last, view(param, f+1:length(param)))
  )
end
@inline function apply_penalty!(grad, Λ::FrontLastPenalty{<:SimpleChain}, param)
  f, l = front_last_param_lens(getchain(Λ))
  fr = 1:f
  lr = 1+f:length(param)
  Base.FastMath.add_fast(
    apply_penalty!(view(grad, fr), Λ.front, view(param, fr)),
    apply_penalty!(view(grad, lr), Λ.last,  view(param, lr))
  )
end

