using SimpleChains, ForwardDiff, Test

struct Tag end;
tagtype(::Type{T}) where {T<:Number} = typeof(ForwardDiff.Tag(Tag(), T))

function dual4x3(rng::AbstractRNG, x::T) where {T}
  ForwardDiff.Dual{tagtype(T)}(
    x,
    randn(rng),
    randn(rng),
    randn(rng),
    randn(rng)
  )
end
function dual4x3(rng::AbstractRNG, x::T) where {T<:ForwardDiff.Dual}
  ForwardDiff.Dual{tagtype(T)}(
    x,
    dual4x3(rng, randn(rng)),
    dual4x3(rng, randn(rng)),
    dual4x3(rng, randn(rng))
  )
end

rng = StableRNG(42)

for bias in (true, false)
  local M = 16
  local K = 20
  local N = 17
  local A = rand(rng, M, K + bias)
  local B = rand(rng, K, N)
  local bm = rand(rng, K, 1)

  for fa1 in (last ∘ tuple, dual4x3),
    fa2 in (last ∘ tuple, dual4x3),
    fb1 in (last ∘ tuple, dual4x3),
    fb2 in (last ∘ tuple, dual4x3)

    let A = fa2.(rng, fa1.(rng, A)),
      B = fb2.(rng, fb1.(rng, B)),
      bm = fb2.(rng, fb1.(rng, bm)),
      b = vec(bm)

      T = Base.promote_eltype(A, B)
      T === Float64 && continue
      C = Matrix{T}(undef, M, N)
      cm = Matrix{T}(undef, M, 1)
      c = Vector{T}(undef, M)

      AB = if bias
        @view(A[:, begin:end-1]) * B .+ @view(A[:, end])
      else
        A * B
      end
      Ab = if bias
        @view(A[:, begin:end-1]) * b .+ @view(A[:, end])
      else
        A * b
      end

      SimpleChains.matmul!(C, A, B, static(bias))
      @test reinterpret(Float64, C) ≈ reinterpret(Float64, AB)
      SimpleChains.matmul!(c, A, b, static(bias))
      @test reinterpret(Float64, c) ≈ reinterpret(Float64, Ab)
      SimpleChains.matmul!(c, A, bm, static(bias))
      @test reinterpret(Float64, c) ≈ reinterpret(Float64, Ab)
      SimpleChains.matmul!(cm, A, b, static(bias))
      @test reinterpret(Float64, vec(cm)) ≈ reinterpret(Float64, Ab)
      SimpleChains.matmul!(cm, A, bm, static(bias))
      @test reinterpret(Float64, vec(cm)) ≈ reinterpret(Float64, Ab)
    end
  end
end
