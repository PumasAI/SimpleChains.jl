using SimpleChains, Zygote, StaticArrays, ForwardDiff, Test

u0 = @SArray [2.0f0, 0.0f0]

sc = SimpleChain(
  static(2),
  Activation(x -> x^3),
  TurboDense{true}(tanh, static(50)),
  TurboDense{true}(identity, static(2))
)

p_nn = @inferred(SimpleChains.init_params(sc))
@test p_nn isa SimpleChains.StrideArraysCore.StaticStrideArray

out = @inferred(sc(u0, p_nn))
@test out isa SVector{2,Float32}

f = let sc = sc
  (u, p, t) -> sc(u, p)
end

t = 0.45f0
y = @SArray [1.6832f0, -0.174f0]

λ = @SArray [1.44533f0, 0.34325f0]

ref_p = ForwardDiff.gradient(q -> sum(λ .* sc(y, q)), Vector(p_nn))
ref_y = ForwardDiff.gradient(x -> sum(λ .* sc(SVector{2}(x), p_nn)), Vector(y))

_dy, back = Zygote.pullback(y, p_nn) do u, p
  f(u, p, t)
end

tmp1, tmp2 = @inferred(back(λ))

@test tmp1 isa SVector{2,Float32}
@test tmp2 isa SVector{252,Float32}
@test _dy isa SVector{2,Float32}
@test tmp1 ≈ ref_y rtol = 1e-4
@test tmp2 ≈ ref_p rtol = 1e-4

forw = f(y, p_nn, t)
@test forw isa SVector{2,Float32}

# `SVector` parameters are currently problematic,
# see https://github.com/PumasAI/SimpleChains.jl/issues/224
@testset "SVector parameters" begin
  p_nn_sv = SVector{252}(p_nn)

  # If the chain input is an SArray too, then evaluation is fine
  # since that case is specialized on
  forw_sv = f(y, p_nn_sv, t)
  @test forw_sv isa SVector{2,Float32}
  @test forw_sv ≈ forw

  # Otherwise a pointer of an SVector is used, which cannot be relied upon on all platforms
  # @test f(Array(y), p_nn_sv, t) ≈ forw

  # Gradient computation currently not supported in any case
  scl = SimpleChains.add_loss(sc, SquaredLoss(Vector(y)))
  g = similar(p_nn)
  for x in (y, Array(y))
    gx = similar(x)
    @test_throws "pointer-backed" SimpleChains.valgrad!(g, scl, x, p_nn_sv)
    @test_throws "pointer-backed" SimpleChains.valgrad!(
      (gx, g),
      scl,
      x,
      p_nn_sv
    )
    @test_throws "pointer-backed" SimpleChains.valgrad(scl, x, p_nn_sv)
    @test_throws "pointer-backed" SimpleChains.valgrad_noloss(sc, x, p_nn_sv)
    @test_throws "pointer-backed" Zygote.pullback(
      (u, p) -> f(u, p, t),
      x,
      p_nn_sv
    )
  end

  # The convenience function with both gradients set to nothing is the same as the forward call
  l_ref = scl(y, p_nn_sv)
  @test SimpleChains.valgrad!((nothing, nothing), scl, y, p_nn_sv) == l_ref
  # Here a pointer of an SVector is used, which cannot be relied upon on all platforms
  # @test SimpleChains.valgrad!((nothing, nothing), scl, Array(y), p_nn_sv) ≈ l_ref

  # `MVectors` are fine instead
  mv = MVector(p_nn_sv)
  g_mv = similar(mv)
  @test SimpleChains.valgrad!(g, scl, y, p_nn) ≈ l_ref
  @test g ≈ ForwardDiff.gradient(q -> scl(y, q), Vector(p_nn)) rtol = 1e-4
  for x in (y, Array(y))
    forw_mv = f(x, mv, t)
    @test forw_mv isa SVector{2,Float32}
    @test forw_mv ≈ forw
    l_mv = SimpleChains.valgrad!(g_mv, scl, x, mv)
    @test l_mv ≈ l_ref
    @test g_mv ≈ g
  end
end
