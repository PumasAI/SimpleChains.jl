using SimpleChains, Zygote, StaticArrays, Test

u0 = @SArray [2.0f0, 0.0f0]

sc = SimpleChain(
  static(2),
  Activation(x -> x^3),
  TurboDense{true}(tanh, static(50)),
  TurboDense{true}(identity, static(2))
)

p_nn = @inferred(SimpleChains.init_params(sc));
@test p_nn isa SimpleChains.StrideArraysCore.StaticStrideArray

out = @inferred(sc(u0, p_nn));
@test out isa SVector{2,Float32}

f = let sc = sc
  (u, p, t) -> sc(u, p)
end

t = 0.45f0
y = @SArray [1.6832f0, -0.174f0]

λ = @SArray [1.44533f0, 0.34325f0]

p_nn_sv = SVector{252}(p_nn);

for pv in (p_nn, p_nn_sv)
  _dy, back = Zygote.pullback(y, pv) do u, p
    f(u, p, t)
  end

  tmp1, tmp2 = @inferred(back(λ))

  @test tmp1 isa SVector{2,Float32}
  @test tmp2 isa SVector{252,Float32}
  @test _dy isa SVector{2,Float32}

  forw = f(y, pv, t)
  @test forw isa SVector{2,Float32}
end
