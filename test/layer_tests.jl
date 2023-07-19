using SimpleChains, Zygote, Test

x = rand(5)
y = rand(2)

sc = SimpleChain(
  5,
  TurboDense{true}(tanh, 5),
  TurboDense{false}(tanh, 5),
  TurboDense{true}(SimpleChains.relu, 5),
  SimpleChains.Dropout(0.3),
  TurboDense{false}(SimpleChains.relu, 5),
  TurboDense{true}(identity, 2),
  TurboDense{false}(identity, 2),
  SquaredLoss(y)
)
sc_dynamic = SimpleChain(
  (5,),
  TurboDense{true}(tanh, 5),
  TurboDense{false}(tanh, 5),
  TurboDense{true}(SimpleChains.relu, 5),
  SimpleChains.Dropout(0.3),
  TurboDense{false}(SimpleChains.relu, 5),
  TurboDense{true}(identity, 2),
  TurboDense{false}(identity, 2),
  SquaredLoss(y)
)

p = SimpleChains.init_params(sc)

@test SimpleChains.remove_loss(sc)(x, p) isa AbstractVector
using SimpleChains.StaticArrays
@test @inferred(SimpleChains.remove_loss(sc)(SVector{5}(x), p)) isa
      SVector{2,Float64}
@test @inferred(SimpleChains.valgrad(sc, SVector{5}(x), p)) isa
      Tuple{Float64,SVector{126,Float32}}

g = similar(p);
g2 = similar(g);
g3 = similar(g);
g4 = similar(g);
xm = reshape(x, length(x), 1);
yml = SquaredLoss(reshape(y, length(y), 1));

for seed = 1:4
  SimpleChains.VectorizedRNG.seed!(seed)
  valgrad!(g, sc, x, p)

  SimpleChains.VectorizedRNG.seed!(seed)
  valgrad!(g2, sc, xm, p)
  SimpleChains.VectorizedRNG.seed!(seed)
  valgrad!(g3, SimpleChains.add_loss(sc, yml), xm, p)
  SimpleChains.VectorizedRNG.seed!(seed)
  valgrad!(g4, SimpleChains.add_loss(sc, yml), x, p)

  SimpleChains.VectorizedRNG.seed!(seed)
  gzyg = Zygote.gradient(p) do p
    sum(abs2, Base.front(sc)(x, p) .- y) / 2
  end |> only |> Vector
  gzyg2 = Zygote.gradient(p) do p
    sum(abs2, Base.front(sc_dynamic)(x, p) .- y) / 2
  end |> only |> Vector
  SimpleChains.VectorizedRNG.seed!(seed)
  gz = Zygote.gradient(sc, x, p)[2]
  @test gzyg ≈ gz

  @test size(gz) == size(p)
  @test size(g) == size(gz)

  @test !iszero(gz)
  @test !iszero(g)
  @test !iszero(g2)
  @test !iszero(g3)
  @test !iszero(g4)

  @test gz ≈ g rtol = 1e-6
  @test gz ≈ g2 rtol = 1e-6
  @test gz ≈ g3 rtol = 1e-6
  @test gz ≈ g4 rtol = 1e-6
end

xmat = rand(5, 20)
ymat = rand(2, 20)

sc2 = SimpleChains.add_loss(sc, SquaredLoss(ymat))

p2 = SimpleChains.init_params(sc2)

g2 = similar(p2)
SimpleChains.VectorizedRNG.seed!(2);
valgrad!(g2, sc2, xmat, p2)

SimpleChains.VectorizedRNG.seed!(2);
gz2 = Zygote.gradient(sc2, xmat, p2)[2]

@test size(gz2) == size(p2)
@test size(g2) == size(gz2)

@test !iszero(gz2)
@test !iszero(g2)

@test g2 ≈ gz2 rtol = 1e-6
