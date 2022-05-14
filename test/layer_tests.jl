using SimpleChains, Zygote, Test

x = rand(5)
y = rand(2)

sc = SimpleChain(
  static(5),
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

g = similar(p)
valgrad!(g, sc, x, p)

gz = Zygote.gradient(sc, x, p)[2]

@test size(gz) == size(p)
@test size(g) == size(gz)

@test !iszero(gz)
@test !iszero(g)

@test gz ≈ g rtol=1e-6

xmat = rand(5, 20)
ymat = rand(2, 20)

sc2 = SimpleChain(
    static(5),
    TurboDense{true}(tanh, 5),
    TurboDense{false}(tanh, 5),
    TurboDense{true}(SimpleChains.relu, 5),
    SimpleChains.Dropout(0.3),
    TurboDense{false}(SimpleChains.relu, 5),
    TurboDense{true}(identity, 2),
    TurboDense{false}(identity, 2),
    SquaredLoss(ymat)
)

p2 = SimpleChains.init_params(sc2)

g2 = similar(p2)
valgrad!(g2, sc2, xmat, p2)

gz2 = Zygote.gradient(sc2, xmat, p2)[2]

@test size(gz2) == size(p2)
@test size(g2) == size(gz2)

@test !iszero(gz2)
@test !iszero(g2)

@test g2 ≈ gz2 rtol=1e-6
