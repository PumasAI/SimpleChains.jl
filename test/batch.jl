using SimpleChains, Test
@testset "batch" begin
  x = randn(3, 100)
  chain = SimpleChain(
    static(3),
    TurboDense{true}(tanh, 8),
    TurboDense{true}(identity, 4)
  )
  opt = SimpleChains.ADAM()
  penalty = L2Penalty(0.1)
  loss = SquaredLoss

  p0 = SimpleChains.init_params(chain)
  y = chain(x, p0)
  model = penalty(SimpleChains.add_loss(chain, loss(y)))
  p1 = SimpleChains.init_params(chain)
  origloss = model(x, p1)
  @time SimpleChains.train_batched!(copy(p1), p1, model, x, opt, 10_000)
  @test model(x, p1) < origloss
end
