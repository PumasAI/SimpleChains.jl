ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
using SimpleChains, MLDatasets, Test

lenet = SimpleChain(
  (static(28), static(28), static(1)),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 6),
  SimpleChains.MaxPool(2, 2),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 16),
  SimpleChains.MaxPool(2, 2),
  Flatten(3),
  TurboDense(SimpleChains.relu, 120),
  TurboDense(SimpleChains.relu, 84),
  TurboDense(identity, 10),
)
# 3d and 0-indexed
xtrain3, ytrain0 = MLDatasets.MNIST.traindata(Float32);
xtest3, ytest0 = MLDatasets.MNIST.testdata(Float32);
xtrain4 = reshape(xtrain3, 28, 28, 1, :);
xtest4 = reshape(xtest3, 28, 28, 1, :);
ytrain1 = UInt32.(ytrain0 .+ 1);
ytest1 = UInt32.(ytest0 .+ 1);
lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain1));

@test SimpleChains.outputdim(lenet, size(xtrain4)) == (10, length(ytrain1));
@test SimpleChains.outputdim(lenet, size(xtest4)) == (10, length(ytest1));

# initialize parameters
@time p = SimpleChains.init_params(lenet);

@testset "Cache Corrupting Results" begin
  g = similar(p)
  subset = 1:200
  x = xtrain4[:, :, :, subset]
  y = ytrain1[subset]
  letnetloss = SimpleChains.add_loss(lenet, SimpleChains.LogitCrossEntropyLoss(y))
  lenetloss.memory .= 0x00
  valgrad!(g, lenetloss, x, p)
  g2 = similar(g)
  lenetloss.memory .= 0xff
  valgrad!(g2, lenetloss, x, p)
  @test g == g2
end

# initialize a gradient buffer matrix; number of columns places an upper bound
# on the number of threads used.
# G = similar(p, length(p), min(Threads.nthreads(), (Sys.CPU_THREADS ÷ ((Sys.ARCH === :x86_64) + 1))));
G = SimpleChains.alloc_threaded_grad(lenetloss);
# train
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 10);
# assess training and test loss
a0, l0 = SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)
a1, l1 = SimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)
# train without additional memory allocations
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 10);
# assess training and test loss
a2, l2 = SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)
a3, l3 = SimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)
if Threads.nthreads() <= 4
  @test a0 > 0.96
  @test a2 > 0.98
  @test a1 > 0.96
  @test a3 > 0.98
else
  @test a0 > 0.94
  @test a2 > 0.96
  @test a1 > 0.94
  @test a3 > 0.96
end
