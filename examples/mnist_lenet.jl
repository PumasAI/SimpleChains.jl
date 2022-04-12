# Common

import MLDatasets
function get_data()
  xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
  xtest, ytest = MLDatasets.MNIST.testdata(Float32)

  (
    (reshape(xtrain, 28, 28, 1, :), UInt32.(ytrain .+ 1)),
    (reshape(xtest, 28, 28, 1, :), UInt32.(ytest .+ 1)),
  )
end
(xtrain, ytrain), (xtest, ytest) = get_data();



# SimpleChains
using SimpleChains
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



@time p = SimpleChains.init_params(lenet);

@time lenet(xtrain, p)

lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain));

g = similar(p);
@time valgrad!(g, lenetloss, xtrain, p)

G = similar(
  p,
  length(p),
  min(Threads.nthreads(), (Sys.CPU_THREADS ÷ ((Sys.ARCH === :x86_64) + 1))),
);

@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);

SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)

SimpleChains.init_params!(lenet, p);
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)



lenet.memory .= 0;
SimpleChains.init_params!(lenet, p);
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)
SimpleChains.init_params!(lenet, p);
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)



g0 = similar(g);
g1 = similar(g);
lenetloss.memory .= 0xff;
@time valgrad!(g0, lenetloss, xtrain, p)
lenetloss.memory .= 0x00;
@time valgrad!(g1, lenetloss, xtrain, p)
g0 == g1
lenet.memory .= 0;
SimpleChains.init_params!(lenet, p);
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)
SimpleChains.init_params!(lenet, p);
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)




## Classification of MNIST dataset
## with the convolutional neural network known as LeNet5.
## This script also combines various
## packages from the Julia ecosystem with Flux.
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Statistics, Random
# using Logging: with_logger
# using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
# using ProgressMeter: @showprogress
# import BSON
using CUDA
# arguments for the `train` function
Base.@kwdef struct Args
  η::Float64 = 3e-4             # learning rate
  λ::Float64 = 0                # L2 regularizer param, implemented as weight decay
  batchsize::Int = 128      # batch size
  epochs::Int = 10          # number of epochs
  seed::Int = 0             # set seed > 0 for reproducibility
end
args = Args();
const use_cuda = true && CUDA.functional()
const device = if use_cuda
  @info "Training on GPU"
  gpu
else
  @info "Training on CPU"
  cpu
end

# LeNet5 "constructor".
# The model can be adapted to any image size
# and any number of output classes.
function LeNet5(; imgsize = (28, 28, 1), nclasses = 10)
  out_conv_size = (imgsize[1] ÷ 4 - 3, imgsize[2] ÷ 4 - 3, 16)

  return Chain(
    Flux.Conv((5, 5), imgsize[end] => 6, Flux.relu),
    Flux.MaxPool((2, 2)),
    Flux.Conv((5, 5), 6 => 16, Flux.relu),
    Flux.MaxPool((2, 2)),
    Flux.flatten,
    Flux.Dense(prod(out_conv_size), 120, Flux.relu),
    Flux.Dense(120, 84, Flux.relu),
    Flux.Dense(84, nclasses),
  ) |> device
end

function loaders(xtrain, ytrain, xtest, ytest, args)
  ytrain, ytest = onehotbatch(ytrain, 1:10), onehotbatch(ytest, 1:10)

  train_loader = DataLoader((xtrain, ytrain), batchsize = args.batchsize, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = args.batchsize)

  return train_loader, test_loader
end
function loaders(args)
  (xtrain, ytrain), (xtest, ytest) = get_data()
  loaders(xtrain, ytrain, xtest, ytest, args)
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)

function eval_loss_accuracy(loader, model, device)
  l = 0.0f0
  acc = 0
  ntot = 0
  for (x, y) in loader
    x, y = x |> device, y |> device
    ŷ = model(x)
    l += loss(ŷ, y) * size(x)[end]
    acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
    ntot += size(x)[end]
  end
  return (loss = l / ntot |> round4, acc = acc / ntot * 100 |> round4)
end

## utility functions
num_params(model) = sum(length, Flux.params(model))
round4(x) = round(x, digits = 4)



function train(; kws...)
  args = Args(; kws...)
  args.seed > 0 && Random.seed!(args.seed)

  ## DATA
  train_loader, test_loader = get_data(args)
  @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

  ## MODEL AND OPTIMIZER
  model = LeNet5() |> device
  @info "LeNet5 model: $(num_params(model)) trainable params"


  opt = ADAM(args.η)
  train!(model, args, opt)

end
function train!(model, train_loader, args = Args(), opt = ADAM(args.η))
  ps = Flux.params(model)
  if args.λ > 0 # add weight decay, equivalent to L2 regularization
    opt = Optimiser(WeightDecay(args.λ), opt)
  end
  for _ = 1:args.epochs
    for (x, y) in train_loader
      x = device(x)
      y = device(y)
      gs = Flux.gradient(ps) do
        ŷ = model(x)
        loss(ŷ, y)
      end
      Flux.Optimise.update!(opt, ps, gs)
    end
  end
end




# Flux # @time model(device(xtrain))

model = LeNet5();
batchsize = use_cuda ? 2048 : 96Threads.nthreads();
train_loader, test_loader = loaders(xtrain, ytrain, xtest, ytest, Args(; batchsize));

@time train!(model, train_loader)
eval_loss_accuracy(train_loader, model, device),
eval_loss_accuracy(test_loader, model, device)

@time train!(model, train_loader)
eval_loss_accuracy(train_loader, model, device),
eval_loss_accuracy(test_loader, model, device)
