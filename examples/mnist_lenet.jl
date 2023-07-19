# Comparison of Flux and SimpleChains implementations of classification of
# MNIST dataset with the convolutional neural network known as LeNet5.
# Each implementation is run twice so the second runs will demonstrate
# the performance after compilation has been performed.

import MLDatasets

# Get MNIST data
function get_data(split)
  x, y = MLDatasets.MNIST(split)[:]
  (reshape(x, 28, 28, 1, :), UInt32.(y .+ 1))
end

xtrain, ytrain = get_data(:train)
img_size = Base.front(size(xtrain))
xtest, ytest = get_data(:test)

# Training parameters
num_image_classes = 10
learning_rate = 3e-4
num_epochs = 10

using Printf

function display_loss(accuracy, loss)
  @printf("    training accuracy %.2f, loss %.4f\n", 100 * accuracy, loss)
end

# SimpleChains implementation
begin
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
    TurboDense(identity, num_image_classes)
  )
  lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain))

  for i = 1:2
    println("SimpleChains run #$i")
    @time "  gradient buffer allocation" G =
      SimpleChains.alloc_threaded_grad(lenetloss)
    @time "  parameter initialization" p = SimpleChains.init_params(lenet)
    #@time "  forward pass" lenet(xtrain, p)

    #g = similar(p);
    #@time "  valgrad!" valgrad!(g, lenetloss, xtrain, p)

    opt = SimpleChains.ADAM(learning_rate)
    @time "  train $(num_epochs) epochs" SimpleChains.train_batched!(
      G,
      p,
      lenetloss,
      xtrain,
      opt,
      num_epochs
    )

    @time "  compute training accuracy and loss" train_acc, train_loss =
      SimpleChains.accuracy_and_loss(lenetloss, xtrain, ytrain, p)
    display_loss(train_acc, train_loss)

    @time "  compute test accuracy and loss" test_acc, test_loss =
      SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)
    display_loss(test_acc, test_loss)
  end
end

# Flux implementation
begin
  using Flux
  using Flux.Data: DataLoader
  using Flux.Optimise: Optimiser, WeightDecay
  using Flux: onehotbatch, onecold
  using Flux.Losses: logitcrossentropy
  using Statistics, Random
  using CUDA

  use_cuda = CUDA.functional()
  batchsize = 0
  device = if use_cuda
    @info "Flux training on GPU"
    batchsize = 2048
    gpu
  else
    @info "Flux training on CPU"
    batchsize = 96 * Threads.nthreads()
    cpu
  end

  function create_loader(x, y, batch_size, shuffle)
    y = onehotbatch(y, 1:num_image_classes)
    DataLoader(
      (device(x), device(y));
      batchsize = batch_size,
      shuffle = shuffle
    )
  end

  train_loader = create_loader(xtrain, ytrain, batchsize, true)
  test_loader = create_loader(xtest, ytest, batchsize, false)

  function LeNet5()
    out_conv_size = (img_size[1] ÷ 4 - 3, img_size[2] ÷ 4 - 3, 16)

    return Chain(
      Flux.Conv((5, 5), img_size[end] => 6, Flux.relu),
      Flux.MaxPool((2, 2)),
      Flux.Conv((5, 5), 6 => 16, Flux.relu),
      Flux.MaxPool((2, 2)),
      Flux.flatten,
      Flux.Dense(prod(out_conv_size), 120, Flux.relu),
      Flux.Dense(120, 84, Flux.relu),
      Flux.Dense(84, num_image_classes)
    ) |> device
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
    return (acc = acc / ntot, loss = l / ntot)
  end

  function train!(model, train_loader, opt)
    ps = Flux.params(model)
    for _ = 1:num_epochs
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

  for run = 1:2
    println("Flux run #$run")
    @time "  create model" model = LeNet5()
    opt = ADAM(learning_rate)
    @time "  train $num_epochs epochs" train!(model, train_loader, opt)
    @time "  compute training loss" train_acc, train_loss =
      eval_loss_accuracy(test_loader, model, device)
    display_loss(train_acc, train_loss)
    @time "  compute test loss" test_acc, test_loss =
      eval_loss_accuracy(train_loader, model, device)
    display_loss(test_acc, test_loss)
  end
end
