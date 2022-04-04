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
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
import MLDatasets
import BSON
using CUDA
# arguments for the `train` function
Base.@kwdef mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 128      # batch size
    epochs = 10          # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    use_cuda = true      # if true use cuda (if available)
    infotime = 1             # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true      # log training with tensorboard
    savepath = "runs/"    # results path
end
args = Args();
const use_cuda = args.use_cuda && CUDA.functional()
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
function LeNet5(; imgsize=(28,28,1), nclasses=10)
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)

  return Chain(
    Flux.Conv((5, 5), imgsize[end]=>6, Flux.relu),
    Flux.MaxPool((2, 2)),
    Flux.Conv((5, 5), 6=>16, Flux.relu),
    Flux.MaxPool((2, 2)),
    flatten,
    Flux.Dense(prod(out_conv_size), 120, Flux.relu),
    Flux.Dense(120, 84, Flux.relu),
    Flux.Dense(84, nclasses)
  ) |> device
end

function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest),  batchsize=args.batchsize)

    return train_loader, test_loader
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)

function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end

## utility functions
num_params(model) = sum(length, Flux.params(model))
round4(x) = round(x, digits=4)



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
function train!(model, args = Args(), opt = ADAM(args.η))
    ps = Flux.params(model)
    use_cuda = args.use_cuda && CUDA.functional()
    if args.λ > 0 # add weight decay, equivalent to L2 regularization
        opt = Optimiser(WeightDecay(args.λ), opt)
    end

    ## LOGGING UTILITIES
    if args.tblogger
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        # @info "TensorBoard logging at \"$(args.savepath)\""
    end

    # function report(epoch)
    #     train = eval_loss_accuracy(train_loader, model, device)
    #     test = eval_loss_accuracy(test_loader, model, device)
    #     println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    #     if args.tblogger
    #         set_step!(tblogger, epoch)
    #         with_logger(tblogger) do
    #             @info "train" loss = train.loss acc = train.acc
    #             @info "test" loss = test.loss acc = test.acc
    #         end
    #     end
    # end

    ## TRAINING
    # @info "Start Training"
    # report(0)
    for epoch in 1:args.epochs
      for (x, y) in train_loader
        x, y = x |> device, y |> device
        gs = Flux.gradient(ps) do
          ŷ = model(x)
          loss(ŷ, y)
        end

        Flux.Optimise.update!(opt, ps, gs)
      end

        ## Printing and logging
      # epoch % args.infotime == 0 && report(epoch)
      # if args.checktime > 0 && epoch % args.checktime == 0
      #       !ispath(args.savepath) && mkpath(args.savepath)
      #       modelpath = joinpath(args.savepath, "model.bson")
      #       let model = cpu(model) #return model to cpu before serialization
      #           BSON.@save modelpath model epoch
      #       end
      #       @info "Model saved in \"$(modelpath)\""
      #   end
    end
end


using SimpleChains
lenet = SimpleChain(
  (static(28),static(28),static(1)),
  SimpleChains.Conv(SimpleChains.relu, (5,5), 6),
  SimpleChains.MaxPool(2, 2),
  SimpleChains.Conv(SimpleChains.relu, (5,5), 16),
  SimpleChains.MaxPool(2, 2),
  Flatten(3),
  TurboDense(SimpleChains.relu, 120),
  TurboDense(SimpleChains.relu, 84),
  TurboDense(identity, 10)
)
function eval_loss_accuracy(X, Y, model, p)
  Yoc = onecold(Y) .% UInt32
  mloss = SimpleChains.add_loss(model, SimpleChains.LogitCrossEntropyLoss(Yoc))
  l = mloss(X, p) * size(X)[end]
  Ŷ = Base.front(mloss)(X, p)
  acc = sum( onecold(Matrix(Ŷ)) .== Yoc)
  ntot = size(X)[end]
  return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end


train_loader, test_loader = get_data(args);
model = LeNet5();

(x, y) = first(train_loader);

p = mapreduce(vec, vcat, Flux.params(model).order);

@time model(device(x))
@time lenet(x, p)

lenetloss = SimpleChains.add_loss(lenet, SimpleChains.LogitCrossEntropyLoss(y.indices));
g = similar(p);
@time valgrad!(g, lenetloss, x, p)

@time train!(model)
eval_loss_accuracy(train_loader, model, device), eval_loss_accuracy(test_loader, model, device)

X, Y = train_loader.data;
lenetfull = SimpleChains.add_loss(lenet, SimpleChains.LogitCrossEntropyLoss(Y.indices));

G = similar(p, length(p), Threads.nthreads() ÷ 2);
@time SimpleChains.train_batched!(G, p, lenetfull, X, SimpleChains.ADAM(3e-4), 10);

Xtest, Ytest = test_loader.data;
eval_loss_accuracy(X,Y,lenet,p), eval_loss_accuracy(Xtest, Ytest, lenet, p)

eval_loss_accuracy(X,Y,lenet,p)
eval_loss_accuracy(Xtest, Ytest, lenet, p)
