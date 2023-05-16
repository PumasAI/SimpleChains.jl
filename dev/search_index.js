var documenterSearchIndex = {"docs":
[{"location":"examples/mnist/#MNIST-Convolutions","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"","category":"section"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"First, we load the data using MLDatasets.jl:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"using MLDatasets\nxtrain3, ytrain0 = MLDatasets.MNIST.traindata(Float32);\nxtest3, ytest0 = MLDatasets.MNIST.testdata(Float32);\nsize(xtest3)\n# (28, 28, 60000)\nextrema(ytrain0) # digits, 0,...,9\n# (0, 9)","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"The covariate data (x) were named 3 as these are three-dimensional arrays, containing the height x width x number of images. The training data are vectors indicating the digit.","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"xtrain4 = reshape(xtrain3, 28, 28, 1, :);\nxtest4 = reshape(xtest3, 28, 28, 1, :);\nytrain1 = UInt32.(ytrain0 .+ 1);\nytest1 = UInt32.(ytest0 .+ 1);","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"SimpleChains' convolutional layers expect that we have a channels-in dimension, so we shape the images to be four dimensional It also currently defaults to 1-based indexing for its categories, so we shift all categories by 1.","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"We now define our model, LeNet5:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"using SimpleChains\n\nlenet = SimpleChain(\n  (static(28), static(28), static(1)),\n  SimpleChains.Conv(SimpleChains.relu, (5, 5), 6),\n  SimpleChains.MaxPool(2, 2),\n  SimpleChains.Conv(SimpleChains.relu, (5, 5), 16),\n  SimpleChains.MaxPool(2, 2),\n  Flatten(3),\n  TurboDense(SimpleChains.relu, 120),\n  TurboDense(SimpleChains.relu, 84),\n  TurboDense(identity, 10),\n)\n\nlenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain1));","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"We define the inputs as being statically sized (28,28,1) images. Specifying the input sizes allows these to be checked. Making them static, which we can do either in our simple chain, or by adding static sizing to the images themselves using a package like StrideArrays.jl or HybridArrays.jl. These packages are recommended for allowing you to mix dynamic and static sizes; the batch size should probably be left dynamic, as you're unlikely to want to specialize code generation on this, given that it is likely to vary, increasing compile times while being unlikely to improve runtimes.","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"In SimpleChains, the parameters are not a part of the model, but live as a separate vector that you can pass around to optimizers of your choosing. If you specified the input size, you create a random initial parameter vector corresponding to the model:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"@time p = SimpleChains.init_params(lenet);","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"The convolutional layers are initialized with a Glorot (Xavier) uniform distribution, while the dense layers are initialized with a Glorot (Xaviar) normal distribution. Biases are initialized to zero. Because the number of parameters can be a function of the input size, these must be provided if you didn't specify input dimension. For example:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"@time p = SimpleChains.init_params(lenet, size(xtrain4));","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"To allow training to use multiple threads, you can create a gradient matrix, with a number of rows equal to the length of the parameter vector p, and one column per thread. For example:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"estimated_num_cores = (Sys.CPU_THREADS ÷ ((Sys.ARCH === :x86_64) + 1));\nG = SimpleChains.alloc_threaded_grad(lenetloss);","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"Here, we're estimating that the number of physical cores is half the number of threads on an x86_64 system, which is true for most – but not all!!! – of them. Otherwise, we're assuming it is equal to the number of threads. This is of course also likely to be wrong, e.g. recent Power CPUs may have 4 or even 8 threads per core. You may wish to change this, or use Hwloc.jl for an accurate number.","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"Now that this is all said and done, we can train for 10 epochs using the ADAM optimizer with a learning rate of 3e-4, and then assess the accuracy and loss of both the training and test data:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"@time SimpleChains.train_batched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 10);\nSimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)\nSimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"Training for an extra 10 epochs should be fast on most systems. Performance is currently known to be poor on the M1 (PRs welcome, otherwise we'll look into this eventually), but should be  good/great on systems with AVX2/AVX512:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"@time SimpleChains.train_batched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 10);\nSimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)\nSimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)","category":"page"},{"location":"examples/custom_loss_layer/#Adding-a-custom-loss-layer","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"","category":"section"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"Loss functions like the LogitCrossEntropyLoss are defined for users to be able to quickly prototype models on new problems. However, sometimes there is a need to write one's own customized loss function. This example will walk through this process.","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"To show which functions need to be implemented for your own custom loss, this example will walk through implementing a BinaryLogitCrossEntropyLoss, which acts on a model with only a single output, and binary targets.","category":"page"},{"location":"examples/custom_loss_layer/#Mathematical-background-for-a-Binary-Cross-Entropy-Loss","page":"Adding a custom loss layer","title":"Mathematical background for a Binary Cross Entropy Loss","text":"","category":"section"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"Consider the following model:","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"p_theta(X_i) = sigma (f(X_i))","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"where X_i is the input features, sigma is the sigmoid function, given by sigma(x)=(1+e^-x)^-1 and f_theta is some function mapping defined by your model, which is parameterized by parameters theta. The output of f_theta (X_i) is called the \"logit\". The loss function we want to calculate is the following:","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"L(theta X Y) = -sum_i left  Y_ilnp_theta (X_i) + (1-Y_i)ln(1-p_theta (X_i)) right ","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"where Y_i is the true binary label of the i^textth sample. In order to implement this custom loss, we have to know what the gradient of this loss function is, w.r.t the parameters:","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"    fracpartial partial  theta L(theta  X Y) = -sum_i  Y_ifracpartial partial  thetalnp_theta (X_i) + (1-Y_i)fracpartial partial  thetaln (1-p_theta (X_i))","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"To simplify this calculation, we can use the fact that 1-p_theta (x)=p_theta (-x), and fracpartial partial  thetaln(p_theta(X_i))=(1+e^f_theta(X_i))^-1 fracpartial partial  theta f_theta(X_i). We are left with:","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"fracpartial partial  theta L(theta X Y) = -sum_i left  (2Y_i - 1)left (1+e^(2Y_i-1)f_theta(X_i) right )^-1 right  fracpartial partial  theta f_theta(X_i)","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"We have managed to write the derivative of the loss function, in terms of the derivative of the model, independently for each sample. The important part of this equation is the multiplicand of the partial derivative term; this term is the partial gradient used for back-propagation. From this point, we can begin writing the code.","category":"page"},{"location":"examples/custom_loss_layer/#Implementing-a-custom-loss-type","page":"Adding a custom loss layer","title":"Implementing a custom loss type","text":"","category":"section"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"We start by importing SimpleChains.jl into the current namespace:","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"using SimpleChains","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"We can now define our own type, which is a subtype of SimpleChains.AbstractLoss:","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"struct BinaryLogitCrossEntropyLoss{T,Y<:AbstractVector{T}} <: SimpleChains.AbstractLoss{T}\n    targets::Y\nend","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"The function used to get the inner targets is called target and can be defined easily:","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"SimpleChains.target(loss::BinaryLogitCrossEntropyLoss) = loss.targets\n(loss::BinaryLogitCrossEntropyLoss)(x::AbstractArray) = BinaryLogitCrossEntropyLoss(x)","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"Next, we define how to calculate the loss, given some logits:","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"function calculate_loss(loss::BinaryLogitCrossEntropyLoss, logits)\n    y = loss.targets\n    total_loss = zero(eltype(logits))\n    for i in eachindex(y)\n        p_i = inv(1 + exp(-logits[i]))\n        y_i = y[i]\n        total_loss -= y_i * log(p_i) + (1 - y_i) * (1 - log(p_i))\n    end\n    total_loss\nend\nfunction (loss::BinaryLogitCrossEntropyLoss)(previous_layer_output::AbstractArray{T}, p::Ptr, pu) where {T}\n    total_loss = calculate_loss(loss, previous_layer_output)\n    total_loss, p, pu\nend","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"As the other loss functions do this, we should define some functions to say that we don't want any preallocated temporary arrays:","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"function SimpleChains.layer_output_size(::Val{T}, sl::BinaryLogitCrossEntropyLoss, s::Tuple) where {T}\n    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)\nend\nfunction SimpleChains.forward_layer_output_size(::Val{T}, sl::BinaryLogitCrossEntropyLoss, s) where {T}\n    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)\nend","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"Finally, we define how to back-propagate the gradient from this loss function:","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"function SimpleChains.chain_valgrad!(\n    __,\n    previous_layer_output::AbstractArray{T},\n    layers::Tuple{BinaryLogitCrossEntropyLoss},\n    _::Ptr,\n    pu::Ptr{UInt8},\n) where {T}\n    loss = getfield(layers, 1)\n    total_loss = calculate_loss(loss, previous_layer_output)\n    y = loss.targets\n\n    # Store the backpropagated gradient in the previous_layer_output array.\n    for i in eachindex(y)\n        sign_arg = 2 * y[i] - 1\n        # Get the value of the last logit\n        logit_i = previous_layer_output[i]\n        previous_layer_output[i] = -(sign_arg * inv(1 + exp(sign_arg * logit_i)))\n    end\n\n    return total_loss, previous_layer_output, pu\nend","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"That's all! The way we can now use this loss function, just like any other:","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"using SimpleChains\n\nmodel = SimpleChain(\n    static(2),\n    TurboDense(tanh, 32),\n    TurboDense(tanh, 16),\n    TurboDense(identity, 1)\n)\n\nbatch_size = 64\nX = rand(Float32, 2, batch_size)\nY = rand(Bool, batch_size)\n\nparameters = SimpleChains.init_params(model);\ngradients = SimpleChains.alloc_threaded_grad(model);\n\n# Add the loss like any other loss type\nmodel_loss = SimpleChains.add_loss(model, BinaryLogitCrossEntropyLoss(Y));\n\nSimpleChains.valgrad!(gradients, model_loss, X, parameters)","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"Or alternatively, if you want to just train the parameters in full:","category":"page"},{"location":"examples/custom_loss_layer/","page":"Adding a custom loss layer","title":"Adding a custom loss layer","text":"epochs = 100\nSimpleChains.train_unbatched!(gradients, parameters, model_loss, X, SimpleChains.ADAM(), epochs); ","category":"page"},{"location":"examples/smallmlp/#Small-Multi-Layer-Perceptron","page":"Small Multi-Layer Perceptron","title":"Small Multi-Layer Perceptron","text":"","category":"section"},{"location":"examples/smallmlp/","page":"Small Multi-Layer Perceptron","title":"Small Multi-Layer Perceptron","text":"Here, we'll fit a simple network:","category":"page"},{"location":"examples/smallmlp/","page":"Small Multi-Layer Perceptron","title":"Small Multi-Layer Perceptron","text":"using SimpleChains\n\nmlpd = SimpleChain(\n  static(4),\n  TurboDense(tanh, 32),\n  TurboDense(tanh, 16),\n  TurboDense(identity, 4)\n)","category":"page"},{"location":"examples/smallmlp/","page":"Small Multi-Layer Perceptron","title":"Small Multi-Layer Perceptron","text":"Our goal here will be to try and approximate the matrix exponential:","category":"page"},{"location":"examples/smallmlp/","page":"Small Multi-Layer Perceptron","title":"Small Multi-Layer Perceptron","text":"function f(x)\n  N = Base.isqrt(length(x))\n  A = reshape(view(x, 1:N*N), (N,N))\n  expA = exp(A)\n  vec(expA)\nend\n\nT = Float32;\nX = randn(T, 2*2, 10_000);\nY = reduce(hcat, map(f, eachcol(X)));\nXtest = randn(T, 2*2, 10_000);\nYtest = reduce(hcat, map(f, eachcol(Xtest)));","category":"page"},{"location":"examples/smallmlp/","page":"Small Multi-Layer Perceptron","title":"Small Multi-Layer Perceptron","text":"Now, to train our network:","category":"page"},{"location":"examples/smallmlp/","page":"Small Multi-Layer Perceptron","title":"Small Multi-Layer Perceptron","text":"@time p = SimpleChains.init_params(mlpd);\nG = SimpleChains.alloc_threaded_grad(mlpd);\n\nmlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(Y));\nmlpdtest = SimpleChains.add_loss(mlpd, SquaredLoss(Ytest));\n\n# define a function named report to calculate and report the value of loss function with train and test sets.\nreport = let mtrain = mlpdloss, X=X, Xtest=Xtest, mtest = mlpdtest\n  p -> begin\n    let train = mtrain(X, p), test = mtest(Xtest, p)\n      @info \"Loss:\" train test\n    end\n  end\nend\n\nreport(p)\nfor _ in 1:3\n  @time SimpleChains.train_unbatched!(\n    G, p, mlpdloss, X, SimpleChains.ADAM(), 10_000\n  );\n  report(p)\nend","category":"page"},{"location":"examples/smallmlp/","page":"Small Multi-Layer Perceptron","title":"Small Multi-Layer Perceptron","text":"I get","category":"page"},{"location":"examples/smallmlp/","page":"Small Multi-Layer Perceptron","title":"Small Multi-Layer Perceptron","text":"julia> for _ in 1:3\n         @time SimpleChains.train_unbatched!(\n           G, p, mlpdloss, X, SimpleChains.ADAM(), 10_000\n         );\n         report(p)\n       end\n  5.258996 seconds (7.83 M allocations: 539.553 MiB, 4.18% gc time, 69.59% compilation time)\n┌ Info: Loss:\n│   train = 1243.1248f0\n└   test = 483.38852f0\n  1.638860 seconds\n┌ Info: Loss:\n│   train = 96.98259f0\n└   test = 210.4579f0\n  1.654781 seconds\n┌ Info: Loss:\n│   train = 44.350838f0\n└   test = 164.85913f0\n\njulia> versioninfo()\nJulia Version 1.9.0-DEV.1189\nCommit 293031b4a5* (2022-08-26 20:24 UTC)\nPlatform Info:\n  OS: Linux (x86_64-redhat-linux)\n  CPU: 8 × 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz\n  WORD_SIZE: 64\n  LIBM: libopenlibm\n  LLVM: libLLVM-14.0.5 (ORCJIT, tigerlake)\n  Threads: 8 on 8 virtual cores\nEnvironment:\n  JULIA_NUM_THREADS = 8","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SimpleChains","category":"page"},{"location":"#SimpleChains","page":"Home","title":"SimpleChains","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SimpleChains.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SimpleChains]","category":"page"},{"location":"#SimpleChains.ADAM","page":"Home","title":"SimpleChains.ADAM","text":"ADAM(η = 0.001, β = (0.9, 0.999))\n\nADAM optimizer.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.AbsoluteLoss","page":"Home","title":"SimpleChains.AbsoluteLoss","text":"AbsoluteLoss\n\nCalculates mean absolute loss of the target.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.AbstractPenalty","page":"Home","title":"SimpleChains.AbstractPenalty","text":"AbstractPenalty\n\nThe AbstractPenalty interface requires supporting the following methods:\n\ngetchain(::AbstractPenalty)::SimpleChain returns a SimpleChain if it is carrying one.\napply_penalty(::AbstractPenalty, params)::Number returns the penalty\napply_penalty!(grad, ::AbstractPenalty, params)::Number returns the penalty and updates grad to add the gradient.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.Activation","page":"Home","title":"SimpleChains.Activation","text":"Activation(activation)\n\nApplies activation function elementwise.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.Conv","page":"Home","title":"SimpleChains.Conv","text":"Conv(activation, dims::Tuple{Vararg{Integer}}, outputdim::Integer)\n\nPerforms a convolution with dims and maps it to outputdim output channels, then adds a bias (one per outputdim) and applies activation elementwise.\n\nE.g., Conv(relu, (5, 5), 16) performs a 5 × 5 convolution, and maps the input channels to 16 output channels, before adding a bias and applying relu.\n\nRandomly initializing weights using the (Xavier) Glorot uniform distribution. The bias is zero-initialized.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.Dropout","page":"Home","title":"SimpleChains.Dropout","text":"Dropout(p) # 0 < p < 1\n\nDropout layer.\n\nWhen evaluated without gradients, it multiplies inputs by (1 - p). When evaluated with gradients, it randomly zeros p proportion of inputs.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.Flatten","page":"Home","title":"SimpleChains.Flatten","text":"Flatten{N}()\n\nFlattens the first N dimensions. E.g.,\n\njulia> Flatten{2}()(rand(2,3,4))\n6×4 Matrix{Float64}:\n 0.0609115  0.597285  0.279899  0.888223\n 0.0667422  0.315741  0.351003  0.805629\n 0.678297   0.350817  0.984215  0.399418\n 0.125801   0.566696  0.96873   0.57744\n 0.331961   0.350742  0.59598   0.741998\n 0.26345    0.144635  0.076433  0.330475\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.FrontLastPenalty","page":"Home","title":"SimpleChains.FrontLastPenalty","text":"FrontLastPenalty(SimpleChain, frontpen(λ₁...), lastpen(λ₂...))\n\nApplies frontpen to all but the last layer, applying lastpen to the last layer instead. \"Last layer\" here ignores the loss function, i.e. if the last element of the chain is a loss layer, the then lastpen applies to the layer preceding this.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.L1Penalty","page":"Home","title":"SimpleChains.L1Penalty","text":"L1Penalty(λ)\n\nApplies a L1 penalty of λ to parameters, i.e. penalizing by their absolute value.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.L2Penalty","page":"Home","title":"SimpleChains.L2Penalty","text":"L2Penalty(λ)\n\nApplies a L2 penalty of λ to parameters, i.e. penalizing by their squares.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.LogitCrossEntropyLoss","page":"Home","title":"SimpleChains.LogitCrossEntropyLoss","text":"LogitCrossEntropyLoss\n\nCalculates mean logit cross-entropy loss.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.MaxPool","page":"Home","title":"SimpleChains.MaxPool","text":"MaxPool(dims::Tuple{Vararg{Integer}}\n\nCalculates the maximum of pools of size dims.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.SimpleChain","page":"Home","title":"SimpleChains.SimpleChain","text":"SimpleChain([inputdim::Union{Integer,Tuple{Vararg{Integer}}, ] layers)\n\nConstruct a SimpleChain. Optional inputdim argument allows SimpleChains to check the size of inputs. Making these static will allow SimpleChains to infer size and loop bounds at compile time. Batch size generally should not be included in the inputdim. If inputdim is not specified, some methods, e.g. init_params, will require passing the size as an additional argument, because the number of parameters may be a function of the input size (e.g., for a TurboDense layer).\n\nThe layers argument holds various SimpleChains layers, e.g. TurboDense, Conv, Activation, Flatten, Dropout, or MaxPool. It may optionally terminate in an AbstractLoss layer.\n\nThese objects are callable, e.g.\n\nc = SimpleChain(...);\np = SimpleChains.init_params(c);\nc(X, p) # X are the independent variables, and `p` the parameter vector.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.SquaredLoss","page":"Home","title":"SimpleChains.SquaredLoss","text":"SquaredLoss(target)\n\nCalculates half of mean squared loss of the target.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.TurboDense","page":"Home","title":"SimpleChains.TurboDense","text":"TurboDense{B=true}(activation, outputdim::Integer)\n\nLinear (dense) layer.\n\nB specifies whether the layer includes a bias term.\nThe activation function is applied elementwise to the result.\noutputdim indicates how many dimensions the input is mapped to.\n\nRandomly initializing weights using the (Xavier) Glorot normal distribution. The bias is zero-initialized.\n\n\n\n\n\n","category":"type"},{"location":"#Base.front-Tuple{SimpleChain}","page":"Home","title":"Base.front","text":"Base.front(c::SimpleChain)\n\nUseful for popping off a loss layer.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleChains.add_loss-Tuple{SimpleChain, SimpleChains.AbstractLoss}","page":"Home","title":"SimpleChains.add_loss","text":"add_loss(chn, l::AbstractLoss)\n\nAdd the loss function l to the simple chain. The loss function should hold the target you're trying to fit.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleChains.alloc_threaded_grad-Union{Tuple{SimpleChain}, Tuple{T}, Tuple{SimpleChain, Union{Nothing, SimpleChains.InputDimUnknown, Tuple{Vararg{Union{Integer, Static.StaticInt}}}}}, Tuple{SimpleChain, Union{Nothing, SimpleChains.InputDimUnknown, Tuple{Vararg{Union{Integer, Static.StaticInt}}}}, Type{T}}} where T","page":"Home","title":"SimpleChains.alloc_threaded_grad","text":"alloc_threaded_grad(chn, id = nothing, ::Type{T} = Float32; numthreads = min(Threads.nthreads(), SimpleChains.num_cores())\n\nReturns a preallocated array for writing gradients, for use with train_batched and train_unbatched. If Julia was started with multiple threads, returns a matrix with one column per thread, so they may accumulate gradients in parallel.\n\nNote that the memory is aligned to avoid false sharing.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleChains.biases","page":"Home","title":"SimpleChains.biases","text":"biases(sc::SimpleChain, p::AbstractVector, inputdim = nothing)\n\nReturns a tuple of the biases of the SimpleChain sc, as a view of the parameter vector p.\n\n\n\n\n\n","category":"function"},{"location":"#SimpleChains.init_params!","page":"Home","title":"SimpleChains.init_params!","text":"SimpleChains.init_params!(chn, p, id = nothing)\n\nRandomly initializes parameter vector p with input dim id. Input dim does not need to be specified if these were provided to the chain object itself. See the documentation of the individual layers to see how they are initialized, but it is generally via (Xavier) Glorot uniform or normal distributions.\n\n\n\n\n\n","category":"function"},{"location":"#SimpleChains.init_params-Union{Tuple{T}, Tuple{SimpleChain, Type{T}}} where T","page":"Home","title":"SimpleChains.init_params","text":"SimpleChains.init_params(chn[, id = nothing][, ::Type{T} = Float32])\n\nCreates a parameter vector of element type T with size matching that by id (argument not required if provided to the chain object itself). See the documentation of the individual layers to see how they are initialized, but it is generally via (Xavier) Glorot uniform or normal distributions.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleChains.numparam-Tuple{TurboDense, Tuple}","page":"Home","title":"SimpleChains.numparam","text":"numparam(d::Layer, inputdim::Tuple)\n\nReturns a Tuple{Int,S}. The first element is the number of parameters required by the layer given an argument of size inputdim. The second argument is the size of the object returned by the layer, which can be fed into numparam of the following layer.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleChains.params","page":"Home","title":"SimpleChains.params","text":"params(sc::SimpleChain, p::AbstractVector, inputdim = nothing)\n\nReturns a tuple of the parameters of the SimpleChain sc, as a view of the parameter vector p.\n\n\n\n\n\n","category":"function"},{"location":"#SimpleChains.train_batched!-Tuple{Union{Nothing, AbstractVector, AbstractMatrix}, AbstractVector, Union{SimpleChains.AbstractPenalty{<:SimpleChain}, SimpleChain}, Any, SimpleChains.AbstractOptimizer, Any}","page":"Home","title":"SimpleChains.train_batched!","text":"train_batched!(g::AbstractVecOrMat, p, chn, X, opt, iters; batchsize = nothing)\n\nTrain while batching arguments.\n\nArguments:\n\ng pre-allocated gradient buffer. Can be allocated with similar(p) (if you want to run single threaded), or alloc_threaded_grad(chn, size(X)) (size(X) argument is only necessary if the input dimension was not specified when constructing the chain). If a matrix, the number of columns gives how many threads to use. Do not use more threads than batch size would allow.\np is the parameter vector. It is updated inplace. It should be pre-initialized, e.g. with init_params/init_params!. This is to allow calling train_unbatched! several times to train in increments.\nchn is the SimpleChain. It must include a loss (see SimpleChains.add_loss) containing the target information (dependent variables) you're trying to fit.\nX the training data input argument (independent variables).\nopt is the optimizer. Currently, only SimpleChains.ADAM is supported.\niters, how many iterations to train for.\nbatchsize keyword argument: the size of the batches to use. If batchsize = nothing, it'll try to do a half-decent job of picking the batch size for you. However, this is not well optimized at the moment.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleChains.train_unbatched!-Tuple{Any, AbstractVector, Union{SimpleChains.AbstractPenalty{<:SimpleChain}, SimpleChain}, Any, SimpleChains.AbstractOptimizer, Any}","page":"Home","title":"SimpleChains.train_unbatched!","text":"train_unbatched!([g::AbstractVecOrMat, ]p, chn, X, opt, iters)\n\nTrain without batching inputs.\n\nArguments:\n\ng pre-allocated gradient buffer. Can be allocated with similar(p) (if you want to run single threaded), or alloc_threaded_grad(chn, size(X)) (size(X) argument is only necessary if the input dimension was not specified when constructing the chain). If a matrix, the number of columns gives how many threads to use. Do not use more threads than batch size would allow. This argument is optional. If excluded, it will run multithreaded (assuming you started Julia with multiple threads).\np is the parameter vector. It is updated inplace. It should be pre-initialized, e.g. with init_params/init_params!. This is to allow calling train_unbatched! several times to train in increments.\nchn is the SimpleChain. It must include a loss (see SimpleChains.add_loss) containing the target information (dependent variables) you're trying to fit.\nX the training data input argument (independent variables).\nopt is the optimizer. Currently, only SimpleChains.ADAM is supported.\niters, how many iterations to train for.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleChains.valgrad!-Tuple{Ptr{UInt8}, Any, SimpleChain, Any, Any}","page":"Home","title":"SimpleChains.valgrad!","text":"Allowed destruction:\n\nvalgrad_layer!\n\nAccepts return of previous layer (B) and returns an ouput C. If an internal layer, allowed to destroy B (e.g. dropout layer).\n\npullback!\n\nAccepts adjoint of its return (C̄). It is allowed to destroy this. It is also allowed to destroy the previous layer's return B to produce B̄ (the C̄ it receives). Thus, the pullback is not allowed to depend on C, as it may have been destroyed in producing C̄.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleChains.weights","page":"Home","title":"SimpleChains.weights","text":"weights(sc::SimpleChain, p::AbstractVector, inputdim = nothing)\n\nReturns a tuple of the weights (parameters other than biases) of the SimpleChain sc, as a view of the parameter vector p.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"examples/smallmlp.md\", \"examples/mnist.md\", \"examples/custom_loss_layer.md\"]","category":"page"}]
}
