using SimpleChains
using Test, Aqua, ForwardDiff, Zygote

function countallocations!(g, sc, x, p)
  @allocated valgrad!(g, sc, x, p)
end
dual(x) = ForwardDiff.Dual(x, randn(), randn(), randn())
dual(x::ForwardDiff.Dual) = ForwardDiff.Dual(x, dual(randn()), dual(randn()))

@testset "SimpleChains.jl" begin
  @test isempty(Test.detect_unbound_args(SimpleChains))
  @test isempty(Test.detect_ambiguities(SimpleChains))
  @testset "Dropout ForwardDiff" begin
for T in (Float32, Float64)

  if T === Float32 # test construction using tuples
    scbase = SimpleChain(
      static(24),
      (
        Activation(abs2),
        TurboDense{true}(tanh, static(8)),
        TurboDense{true}(identity, static(2)),
      ),
    ) # test inputdim
    scdbase = SimpleChain((
      TurboDense{true,Int}(tanh, 8),
      Dropout(0.2),
      TurboDense{true,Int}(identity, 2),
    )) # test inputdim unknown
  else # test construction using `Vararg`
    scbase = SimpleChain(
      static(24),
      Activation(abs2),
      TurboDense{true}(tanh, static(8)),
      TurboDense{true}(identity, static(2)),
    ) # test inputdim
    scdbase = SimpleChain(
      TurboDense{true,Int}(tanh, 8),
      Dropout(0.2),
      TurboDense{true,Int}(identity, 2),
    ) # test inputdim unknown
  end

  x = rand(T, 24, 199)

  y = StrideArray{T}(undef, (static(2), size(x, 2))) .= randn.() .* 10
  sc = SimpleChains.add_loss(scbase, SquaredLoss(y))

  @test first(Dropout(0.5)(x, pointer(x), pointer(sc.memory))) === x
  @test sum(iszero, x) == 0
  x .= rand.()

  scflp = FrontLastPenalty(sc, L2Penalty(2.3), L1Penalty(0.45))
  print_str0 = """
  SimpleChain with the following layers:
  Activation layer applying: abs2
  TurboDense static(8) with bias.
  Activation layer applying: tanh
  TurboDense static(2) with bias.
  SquaredLoss"""
  print_str1 = """
  Penalty on all but last layer: L2Penalty (λ=2.3)
  Penalty on last layer: L1Penalty (λ=0.45) applied to:
  SimpleChain with the following layers:
  Activation layer applying: abs2
  TurboDense static(8) with bias.
  Activation layer applying: tanh
  TurboDense static(2) with bias.
  SquaredLoss"""

  @test sprint((io, t) -> show(io, t), sc) == print_str0
  if VERSION >= v"1.6"
    @test sprint((io, t) -> show(io, t), scflp) == print_str1
  else
    # typename doesn't work on 1.5
    @test_broken sprint((io, t) -> show(io, t), scflp) == print_str1
  end
  p = SimpleChains.init_params(scflp, T)
  g = similar(p)
  @test_throws ArgumentError sc(rand(T, 23, 2), p)
  @test_throws ArgumentError sc(rand(T, 23), p)
  @test_throws MethodError sc(Array{T,0}(undef), p)
  @test_throws ArgumentError valgrad!(g, sc, rand(T, 23, 2), p)
  @test_throws ArgumentError valgrad!(g, sc, rand(T, 23), p)
  valgrad!(g, scflp, x, p)
  if VERSION < v"1.9-DEV" # FIXME: remove check when Zygote stops segfaulting on 1.8-DEV 
    @test g == only(
      Zygote.gradient(p -> FrontLastPenalty(sc, L2Penalty(2.3), L1Penalty(0.45))(x, p), p),
    )
    _gzyg = Zygote.gradient(p) do p
      0.5 / size(x)[end] * sum(abs2, Base.front(sc)(x, p) .- y)
    end
    gzyg = copy(_gzyg[1])
    g2 = similar(g)
    valgrad!(g2, sc, x, p)
    @test g2 ≈ gzyg
  end

  gfd = ForwardDiff.gradient(p) do p
    off = 8 * 24
    A1 = reshape(view(p, 1:off), (8, 24))
    off_old = off
    off += 8
    b1 = view(p, 1+off_old:off)
    l1 = tanh.(A1 * abs2.(x) .+ b1)

    off_old = off
    off += 8 * 2
    A2 = reshape(view(p, 1+off_old:off), (2, 8))

    off_old = off
    off += 2
    b2 = view(p, 1+off_old:off)
    l2 = (A2 * l1 .+ b2)

    l = 0.5mapreduce(+, vec(l2), y.data) do xi, yi
      abs2(xi - yi)
    end
    l / size(x)[end] +
    2.3 * (sum(abs2, A1) + sum(abs2, b1)) +
    0.45 * (sum(abs, A2) + sum(abs, b2))
  end
  @test g ≈ gfd
  scd = SimpleChains.add_loss(scdbase, SquaredLoss(y))
  @test_throws ArgumentError SimpleChains.init_params(scd, T)
  @test length(SimpleChains.init_params(scd, size(x), T)) == length(p)
  @test sprint((io, t) -> show(io, t), scd) == """
SimpleChain with the following layers:
TurboDense 8 with bias.
Activation layer applying: tanh
Dropout(p=0.2)
TurboDense 2 with bias.
SquaredLoss"""

  valgrad!(g, scd, x, p)
  offset = 2SimpleChains.align(first(scd.layers).outputdim * size(x, 2) * sizeof(T))
  si = SimpleChains.StrideIndex{1,(1,),1}(
    (SimpleChains.StaticInt(1),),
    (SimpleChains.StaticInt(1),),
  )
  m = SimpleChains.StrideArray(
    SimpleChains.PtrArray(
      SimpleChains.stridedpointer(
        reinterpret(Ptr{SimpleChains.Bit}, pointer(scd.memory) + offset),
        si,
      ),
      (size(x, 2) * 8,),
      Val((true,)),
    ),
    scd.memory,
  )

  valgrad!(g, scd, x, p)
  offset = 2SimpleChains.align(first(scd.layers).outputdim * size(x, 2) * sizeof(T))
  si = SimpleChains.StrideIndex{1,(1,),1}(
    (SimpleChains.StaticInt(1),),
    (SimpleChains.StaticInt(1),),
  )
  m = SimpleChains.StrideArray(
    SimpleChains.PtrArray(
      SimpleChains.stridedpointer(
        reinterpret(Ptr{SimpleChains.Bit}, pointer(scd.memory) + offset),
        si,
      ),
      (size(x, 2) * 8,),
      Val((true,)),
    ),
    scd.memory,
  )

  let
    off = 8 * 24
    A1 = reshape(view(p, 1:off), (8, 24))
    off_old = off
    off += 8
    b1 = view(p, 1+off_old:off)
    l1 = tanh.(A1 * x .+ b1)
    vec(l1) .*= m

    off_old = off
    off += 8 * 2
    A2 = reshape(view(p, 1+off_old:off), (2, 8))

    off_old = off
    off += 2
    b2 = view(p, 1+off_old:off)
    (W1,x1), (W2,x2) = SimpleChains.params(L2Penalty(sc, 2.3), p)
    @test W1 == A1
    @test x1 == b1
    @test W2 == A2
    @test x2 == b2
  end

  gfdd = ForwardDiff.gradient(p) do p
    off = 8 * 24
    A1 = reshape(view(p, 1:off), (8, 24))
    off_old = off
    off += 8
    b1 = view(p, 1+off_old:off)
    l1 = tanh.(A1 * x .+ b1)
    vec(l1) .*= m

    off_old = off
    off += 8 * 2
    A2 = reshape(view(p, 1+off_old:off), (2, 8))

    off_old = off
    off += 2
    b2 = view(p, 1+off_old:off)
    l2 = (A2 * l1 .+ b2)

    (0.5 / size(x)[end]) * mapreduce(+, vec(l2), y.data) do xi, yi
      abs2(xi - yi)
    end
  end
  if T === Float64
    @test g ≈ gfdd
  else
    @show g ≈ gfdd
    @show isapprox(g, gfdd, rtol = 1e-1)
    @show isapprox(g, gfdd, rtol = 1e-2)
    @show isapprox(g, gfdd, rtol = 1e-3)
    @show isapprox(g, gfdd, rtol = 1e-4)
    @show isapprox(g, gfdd, rtol = 1e-5)
    @show isapprox(g, gfdd, rtol = 1e-6)
    @show isapprox(g, gfdd, rtol = 1e-7)
    @show isapprox(g, gfdd, rtol = 1e-8)
  end
  # let g=g, sc=sc, x=x, p=p
  @test iszero(
    countallocations!(g, FrontLastPenalty(sc, L2Penalty(2.3), NoPenalty()), x, p),
  )
  @test iszero(
    countallocations!(g, FrontLastPenalty(scd, L2Penalty(2.3), L1Penalty(0.45)), x, p),
  )
  # @test iszero(@allocated(valgrad!(g, sc, x, p)))

  td = TurboDense{true}(tanh, static(8))
  pd = dual.(p)
  xd = dual.(x)

  pdd = dual.(pd)
  xdd = dual.(xd)

  pu = Vector{UInt8}(
    undef,
    first(SimpleChains.layer_output_size(Val(eltype(xdd)), td, size(x))),
  )

  A = reshape(view(p, 1:8*24), (8, 24))
  b = view(p, 1+8*24:8*25)
  Ad = reshape(view(pd, 1:8*24), (8, 24))
  bd = view(pd, 1+8*24:8*25)
  ld = tanh.(Ad * x .+ bd)
  l_d = tanh.(A * xd .+ b)
  ld_d = tanh.(Ad * xd .+ bd)

  Add = reshape(view(pdd, 1:8*24), (8, 24))
  bdd = view(pdd, 1+8*24:8*25)
  ldd = tanh.(Add * x .+ bdd)
  ldd_dd = tanh.(Add * xdd .+ bdd)
  if T === Float64
    GC.@preserve pd pu begin
      @test reinterpret(T, ld) ≈ reinterpret(T, td(x, pointer(pd), pointer(pu))[1])
      @test reinterpret(T, l_d) ≈ reinterpret(T, td(xd, pointer(p), pointer(pu))[1])
      @test reinterpret(T, ld_d) ≈ reinterpret(T, td(xd, pointer(pd), pointer(pu))[1])

      @test reinterpret(T, ldd) ≈ reinterpret(T, td(x, pointer(pdd), pointer(pu))[1])
      @test reinterpret(T, ldd_dd) ≈ reinterpret(T, td(xdd, pointer(pdd), pointer(pu))[1])
    end
  else
    GC.@preserve pd pu begin
      @test_broken reinterpret(T, ld) ≈ reinterpret(T, td(x, pointer(pd), pointer(pu))[1])
      @test_broken reinterpret(T, l_d) ≈ reinterpret(T, td(xd, pointer(p), pointer(pu))[1])
      @test_broken reinterpret(T, ld_d) ≈
                   reinterpret(T, td(xd, pointer(pd), pointer(pu))[1])

      @test_broken reinterpret(T, ldd) ≈ reinterpret(T, td(x, pointer(pdd), pointer(pu))[1])
      @test_broken reinterpret(T, ldd_dd) ≈
                   reinterpret(T, td(xdd, pointer(pdd), pointer(pu))[1])
    end
  end
  @testset "training" begin
    p .= randn.() .* 100
    # small penalties since we don't care about overfitting here
    vg1 = valgrad!(g, FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)), x, p)
    SimpleChains.train!(
      g,
      p,
      FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)),
      x,
      SimpleChains.ADAM(),
      1000,
    )
    @test valgrad!(g, FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)), x, p) < vg1
    p .= randn.() .* 100
    vg2 = FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))(x, p)
    SimpleChains.train!(
      g,
      p,
      FrontLastPenalty(scd, L2Penalty(2.3), L1Penalty(0.45)),
      x,
      SimpleChains.ADAM(),
      1000,
    )
    @test FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))(x, p) < vg2
  end
  @testset "vector of targets" begin
    p .= randn.() .* 100
    ys = [@. y + 0.1randn() for _ = 1:1000]
    # small penalties since we don't care about overfitting here
    vg1 = valgrad!(g, FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)), x, p)
    SimpleChains.train_unbatched!(
      g,
      p,
      FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)),
      x,
      SimpleChains.ADAM(),
      ys,
    )
    @test valgrad!(g, FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)), x, p) < vg1
    p .= randn.() .* 100
    vg2 = FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))(x, p)
    SimpleChains.train_unbatched!(
      g,
      p,
      FrontLastPenalty(scd, L2Penalty(2.3), L1Penalty(0.45)),
      x,
      SimpleChains.ADAM(),
      ys,
    )
    @test FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))(x, p) < vg2

  end
end
  end
  @testset "Convolution" begin
    function clt(f, x, K, b)
      d1 = size(x, 1) - size(K, 1) + 1
      d2 = size(x, 2) - size(K, 2) + 1
      closuresshouldntbeabletomutatebindings =
        Array{Base.promote_eltype(x, K, b)}(undef, d1, d2, size(K, 4), size(x, 4))
      SimpleChains.convlayer!(f, closuresshouldntbeabletomutatebindings, x, K, b)
      return closuresshouldntbeabletomutatebindings
    end

    function convlayertest(x, y, K, b)
      closuresshouldntbeabletomutatebindings = clt(relu, x, K, b)
      δ = vec(closuresshouldntbeabletomutatebindings) .- vec(y)
      0.5/size(x)[end] * (δ'δ)
    end
    function convlayertest(x, y, K0, b0, K1, b1)
      csbamb = clt(identity, clt(relu, x, K0, b0), K1, b1)
      closuresshouldntbeabletomutatebindings = tanh.(csbamb)
      δ = vec(closuresshouldntbeabletomutatebindings) .- vec(y)
      0.5/size(x)[end] * (δ'δ)
    end

    scconv = SimpleChain(
      Conv(relu, (3, 3), 4),
      #  Conv(tanh, (5,5), 4, 3)
    )
    batch = 2
    x = rand(Float32, 28, 28, 1, batch)
    y = rand(Float32, 26, 26, 4, batch)
    scconvl = SimpleChains.add_loss(scconv, SquaredLoss(y))
    p = @inferred(SimpleChains.init_params(scconvl, size(x)))
    (K, b), p2 =
      @inferred(SimpleChains.getparams(first(scconv.layers), pointer(p), size(x)))
    refloss = convlayertest(x, y, K, b)
    @test scconvl(x, p) ≈ refloss
    g = similar(p)
    @test valgrad!(g, scconvl, x, p) ≈ refloss

    # gK = ForwardDiff.gradient(k -> convlayertest(x,y,k, b), K);
    # @test Float32.(vec(gK)) ≈ g
    gpfd = ForwardDiff.gradient(p) do p
      K = copy(reshape(@view(p[1:36]), (3, 3, 1, 4)))
      b = p[37:end]
      convlayertest(x, y, K, b)
    end
    @test gpfd ≈ g



    scconv2 = SimpleChain(
      Conv(relu, (3, 3), 4), # fast_fuse == true
      Conv(tanh, (5, 5), 3),  # fast_fuse == false
    )
    z = rand(Float32, 22, 22, 3, batch)
    scconv2l = @inferred(SimpleChains.add_loss(scconv2, SquaredLoss(z)))
    p = @inferred(SimpleChains.init_params(scconv2l, size(x)))
    (K0, b0), p2 =
      @inferred(SimpleChains.getparams((scconv2.layers)[1], pointer(p), size(x)))
    (K1, b1), p3 = @inferred(SimpleChains.getparams((scconv2.layers)[2], p2, (1, 1, 4, 1)))
    refloss = convlayertest(x, z, K0, b0, K1, b1)
    @test @inferred(scconv2l(x, p)) ≈ refloss
    g = similar(p)
    @test @inferred(valgrad!(g, scconv2l, x, p)) ≈ refloss

    gK0 = ForwardDiff.gradient(k -> convlayertest(x, z, k, b0, K1, b1), K0)
    gb0 = ForwardDiff.gradient(b -> convlayertest(x, z, K0, b, K1, b1), b0)
    gK1 = ForwardDiff.gradient(k -> convlayertest(x, z, K0, b0, k, b1), K1)
    gb1 = ForwardDiff.gradient(b -> convlayertest(x, z, K0, b0, K1, b), b1)
    # @show typeof(gK0) typeof(gK1)
    @test Float32.(vcat(vec(gK0), gb0, vec(gK1), gb1)) ≈ g

  end
  @testset "MaxPool" begin
    using Test, ForwardDiff
    A = rand(8, 8, 2)
    firstmax = max(A[1, 1, 1], A[1, 2, 1], A[2, 1, 1], A[2, 2, 1])
    # test duplicates
    A[1, 1, 1] = firstmax
    A[2, 1, 1] = firstmax
    # B = similar(A, (4,4,2));
    mp = MaxPool(2, 2)

    d = rand(SimpleChains.getoutputdim(mp, size(A))...)
    function maxpool(A, mp)
      B = similar(A, SimpleChains.getoutputdim(mp, size(A)))
      SimpleChains.maxpool!(B, A, mp)
      return B
    end
    dot(a, b) = sum(a .* b)
    g = ForwardDiff.gradient(Base.Fix2(dot, d) ∘ Base.Fix2(maxpool, mp), A)
    Ac = copy(A)
    SimpleChains.∂maxpool!(Ac, d, mp)
    firstfdg = (g[1, 1, 1], g[1, 2, 1], g[2, 1, 1], g[2, 2, 1])
    firstcfg = (Ac[1, 1, 1], Ac[1, 2, 1], Ac[2, 1, 1], Ac[2, 2, 1])
    @test sum(iszero, firstfdg) == 3
    @test sum(iszero, firstcfg) == 3
    @test maximum(firstfdg) == maximum(firstcfg)
    (g[1, 1, 1], g[1, 2, 1], g[2, 1, 1], g[2, 2, 1]) = firstcfg
    @test Ac == g
  end
  @testset "LeNet" begin
    include("mnist.jl")
  end
end
# TODO: test ambiguities once ForwardDiff fixes them, or once ForwardDiff is dropped
# For now, there are the tests at the start.
Aqua.test_all(SimpleChains, ambiguities = false, project_toml_formatting = false) 
