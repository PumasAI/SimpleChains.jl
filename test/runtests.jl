using SimpleChains
using Test, Aqua, ForwardDiff, Zygote, ChainRules, Random
using JET: @test_opt

countallocations!(g, sc, x, p) = @allocated valgrad!(g, sc, x, p)
dual(x::T) where {T} = ForwardDiff.Dual(x, 4randn(T), 4randn(T), 4randn(T))
function dual(x::ForwardDiff.Dual{<:Any,T}) where {T}
  ForwardDiff.Dual(x, dual(4randn(T)), dual(4randn(T)))
end

import InteractiveUtils
InteractiveUtils.versioninfo(; verbose = true)

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
            TurboDense{true}(identity, static(2))
          )
        ) # test inputdim
        scdbase = SimpleChain((
          TurboDense{true,Int}(tanh, 8),
          Dropout(0.2),
          TurboDense{true,Int}(identity, 2)
        )) # test inputdim unknown
      else # test construction using `Vararg`
        scbase = SimpleChain(
          static(24),
          Activation(abs2),
          TurboDense{true}(tanh, static(8)),
          TurboDense{true}(identity, static(2))
        ) # test inputdim
        scdbase = SimpleChain(
          TurboDense{true,Int}(tanh, 8),
          Dropout(0.2),
          TurboDense{true,Int}(identity, 2)
        ) # test inputdim unknown
      end

      x = rand(T, 24, 199)

      y = StrideArray{T}(undef, (static(2), size(x, 2))) .= randn.() .* 10
      sc = SimpleChains.add_loss(scbase, SquaredLoss(y))

      @test first(
        Dropout(0.5)(
          x,
          pointer(x),
          pointer(SimpleChains.get_heap_memory(sc, 0))
        )
      ) === x
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
      @test sprint((io, t) -> show(io, t), scflp) == print_str1

      p = SimpleChains.init_params(scflp, T; rng = Random.default_rng())
      g = similar(p)
      let sc = SimpleChains.remove_loss(sc)
        @test_throws ArgumentError sc(rand(T, 23, 2), p)
        @test_throws ArgumentError sc(rand(T, 23), p)
        @test_throws MethodError sc(Array{T,0}(undef), p)
        @test_throws ArgumentError valgrad!(g, sc, rand(T, 23, 2), p)
        @test_throws ArgumentError valgrad!(g, sc, rand(T, 23), p)
      end
      g1 = similar(g)
      g3 = similar(g)
      gx0 = similar(x)
      gx1 = similar(x)
      let ret = scflp(x, p)
        @test_opt valgrad!(g, scflp, x, p)
        @test_opt valgrad!((gx0, g1), scflp, x, p)
        @test_opt valgrad!((nothing, g3), scflp, x, p)
        @test_opt valgrad!((gx1, nothing), scflp, x, p)
        @test ret ≈ valgrad!(g, scflp, x, p)
        @test ret ≈ valgrad!((gx0, g1), scflp, x, p)
        @test ret ≈ valgrad!((nothing, g3), scflp, x, p)
        @test ret ≈ valgrad!((gx1, nothing), scflp, x, p)
      end
      @test g == only(Zygote.gradient(Base.Fix1(scflp, x), p)) == g1 == g3
      # 0.5 and .*= 2.0 is to test !isone path in ElementwisePullback
      @test ((ChainRules.rrule(scflp, x, p)[2](0.5)[3]) .*= 2.0) ≈ g
      _gzyg = Zygote.gradient(p) do p
        sum(abs2, Base.front(sc)(x, p) .- y) / 2# / size(x)[end]
      end
      gzyg = copy(_gzyg[1])
      g2 = similar(g)
      @test sc(x, p) ≈ valgrad!(g2, sc, x, p)
      @test g2 ≈ gzyg

      function f(x, p, y)
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

        l = mapreduce(+, vec(l2), y) do xi, yi
          abs2(xi - yi)
        end / 2 #/size(x)[end]
        l +
        2.3 * (sum(abs2, A1) + sum(abs2, b1)) +
        0.45 * (sum(abs, A2) + sum(abs, b2))
      end
      gfd = ForwardDiff.gradient(p -> f(x, p, y.data), p)
      @test g ≈ gfd ≈ g1 ≈ g3
      gxfd = ForwardDiff.gradient(x -> f(x, p, y.data), x)
      @test gxfd ≈ gx0 ≈ gx1
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
      offset =
        2SimpleChains.align(
          first(scd.layers).outputdim * size(x, 2) * sizeof(T)
        )
      si = SimpleChains.StrideIndex{1,(1,),1}(
        (SimpleChains.StaticInt(1),),
        (SimpleChains.StaticInt(1),)
      )
      scdmem = SimpleChains.get_heap_memory(scd, 0)
      m = SimpleChains.StrideArray(
        SimpleChains.PtrArray(
          SimpleChains.stridedpointer(
            Ptr{SimpleChains.Bit}(pointer(scdmem) + offset),
            si
          ),
          (size(x, 2) * 8,),
          Val((true,))
        ),
        scdmem
      )

      valgrad!(g, scd, x, p)
      offset =
        2SimpleChains.align(
          first(scd.layers).outputdim * size(x, 2) * sizeof(T)
        )
      si = SimpleChains.StrideIndex{1,(1,),1}(
        (SimpleChains.StaticInt(1),),
        (SimpleChains.StaticInt(1),)
      )
      m = SimpleChains.StrideArray(
        SimpleChains.PtrArray(
          SimpleChains.stridedpointer(
            Ptr{SimpleChains.Bit}(pointer(scdmem) + offset),
            si
          ),
          (size(x, 2) * 8,),
          Val((true,))
        ),
        scdmem
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
        _, (W1, x1), (W2, x2) = SimpleChains.params(L2Penalty(sc, 2.3), p)
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

        mapreduce(+, vec(l2), y.data) do xi, yi
          abs2(xi - yi)
        end / 2# / size(x)[end]
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
      @test countallocations!(
        g,
        FrontLastPenalty(sc, L2Penalty(2.3), NoPenalty()),
        x,
        p
      ) == 0
      @test countallocations!(
        g,
        FrontLastPenalty(scd, L2Penalty(2.3), L1Penalty(0.45)),
        x,
        p
      ) == 0
      # @test iszero(@allocated(valgrad!(g, sc, x, p)))

      td = TurboDense{true}(tanh, static(8))
      pd = dual.(p)
      xd = dual.(x)

      pdd = dual.(pd)
      xdd = dual.(xd)

      pu = Vector{UInt8}(
        undef,
        first(SimpleChains.layer_output_size(Val(eltype(xdd)), td, size(x)))
      )
      dim = size(x, 1)
      A = reshape(view(p, 1:8dim), (8, dim))
      b = view(p, 1+8dim:8*25)
      Ad = reshape(view(pd, 1:8dim), (8, dim))
      bd = view(pd, 1+8dim:8*25)
      ld = tanh.(Ad * x .+ bd)
      l_d = tanh.(A * xd .+ b)
      ld_d = tanh.(Ad * xd .+ bd)

      Add = reshape(view(pdd, 1:8dim), (8, dim))
      bdd = view(pdd, 1+8dim:8*25)
      ldd = tanh.(Add * x .+ bdd)
      ldd_dd = tanh.(Add * xdd .+ bdd)
      # call crashes julia td(xdd, pointer(pdd), pointer(pu))
      GC.@preserve pd pu begin
        @test reinterpret(T, td(x, pointer(pd), pointer(pu))[1]) ==
              reinterpret(T, SimpleChain(td)(x, pd))
        @test reinterpret(T, ld) ≈
              reinterpret(T, td(x, pointer(pd), pointer(pu))[1])
        @test reinterpret(T, ld) ≈
              reinterpret(T, td(permutedims(x)', pointer(pd), pointer(pu))[1])
        @test reinterpret(T, l_d) ≈
              reinterpret(T, td(xd, pointer(p), pointer(pu))[1])
        @test reinterpret(T, l_d) ≈
              reinterpret(T, td(permutedims(xd)', pointer(p), pointer(pu))[1])
        @test reinterpret(T, ld_d) ≈
              reinterpret(T, td(xd, pointer(pd), pointer(pu))[1])
        @test reinterpret(T, ld_d) ≈
              reinterpret(T, td(permutedims(xd)', pointer(pd), pointer(pu))[1])

        @test reinterpret(T, ldd) ≈
              reinterpret(T, td(x, pointer(pdd), pointer(pu))[1])
        @test reinterpret(T, ldd_dd) ≈
              reinterpret(T, td(xdd, pointer(pdd), pointer(pu))[1])
        @test reinterpret(T, ldd) ≈
              reinterpret(T, td(permutedims(x)', pointer(pdd), pointer(pu))[1])
        @test reinterpret(T, ldd_dd) ≈ reinterpret(
          T,
          td(permutedims(xdd)', pointer(pdd), pointer(pu))[1]
        )
      end
      @testset "training" begin
        p .= randn.() .* 100
        # small penalties since we don't care about overfitting here
        scpen = FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))
        vg1 = valgrad!(g, scpen, x, p)
        gt = SimpleChains.alloc_threaded_grad(scpen, T)
        SimpleChains.train!(gt, p, scpen, x, SimpleChains.ADAM(), 1000)
        @test valgrad!(
          g,
          FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)),
          x,
          p
        ) < vg1
        p .= randn.() .* 100
        vg2 = FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))(x, p)
        SimpleChains.train!(
          gt,
          p,
          FrontLastPenalty(scd, L2Penalty(2.3), L1Penalty(0.45)),
          x,
          SimpleChains.ADAM(),
          1000
        )
        @test FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))(x, p) < vg2
      end
      @testset "vector of targets" begin
        p .= randn.() .* 100
        ys = [@. y + 0.1randn() for _ = 1:1000]
        # small penalties since we don't care about overfitting here
        scpen = FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))
        vg1 = valgrad!(g, scpen, x, p)
        gt = SimpleChains.alloc_threaded_grad(scpen, T)
        SimpleChains.train_unbatched!(gt, p, scpen, x, SimpleChains.ADAM(), ys)
        @test valgrad!(
          g,
          FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)),
          x,
          p
        ) < vg1
        p .= randn.() .* 100
        vg2 = FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))(x, p)
        SimpleChains.train_unbatched!(
          gt,
          p,
          FrontLastPenalty(scd, L2Penalty(2.3), L1Penalty(0.45)),
          x,
          SimpleChains.ADAM(),
          ys
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
        Array{Base.promote_eltype(x, K, b)}(
          undef,
          d1,
          d2,
          size(K, 4),
          size(x, 4)
        )
      SimpleChains.convlayer!(
        f,
        closuresshouldntbeabletomutatebindings,
        x,
        K,
        b
      )
      return closuresshouldntbeabletomutatebindings
    end

    function convlayertest(x, y, K, b)
      closuresshouldntbeabletomutatebindings = clt(relu, x, K, b)
      δ = vec(closuresshouldntbeabletomutatebindings) .- vec(y)
      (δ'δ) / 2
    end
    function convlayertest(x, y, K0, b0, K1, b1)
      csbamb = clt(identity, clt(relu, x, K0, b0), K1, b1)
      closuresshouldntbeabletomutatebindings = tanh.(csbamb)
      δ = vec(closuresshouldntbeabletomutatebindings) .- vec(y)
      (δ'δ) / 2
    end

    scconv = SimpleChain(
      Conv(relu, (3, 3), 4)
      #  Conv(tanh, (5,5), 4, 3)
    )
    batch = 3
    x = rand(Float32, 28, 28, 1, batch)
    y = rand(Float32, 26, 26, 4, batch)
    scconvl = SimpleChains.add_loss(scconv, SquaredLoss(y))
    p = @inferred(SimpleChains.init_params(scconvl, size(x)))
    (K, b), p2 = @inferred(
      SimpleChains.getparams(first(scconv.layers), pointer(p), size(x))
    )
    refloss = convlayertest(x, y, K, b)
    @test scconvl(x, p) ≈ refloss
    g = similar(p)
    @test @inferred(valgrad!(g, scconvl, x, p)) ≈ refloss

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
      Conv(tanh, (5, 5), 3)  # fast_fuse == false
    )
    z = rand(Float32, 22, 22, 3, batch)
    scconv2l = @inferred(SimpleChains.add_loss(scconv2, SquaredLoss(z)))
    p = @inferred(SimpleChains.init_params(scconv2l, size(x)))
    (K0, b0), p2 = @inferred(
      SimpleChains.getparams((scconv2.layers)[1], pointer(p), size(x))
    )
    (K1, b1), p3 =
      @inferred(SimpleChains.getparams((scconv2.layers)[2], p2, (1, 1, 4, 1)))
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
  @testset "Layer Tests" begin
    include("layer_tests.jl")
  end
  @testset "Matmul Tests" begin
    include("matmul_tests.jl")
  end
  @testset "params" begin
    sc = SimpleChain(
      static(24),
      (
        Activation(abs2),
        TurboDense{true}(tanh, static(8)),
        Dropout(0.5),
        TurboDense{false}(identity, static(2))
      )
    )
    p = SimpleChains.init_params(sc)
    n0, (W1, b1), n2, W3 = SimpleChains.params(sc, p)
    @test n0 === n2 === nothing
    @test W1 == reshape(view(p, 1:24*8), (8, 24))
    @test b1 == view(p, 24*8+1:25*8)
    @test W3 == reshape(@view(p[25*8+1:end]), (2, 8))
    n01, W11, n21, W31 = SimpleChains.weights(sc, p)
    n02, b12, n22, n3 = SimpleChains.biases(sc, p)
    @test n01 === n21 === n02 === n22 === n3
    @test W11 === W1
    @test W31 === W3
    @test b12 === b1
  end
  @testset "dualeval!" begin
    x = fill(ForwardDiff.Dual(ntuple(x -> ((x - 9) / 5), Val(18))...), 10)
    SimpleChains.dualeval!(tanh, x)
    @test reinterpret(reshape, Float64, x) ≈ reinterpret(
      reshape,
      Float64,
      fill(
        (
          -0.9216685544064713,
          -0.21073790614559954,
          -0.18063249098194248,
          -0.1505270758182854,
          -0.12042166065462832,
          -0.09031624549097124,
          -0.06021083032731416,
          -0.03010541516365708,
          0.0,
          0.03010541516365708,
          0.06021083032731416,
          0.09031624549097124,
          0.12042166065462832,
          0.1505270758182854,
          0.18063249098194248,
          0.21073790614559954,
          0.24084332130925665,
          0.27094873647291373
        ),
        10
      )
    )
  end
  @testset "SArray" begin
    include("staticarrays.jl")
  end
  @testset "Glorot" begin
    include("random.jl")
  end
  @testset "Batch" begin
    include("batch.jl")
  end
end
# TODO: test ambiguities once ForwardDiff fixes them, or once ForwardDiff is dropped
# For now, there are the tests at the start.
Aqua.test_all(
  SimpleChains;
  ambiguities = false,
  project_toml_formatting = false
)
