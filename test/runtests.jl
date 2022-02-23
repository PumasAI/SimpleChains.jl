using SimpleChains
using Test, Aqua, ForwardDiff, Zygote

function countallocations!(g, sc, x, p)
    @allocated valgrad!(g, sc, x, p)
end
dual(x) = ForwardDiff.Dual(x, randn(), randn(), randn())
dual(x::ForwardDiff.Dual) = ForwardDiff.Dual(x, dual(randn()), dual(randn()))

@testset "SimpleChains.jl" begin
    # Should throw, as 8 ≠ 10
    @test_throws ArgumentError SimpleChain((
        Activation(abs2), 
        TurboDense{true}(tanh, (static(24), static(8))), 
        TurboDense{true}(identity, (static(10), static(2))))
    )

    x = rand(24, 199)

    y = StrideArray{Float64}(undef, (static(2), size(x, 2))) .= randn.() .* 10
    sc = SimpleChain((Activation(abs2), TurboDense{true}(tanh, (static(24), static(8))), TurboDense{true}(identity, (static(8), static(2))), SquaredLoss(y)))


    @test first(Dropout(0.5)(x, pointer(x), pointer(sc.memory))) === x
    @test sum(iszero, x) == 0
    x .= rand.()

    scflp = FrontLastPenalty(sc, L2Penalty(2.3), L1Penalty(0.45))
    print_str0 = """
SimpleChain with the following layers:
Activation layer applying: abs2
TurboDense (static(24), static(8)) with bias.
Activation layer applying: tanh
TurboDense (static(8), static(2)) with bias.
SquaredLoss"""
    print_str1 = """
Penalty on all but last layer: L2Penalty (λ=2.3)
Penalty on last layer: L1Penalty (λ=0.45) applied to:
SimpleChain with the following layers:
Activation layer applying: abs2
TurboDense (static(24), static(8)) with bias.
Activation layer applying: tanh
TurboDense (static(8), static(2)) with bias.
SquaredLoss"""

    @test sprint((io, t) -> show(io, t), sc) == print_str0
    if VERSION >= v"1.6"
        @test sprint((io, t) -> show(io, t), scflp) == print_str1
    else
        # typename doesn't work on 1.5
        @test_broken sprint((io, t) -> show(io, t), scflp) == print_str1
    end

    p = SimpleChains.init_params(sc, Float64)
    g = similar(p)
    valgrad!(g, scflp, x, p)
    if VERSION < v"1.8-DEV" # FIXME: remove check when Zygote stops segfaulting on 1.8-DEV 
        @test g == Zygote.gradient(p -> FrontLastPenalty(sc, L2Penalty(2.3), L1Penalty(0.45))(x, p), p)[1]
        _gzyg = Zygote.gradient(p) do p
            0.5 * sum(abs2, Base.front(sc)(x, p) .- y)
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
        l + 2.3 * (sum(abs2, A1) + sum(abs2, b1)) + 0.45 * (sum(abs, A2) + sum(abs, b2))
    end
    @test g ≈ gfd

    scd = SimpleChain((TurboDense{true}(tanh, (static(24), static(8))), Dropout(0.2), TurboDense{true}(identity, (static(8), static(2))), SquaredLoss(y)))
    valgrad!(g, scd, x, p)
    offset = SimpleChains.align(size(x, 2) * 8 * sizeof(Float64)) + SimpleChains.align(8 * size(x, 2) * 8)
    si = SimpleChains.StrideIndex{1,(1,),1}((SimpleChains.StaticInt(1),), (SimpleChains.StaticInt(1),))
    m = SimpleChains.StrideArray(SimpleChains.PtrArray(SimpleChains.stridedpointer(reinterpret(Ptr{SimpleChains.Bit}, pointer(scd.memory) + offset), si), (size(x, 2) * 8,), Val((true,))), scd.memory)
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

        0.5mapreduce(+, vec(l2), y.data) do xi, yi
            abs2(xi - yi)
        end
    end
    @test g ≈ gfdd

    # let g=g, sc=sc, x=x, p=p
    @test iszero(countallocations!(g, FrontLastPenalty(sc, L2Penalty(2.3), NoPenalty()), x, p))
    @test iszero(countallocations!(g, FrontLastPenalty(scd, L2Penalty(2.3), L1Penalty(0.45)), x, p))
    # @test iszero(@allocated(valgrad!(g, sc, x, p)))

    td = TurboDense{true}(tanh, (static(24), static(8)))
    pd = dual.(p)
    xd = dual.(x)

    pdd = dual.(pd)
    xdd = dual.(xd)

    pu = Vector{UInt8}(undef, first(SimpleChains.output_size(Val(eltype(xdd)), td, size(x))))

    Ad = reshape(view(pd, 1:8*24), (8, 24))
    bd = view(pd, 1+8*24:8*25)
    ld = tanh.(Ad * x .+ bd)
    ld_d = tanh.(Ad * xd .+ bd)

    Add = reshape(view(pdd, 1:8*24), (8, 24))
    bdd = view(pdd, 1+8*24:8*25)
    ldd = tanh.(Add * x .+ bdd)
    ldd_dd = tanh.(Add * xdd .+ bdd)
    GC.@preserve pd pu begin
        @test reinterpret(Float64, ld) ≈ reinterpret(Float64, td(x, pointer(pd), pointer(pu))[1])
        @test reinterpret(Float64, ld_d) ≈ reinterpret(Float64, td(xd, pointer(pd), pointer(pu))[1])

        @test reinterpret(Float64, ldd) ≈ reinterpret(Float64, td(x, pointer(pdd), pointer(pu))[1])
        @test reinterpret(Float64, ldd_dd) ≈ reinterpret(Float64, td(xdd, pointer(pdd), pointer(pu))[1])
    end
    @testset "training" begin
        p .= randn.() .* 100
        # small penalties since we don't care about overfitting here
        vg1 = valgrad!(g, FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)), x, p)
        SimpleChains.train!(g, p, FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)), x, SimpleChains.ADAM(), 1000)
        @test valgrad!(g, FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)), x, p) < vg1
        p .= randn.() .* 100
        vg2 = FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))(x, p)
        SimpleChains.train!(g, p, FrontLastPenalty(scd, L2Penalty(2.3), L1Penalty(0.45)), x, SimpleChains.ADAM(), 1000)
        @test FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))(x, p) < vg2
    end
    @testset "vector of targets" begin
        p .= randn.() .* 100
        ys = [@. y + 0.1randn() for _ in 1:1000]
        # small penalties since we don't care about overfitting here
        vg1 = valgrad!(g, FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)), x, p)
        SimpleChains.train_unbatched!(g, p, FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)), x, SimpleChains.ADAM(), ys)
        @test valgrad!(g, FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4)), x, p) < vg1
        p .= randn.() .* 100
        vg2 = FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))(x, p)
        SimpleChains.train_unbatched!(g, p, FrontLastPenalty(scd, L2Penalty(2.3), L1Penalty(0.45)), x, SimpleChains.ADAM(), ys)
        @test FrontLastPenalty(sc, L2Penalty(1e-4), L1Penalty(1e-4))(x, p) < vg2

    end
end
Aqua.test_all(SimpleChains, ambiguities = false) #TODO: test ambiguities once ForwardDiff fixes them, or once ForwardDiff is dropped

