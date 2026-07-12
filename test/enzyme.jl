using StaticArrays: @SVector

@testset "Enzyme" begin
    sc = SimpleChain(
        static(2),
        TurboDense{true}(identity, static(2))
    )
    p = Float32[1, 2, 3, 4, 5, 6]
    u = @SVector Float32[2, 0]

    loss(sc, u, p) = sum(abs2, sc(u, p))
    y = sc(u, p)
    _, pullback = ChainRules.rrule(sc, u, p)
    _, _, expected_dp = pullback(2y)

    dp = zero(p)
    reverse_mode = isdefined(Enzyme, :set_runtime_activity) ?
        Enzyme.set_runtime_activity(Enzyme.Reverse) : Enzyme.Reverse
    Enzyme.autodiff(
        reverse_mode,
        loss,
        Enzyme.Active,
        Enzyme.Const(sc),
        Enzyme.Const(u),
        Enzyme.Duplicated(p, dp)
    )
    @test dp == expected_dp

    λ = @SVector Float32[1, 1]
    vjpdot(u, p, λ) = sum(sc(u, p) .* λ)
    _, pullback = ChainRules.rrule(sc, u, p)
    expected_du, expected_dp = pullback(λ)[2:3]
    modes = isdefined(Enzyme, :set_runtime_activity) ?
        (Enzyme.Reverse, Enzyme.set_runtime_activity(Enzyme.Reverse)) :
        (Enzyme.Reverse,)
    for mode in modes
        dp = zero(p)
        result = Enzyme.autodiff(
            mode,
            vjpdot,
            Enzyme.Active,
            Enzyme.Active(u),
            Enzyme.Duplicated(p, dp),
            Enzyme.Const(λ)
        )
        du = result[1][1]
        @test du == expected_du
        @test dp == expected_dp
    end

    if isdefined(Enzyme.EnzymeRules, :RevConfig)
        sc_with_loss = SimpleChain(
            static(2),
            TurboDense{true}(identity, static(2)),
            SquaredLoss(@SVector Float32[1, -1])
        )
        _, pullback = ChainRules.rrule(sc_with_loss, u, p)
        _, _, expected_dp = pullback(1.0f0)
        dp = zero(p)
        Enzyme.autodiff(
            Enzyme.Reverse,
            sc_with_loss,
            Enzyme.Active,
            Enzyme.Const(u),
            Enzyme.Duplicated(p, dp)
        )
        @test dp == expected_dp
    end
end
