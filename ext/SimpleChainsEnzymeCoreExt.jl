module SimpleChainsEnzymeCoreExt

using EnzymeCore
using SimpleChains
using SimpleChains.StaticArrays

const EnzymeRules = EnzymeCore.EnzymeRules

# A `SimpleChain` evaluates through scratch memory whose allocation type deliberately
# disagrees with the values stored through it, which Enzyme's type analysis rejects.
# The chain is therefore treated as the differentiation boundary and the reverse pass
# delegates to SimpleChains' own hand written pullbacks.
#
# The whole VJP runs inside `EnzymeRules.reverse`. That is load bearing: the scratch
# buffer backing a pullback comes from `get_heap_memory`, which is task local and keyed
# by chain type, so a pullback held across the augmented/reverse boundary is corrupted
# by any intervening call to a chain of the same type — Enzyme runs every augmented
# pass before the first reverse pass, so `sc(x, p) + sc(y, p)` is enough to trigger it.
# Recomputing the forward sweep here keeps the buffer's lifetime inside a single call.

@inline _mutable_like(x::AbstractArray{T}) where {T} = zeros(T, size(x))

# `pullback_arg!` is only defined for `PtrArray` inputs, so an input gradient needs the
# argument materialized out of a `StaticArray` first.
@inline _materialize(x::StaticArrays.StaticArray) = Array(x)
@inline _materialize(x) = x

# Loss terminated chains return a scalar and `valgrad!` fills both gradients. The
# `(ga, gp)` form is what makes the input gradient available, which the ChainRules
# `ElementwisePullback` path does not provide, but it costs a materialized argument, so
# an inactive input takes the cheaper parameter-only path.
function chain_vjp(sc, arg, params, seed::Number, wants_arg::Bool)
    gp = _mutable_like(params)
    ga = if wants_arg
        marg = _materialize(arg)
        g = _mutable_like(marg)
        SimpleChains.valgrad!((g, gp), sc, marg, params)
        g
    else
        SimpleChains.valgrad!(gp, sc, arg, params)
        nothing
    end
    if !isone(seed)
        gp .*= seed
        wants_arg && (ga .*= seed)
    end
    return ga, gp
end

function chain_vjp(sc, arg, params, seed, _wants_arg::Bool)
    _, pullback = SimpleChains.valgrad_noloss(sc, arg, params)
    _, ga, gp = pullback(seed)
    return ga, gp
end

# With runtime activity Enzyme may hand a rule a shadow that is the primal itself for
# values that turn out to be inactive; accumulating there would corrupt the primal.
@inline function accumulate!(config, annotation, tangent, batch)
    annotation isa Union{EnzymeCore.Const, EnzymeCore.Active} && return nothing
    target = EnzymeRules.width(config) == 1 ? annotation.dval : annotation.dval[batch]
    EnzymeRules.runtime_activity(config) && target === annotation.val && return nothing
    target .+= tangent
    return nothing
end

@inline active_tangent(a::EnzymeCore.Active{T}, tangent) where {T} = convert(T, tangent)::T
@inline active_tangent(::EnzymeCore.Annotation, _) = nothing

# When Enzyme cannot infer the chain call's return type it reports `RealRt === Any`,
# takes its generic reverse path, and accumulates through a `Base.RefValue` for
# immutable results (its `MixedDuplicated` convention) -- a bare value is rejected
# there. In that case the expected shadow type is `Any`, so the `Ref` needs no flexible
# shadow return. When the return type is known, an immutable result is `Active` and no
# shadow is requested at all, so a plain zero is always right.
@inline _cell(::Type{Any}, result) = ismutable(result) ? zero(result) : Ref(zero(result))
@inline _cell(::Type, result) = zero(result)

@inline _seed(shadow::Base.RefValue) = shadow[]
@inline _seed(shadow) = shadow

@inline function augmented_shadow(config, ::Type{RT}, result) where {RT}
    EnzymeRules.needs_shadow(config) || return nothing
    width = EnzymeRules.width(config)
    RealRt = Base.eltype(RT)
    return width == 1 ? _cell(RealRt, result) :
        ntuple(_ -> _cell(RealRt, result), Val(width))
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        fn::EnzymeCore.Const{<:SimpleChains.SimpleChain},
        ::Type{RT},
        arg::EnzymeCore.Annotation,
        params::EnzymeCore.Annotation
    ) where {RT}
    overwritten = EnzymeRules.overwritten(config)
    saved_arg = overwritten[2] ? copy(arg.val) : arg.val
    saved_params = overwritten[3] ? copy(params.val) : params.val
    result = fn.val(saved_arg, saved_params)
    shadow = augmented_shadow(config, RT, result)
    tape = (shadow, saved_arg, saved_params)
    return EnzymeRules.augmented_rule_return_type(config, RT){typeof(tape)}(
        EnzymeRules.needs_primal(config) ? result : nothing, shadow, tape
    )
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        fn::EnzymeCore.Const{<:SimpleChains.SimpleChain},
        dret,
        tape,
        arg::EnzymeCore.Annotation,
        params::EnzymeCore.Annotation
    )
    shadow, saved_arg, saved_params = tape
    width = EnzymeRules.width(config)
    tangents = ntuple(Val(width)) do batch
        seed = if dret isa EnzymeCore.Active
            dret.val
        elseif dret isa Tuple
            # Batched immutable returns arrive as one `Active` annotation per batch.
            dret[batch].val
        else
            _seed(width == 1 ? shadow : shadow[batch])
        end
        ga, gp = chain_vjp(
            fn.val, saved_arg, saved_params, seed, !(arg isa EnzymeCore.Const)
        )
        accumulate!(config, arg, ga, batch)
        accumulate!(config, params, gp, batch)
        (active_tangent(arg, ga), active_tangent(params, gp))
    end
    arg_tangent = arg isa EnzymeCore.Active ?
        (width == 1 ? tangents[1][1] : ntuple(i -> tangents[i][1], Val(width))) : nothing
    params_tangent = params isa EnzymeCore.Active ?
        (width == 1 ? tangents[1][2] : ntuple(i -> tangents[i][2], Val(width))) : nothing
    return (arg_tangent, params_tangent)
end

end
