module SimpleChainsEnzymeCoreExt

using ChainRulesCore
using EnzymeCore
using SimpleChains

const EnzymeRules = EnzymeCore.EnzymeRules
const ReverseConfig = if isdefined(EnzymeRules, :RevConfig)
    EnzymeRules.RevConfig
else
    EnzymeRules.Config
end
const ReverseConfigWidth = if isdefined(EnzymeRules, :RevConfigWidth)
    EnzymeRules.RevConfigWidth
else
    EnzymeRules.ConfigWidth
end
const RuleReturnAnnotation = if isdefined(EnzymeRules, :augmented_rule_return_type)
    EnzymeCore.Annotation
else
    Union{
        EnzymeCore.Const,
        EnzymeCore.Active{<:AbstractArray},
        EnzymeCore.Duplicated,
        EnzymeCore.DuplicatedNoNeed,
        EnzymeCore.BatchDuplicated,
        EnzymeCore.BatchDuplicatedNoNeed,
    }
end

@inline function EnzymeRules.augmented_primal(
        config::ReverseConfig,
        fn::EnzymeCore.Const{<:SimpleChains.SimpleChain},
        ::Type{ReturnAnnotation},
        arg::ArgumentAnnotation,
        params::ParameterAnnotation
    ) where {
        ReturnAnnotation <: RuleReturnAnnotation,
        ArgumentAnnotation <: EnzymeCore.Annotation,
        ParameterAnnotation <: EnzymeCore.Annotation,
    }
    arg_primal = EnzymeRules.overwritten(config)[2] ? deepcopy(arg.val) : arg.val
    params_primal =
        EnzymeRules.overwritten(config)[3] ? deepcopy(params.val) : params.val
    result, pullback = ChainRulesCore.rrule(fn.val, arg_primal, params_primal)
    primal = EnzymeRules.needs_primal(config) ? result : nothing
    @static if isdefined(EnzymeRules, :augmented_rule_return_type)
        if ReturnAnnotation <: EnzymeCore.Const
            return EnzymeRules.augmented_rule_return_type(
                config,
                ReturnAnnotation
            ){Nothing}(primal, nothing, nothing)
        elseif ReturnAnnotation <: EnzymeCore.Active
            return EnzymeRules.augmented_rule_return_type(
                config,
                ReturnAnnotation
            ){typeof(pullback)}(primal, nothing, pullback)
        end

        shadow = if !EnzymeRules.needs_shadow(config)
            nothing
        elseif EnzymeRules.width(config) == 1
            Ref(zero(result))
        else
            ntuple(_ -> Ref(zero(result)), Val(EnzymeRules.width(config)))
        end
        tape = (shadow, pullback)
        return EnzymeRules.augmented_rule_return_type(
            config,
            ReturnAnnotation
        ){typeof(tape)}(primal, shadow, tape)
    else
        shadow = if !EnzymeRules.needs_shadow(config)
            nothing
        elseif EnzymeRules.width(config) == 1
            zero(result)
        else
            ntuple(_ -> zero(result), Val(EnzymeRules.width(config)))
        end
        return EnzymeRules.AugmentedReturn(primal, shadow, (shadow, pullback))
    end
end

function accumulate_tangent!(annotation, tangent, batch, width)
    if annotation isa Union{EnzymeCore.Const, EnzymeCore.Active} ||
            tangent isa ChainRulesCore.NoTangent
        return nothing
    end
    target = width == 1 ? annotation.dval : annotation.dval[batch]
    target .+= ChainRulesCore.unthunk(tangent)
    return nothing
end

function active_tangent(annotation::EnzymeCore.Active{T}, tangent)::T where {T}
    tangent isa ChainRulesCore.NoTangent && return zero(annotation.val)
    return convert(T, ChainRulesCore.unthunk(tangent))
end

active_tangent(::EnzymeCore.Annotation, tangent) = nothing

function pullback_tangents(config, pullback, seed, arg, params, batch)
    tangents = pullback(seed)
    width = EnzymeRules.width(config)
    accumulate_tangent!(arg, tangents[2], batch, width)
    accumulate_tangent!(params, tangents[3], batch, width)
    return active_tangent(arg, tangents[2]), active_tangent(params, tangents[3])
end

function EnzymeRules.reverse(
        config::ReverseConfig,
        ::EnzymeCore.Const{<:SimpleChains.SimpleChain},
        ::Type{<:EnzymeCore.Const},
        tape,
        arg::EnzymeCore.Annotation,
        params::EnzymeCore.Annotation
    )
    return nothing, nothing
end

function EnzymeRules.reverse(
        config::ReverseConfig,
        ::EnzymeCore.Const{<:SimpleChains.SimpleChain},
        ::Type{ReturnAnnotation},
        tape,
        arg::EnzymeCore.Annotation,
        params::EnzymeCore.Annotation
    ) where {ReturnAnnotation <: EnzymeCore.Annotation}
    shadow, pullback = tape
    width = EnzymeRules.width(config)
    tangents = ntuple(Val(width)) do batch
        @static if isdefined(EnzymeRules, :augmented_rule_return_type)
            seed = width == 1 ? shadow[] : shadow[batch][]
        else
            seed = width == 1 ? shadow : shadow[batch]
        end
        pullback_tangents(config, pullback, seed, arg, params, batch)
    end
    arg_tangent = arg isa EnzymeCore.Active ?
        (width == 1 ? tangents[1][1] : ntuple(i -> tangents[i][1], Val(width))) :
        nothing
    params_tangent = params isa EnzymeCore.Active ?
        (width == 1 ? tangents[1][2] : ntuple(i -> tangents[i][2], Val(width))) :
        nothing
    return arg_tangent, params_tangent
end

function EnzymeRules.reverse(
        config::ReverseConfigWidth{1},
        ::EnzymeCore.Const{<:SimpleChains.SimpleChain},
        return_tangent::EnzymeCore.Active,
        tape,
        arg::EnzymeCore.Annotation,
        params::EnzymeCore.Annotation
    )
    @static if isdefined(EnzymeRules, :augmented_rule_return_type)
        pullback = tape
    else
        pullback = tape[2]
    end
    return pullback_tangents(config, pullback, return_tangent.val, arg, params, 1)
end

end
