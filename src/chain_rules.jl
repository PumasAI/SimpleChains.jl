

function ChainRulesCore.rrule(sc::Union{SimpleChain,AbstractPenalty}, arg, params)
  l, g = valgrad(sc, arg, params)
  # assumes no grad w/ respect to arg
  pullback = let g=g
    l̄ -> begin
      if !isone(l̄)
        @turbo for i ∈ eachindex(g)
          g[i] *= l̄
        end
      end
      ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), g
    end
  end
  l, pullback
end


