using SimpleChains, ForwardDiff, Test

struct Tag end;
tagtype(::Type{T}) where {T<:Number} = typeof(ForwardDiff.Tag(Tag(), T))

function dual4x3(x::T) where {T}
  ForwardDiff.Dual{tagtype(T)}(x, randn(), randn(), randn(), randn())
end
function dual4x3(x::T) where {T<:ForwardDiff.Dual}
  ForwardDiff.Dual{tagtype(T)}(x, dual4x3(randn()), dual4x3(randn()), dual4x3(randn()))
end



for bias in (true, false)
  M = 16
  K = 20
  N = 17
  A = rand(M, K + bias)
  B = rand(K, N)
  bm = rand(K, 1)

  for fa1 in (identity, dual4x3),
    fa2 in (identity, dual4x3),
    fb1 in (identity, dual4x3),
    fb2 in (identity, dual4x3)

    let A = fa2.(fa1.(A)), B = fb2.(fb1.(B)), bm = fb2.(fb1.(bm)), b = vec(bm)
      T = Base.promote_eltype(A, B)
      T === Float64 && continue
      C = Matrix{T}(undef, M, N)
      cm = Matrix{T}(undef, M, 1)
      c = Vector{T}(undef, M)

      AB = if bias
        @view(A[:,begin:end-1]) * B .+ @view(A[:,end])
      else
        A * B
      end
      Ab = if bias
        @view(A[:,begin:end-1]) * b .+ @view(A[:,end])
      else
        A * b
      end
      
      SimpleChains.matmul!(C, A, B, static(bias))
      @test C ≈ AB
      SimpleChains.matmul!(c, A, b, static(bias))
      @test c ≈ Ab
      SimpleChains.matmul!(c, A, bm, static(bias))
      @test c ≈ Ab
      SimpleChains.matmul!(cm, A, b, static(bias))
      @test vec(cm) ≈ Ab
      SimpleChains.matmul!(cm, A, bm, static(bias))
      @test vec(cm) ≈ Ab
    end
  end
end

