using SimpleChains
using Test, Aqua, ForwardDiff

@testset "SimpleChains.jl" begin
  p = rand(8*25 + 2*9); #pu = Vector{UInt8}(undef,sizeof(Float64)*(24*8 + 24*2 + 24));
  x = rand(24,24);

  y = StrideArray{Float64}(undef, (static(2),24)) .= randn.() .* 10;
  sc = SimpleChain((TurboDense{true}(tanh, (static(24),static(8))), TurboDense{true}(identity, (static(8),static(2))), SquaredLoss(y)));
  g = similar(p);

  valgrad!(g, sc, x, p)

  gfd = ForwardDiff.gradient(p) do p
    off = 8*24
    A1 = reshape(view(p, 1:off), (8,24))
    off_old = off
    off += 8
    b1 = view(p, 1+off_old:off)
    l1 = tanh.(A1 * x .+ b1)
    
    off_old = off
    off += 8*2
    A2 = reshape(view(p, 1+off_old:off), (2,8))
    
    off_old = off
    off += 2
    b2 = view(p, 1+off_old:off)
    l2 = (A2 * l1 .+ b2)

    0.5mapreduce(+, vec(l2), y.data) do xi, yi
      abs2(xi - yi)
    end  
  end;

  @test g ≈ gfd

  @test iszero(@allocated(valgrad!(g, sc, x, p)))
end
Aqua.test_all(SimpleChains, ambiguities=false) #TODO: test ambiguities once ForwardDiff fixes them, or once ForwardDiff is dropped

