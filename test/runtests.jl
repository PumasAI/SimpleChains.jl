using SimpleChains
using Test, Aqua, ForwardDiff

function countallocations!(g, sc, x, p)
  @allocated valgrad!(g, sc, x, p)
end

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
  # let g=g, sc=sc, x=x, p=p
  @test iszero(countallocations!(g, sc, x, p))
  # @test iszero(@allocated(valgrad!(g, sc, x, p)))

  pd = ForwardDiff.Dual.(p, randn.(), randn.(), randn.());
  pu = Vector{UInt8}(undef, sizeof(eltype(pd))*(24*8 + 24*2 + 24));
  xd = ForwardDiff.Dual.(x, randn.(), randn.(), randn.());
  td = TurboDense{true}(tanh, (static(24),static(8)));
  
  Ad = reshape(view(pd, 1:8*24), (8,24));
  bd = view(p, 1+8*24:8*25);
  ld = tanh.(Ad * x .+ bd);
  ldd = tanh.(Ad * xd .+ bd);
  GC.@preserve pd pu begin
    @test ld ≈ td(x, pointer(pd), pointer(pu))[1]
    @test ldd ≈ td(xd, pointer(pd), pointer(pu))[1]
  end
  
end
Aqua.test_all(SimpleChains, ambiguities=false) #TODO: test ambiguities once ForwardDiff fixes them, or once ForwardDiff is dropped

