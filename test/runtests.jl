using SimpleChains
using Test, Aqua, ForwardDiff

function countallocations!(g, sc, x, p)
  @allocated valgrad!(g, sc, x, p)
end

@testset "SimpleChains.jl" begin
  x = rand(24,199);

  y = StrideArray{Float64}(undef, (static(2),size(x,2))) .= randn.() .* 10;
  sc = SimpleChain((TurboDense{true}(tanh, (static(24),static(8))), TurboDense{true}(identity, (static(8),static(2))), SquaredLoss(y)));

  @test first(Dropout(0.5)(x, pointer(x), pointer(sc.memory))) === x
  @test sum(iszero, x) == 0
  x .= rand.();

  p = rand(SimpleChains.numparam(sc)); #pu = Vector{UInt8}(undef,sizeof(Float64)*(24*8 + 24*2 + 24));
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
  @test g ≈ @. gfd
  
  scd = SimpleChain((TurboDense{true}(tanh, (static(24),static(8))), Dropout(0.2), TurboDense{true}(identity, (static(8),static(2))), SquaredLoss(y)));
  valgrad!(g, scd, x, p)
  offset = SimpleChains.align(size(x,2) * 8 * sizeof(Float64)) + SimpleChains.align(8*size(x,2)*8)
  si = SimpleChains.StrideIndex{1,(1,),1}((SimpleChains.StaticInt(1),), (SimpleChains.StaticInt(1),))
  m = SimpleChains.StrideArray(SimpleChains.PtrArray(SimpleChains.stridedpointer(reinterpret(Ptr{SimpleChains.Bit}, pointer(scd.memory) + offset), si), (size(x,2)*8,), Val((true,))), scd.memory)
  gfdd = ForwardDiff.gradient(p) do p
    off = 8*24
    A1 = reshape(view(p, 1:off), (8,24))
    off_old = off
    off += 8
    b1 = view(p, 1+off_old:off)
    l1 = tanh.(A1 * x .+ b1)
    vec(l1) .*= m
    
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
  @test g ≈ gfdd

  # let g=g, sc=sc, x=x, p=p
  @test iszero(countallocations!(g, sc, x, p))
  # @test iszero(@allocated(valgrad!(g, sc, x, p)))

  td = TurboDense{true}(tanh, (static(24),static(8)));
  pd = ForwardDiff.Dual.(p, randn.(), randn.(), randn.());
  xd = ForwardDiff.Dual.(x, randn.(), randn.(), randn.());
  pu = Vector{UInt8}(undef, first(SimpleChains.output_size(Val(eltype(xd)), td, size(x))));
  
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

