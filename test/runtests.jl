using SimpleChains
using Test, Aqua, ForwardDiff, Zygote

function countallocations!(g, sc, x, p)
  @allocated valgrad!(g, sc, x, p)
end

@testset "SimpleChains.jl" begin
  x = rand(24,199);

  y = StrideArray{Float64}(undef, (static(2),size(x,2))) .= randn.() .* 10;
  sc = SimpleChain((Activation(abs2), TurboDense{true}(tanh, (static(24),static(8))), TurboDense{true}(identity, (static(8),static(2))), SquaredLoss(y)));

  @test first(Dropout(0.5)(x, pointer(x), pointer(sc.memory))) === x
  @test sum(iszero, x) == 0
  x .= rand.();

  p = SimpleChains.init_params(sc, Float64);
  g = similar(p);
  valgrad!(g, FrontLastPenalty(sc, L2Penalty(2.3), L1Penalty(0.45)), x, p)
  if VERSION < v"1.8-DEV" # FIXME: remove check when Zygote stops segfaulting on 1.8-DEV 
    @test g == Zygote.gradient(p -> FrontLastPenalty(sc, L2Penalty(2.3), L1Penalty(0.45))(x, p), p)[1]
    _gzyg = Zygote.gradient(p) do p
      0.5*sum(abs2, Base.front(sc)(x, p) .- y)
    end;
    gzyg = copy(_gzyg[1]);
    g2 = similar(g);
    valgrad!(g2, sc, x, p)
    @test g2 ≈ gzyg
  end
  
  gfd = ForwardDiff.gradient(p) do p
    off = 8*24
    A1 = reshape(view(p, 1:off), (8,24))
    off_old = off
    off += 8
    b1 = view(p, 1+off_old:off)
    l1 = tanh.(A1 * abs2.(x) .+ b1)
    
    off_old = off
    off += 8*2
    A2 = reshape(view(p, 1+off_old:off), (2,8))
    
    off_old = off
    off += 2
    b2 = view(p, 1+off_old:off)
    l2 = (A2 * l1 .+ b2)

    l = 0.5mapreduce(+, vec(l2), y.data) do xi, yi
      abs2(xi - yi)
    end
    l + 2.3*(sum(abs2, A1) + sum(abs2, b1)) + 0.45*(sum(abs, A2) + sum(abs, b2))
  end;
  @test g ≈ gfd
  
  scd = SimpleChain((TurboDense{true}(tanh, (static(24),static(8))), Dropout(0.2), TurboDense{true}(identity, (static(8),static(2))), SquaredLoss(y)));
  valgrad!(g, scd, x, p)
  offset = SimpleChains.align(size(x,2) * 8 * sizeof(Float64)) + SimpleChains.align(8*size(x,2)*8)
  si = SimpleChains.StrideIndex{1,(1,),1}((SimpleChains.StaticInt(1),), (SimpleChains.StaticInt(1),))
  m = SimpleChains.StrideArray(SimpleChains.PtrArray(SimpleChains.stridedpointer(reinterpret(Ptr{SimpleChains.Bit}, pointer(scd.memory) + offset), si), (size(x,2)*8,), Val((true,))), scd.memory);
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
  @test iszero(countallocations!(g, FrontLastPenalty(sc, L2Penalty(2.3), NoPenalty()), x, p))
  @test iszero(countallocations!(g, FrontLastPenalty(scd, L2Penalty(2.3), L1Penalty(0.45)), x, p))
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

