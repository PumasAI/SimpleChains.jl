
@generated function _flatten(x::ForwardDiff.Dual{<:Any,<:NativeTypesV,N}) where {N}
  t = Expr(:tuple, :(ForwardDiff.value(x)))
  for n = 1:N
    push!(t.args, :(@inbounds p[$n]))
  end
  Expr(:block, Expr(:meta, :inline), :(p = ForwardDiff.partials(x)), t)
end
@generated function _flatten(x::ForwardDiff.Dual{<:Any,<:ForwardDiff.Dual,N}) where {N}
  t = Expr(:tuple, :(_flatten(ForwardDiff.value(x))...))
  for n = 1:N
    push!(t.args, :(_flatten(@inbounds(p[$n]))...))
  end
  Expr(:block, Expr(:meta, :inline), :(p = ForwardDiff.partials(x)), t)
end

struct DualCall{F}
  f::F
end
@generated function (dc::DualCall)(x::T, y::Vararg{T,P}) where {P,T}
  quote
    $(Expr(:meta, :inline))
    @inbounds d = Base.Cartesian.@ncall $P ForwardDiff.Dual x p -> y[p]
    _flatten(dc.f(d))
  end
end
struct DualDualCall{I,F}
  f::F
end

@generated function (dc::DualDualCall{I})(x::Vararg{Any,P}) where {P,I}
  # I is the inner size
  # O is the outer
  # P = (I + 1) * (O + 1)
  O = (P ÷ (I + 1)) - 1
  quote
    $(Expr(:meta, :inline))
    @inbounds d = Base.Cartesian.@ncall $O ForwardDiff.Dual o ->
      Base.Cartesian.@ncall $I ForwardDiff.Dual i -> x[i+o*$I-$I]
    _flatten(dc.f(d))
  end
end


dualeval!(
  ::typeof(identity),
  ::AbstractArray{D},
) where {T,P,R,D<:ForwardDiff.Dual{<:Any,<:ForwardDiff.Dual{<:Any,T,R},P}} = nothing
dualeval!(
  ::typeof(identity),
  ::AbstractArray{D},
) where {T<:Base.HWReal,P,D<:ForwardDiff.Dual{<:Any,T,P}} = nothing
dualeval!(
  ::typeof(identity),
  ::AbstractVector{D},
) where {T,P,R,D<:ForwardDiff.Dual{<:Any,<:ForwardDiff.Dual{<:Any,T,R},P}} = nothing
dualeval!(
  ::typeof(identity),
  ::AbstractVector{D},
) where {T<:Base.HWReal,P,D<:ForwardDiff.Dual{<:Any,T,P}} = nothing


function dualeval!(f::F, Cdual::AbstractArray) where {F}
  dualeval!(f, vec(Cdual))
end
const MAX_NUM_LV_EXTRACT = isdefined(LoopVectorization, :EXTRACTFUNS) ? length(LoopVectorization.EXTRACTFUNS) : 14
@generated function dualeval!(
  f::F,
  Cdual::AbstractVector{D},
) where {F,T<:Base.HWReal,P,D<:ForwardDiff.Dual{<:Any,T,P}}
  if (P+1) <= MAX_NUM_LV_EXTRACT
    quote
      C = reinterpret(reshape, T, Cdual)
      g = DualCall(f)
      @turbo for m ∈ eachindex(Cdual)
        (Base.Cartesian.@ntuple $(P + 1) p -> C[p, m]) =
          Base.Cartesian.@ncall $(P + 1) g p -> C[p, m]
      end
    end
  else
    quote
      @inbounds for i ∈ eachindex(Cdual)
        Cdual[i] = f(Cdual[i])
      end
    end
  end
end
@generated function dualeval!(
  f::F,
  Cdual::AbstractVector{D},
) where {F,T,P,R,D<:ForwardDiff.Dual{<:Any,<:ForwardDiff.Dual{<:Any,T,R},P}}
  TD = (P + 1) * (R + 1)
  if isa(T, Base.HWReal) && TD <= MAX_NUM_LV_EXTRACT
    quote
      C = reinterpret(reshape, T, Cdual)
      g = DualDualCall{$R}(f)
      @turbo for m ∈ eachindex(Cdual)
        (Base.Cartesian.@ntuple $TD p -> C[p, m]) = Base.Cartesian.@ncall $TD g p -> C[p, m]
      end
    end
  else
    quote
      @inbounds for i ∈ eachindex(Cdual)
        Cdual[i] = f(Cdual[i])
      end
    end
  end
end

@inline reinterpret_dual(A::AbstractArray{ForwardDiff.Dual{T,V,N}}) where {T,V,N} =
  reinterpret(V, A)
@inline function reinterpret_reshape_dual(
  A::AbstractArray{ForwardDiff.Dual{T,V,N}},
) where {T,V,N}
  reinterpret(reshape, V, A)
end
function contract_loops(
  DC::Vector{Int},
  DA::Vector{Int},
  DB::Vector{Int},
  lao::Bool,
  update::Bool,
)
  contract_dims = Int[]
  Cref = Expr(:ref, :C)
  Cinds = Symbol[]
  for i in eachindex(DC)
    dimsym = Symbol(:i_, i - 1)
    push!(Cinds, dimsym)
    push!(Cref.args, dimsym)
  end
  Aref = Expr(:ref, :A)
  Rinds = Symbol[]
  for d in DA
    i = findfirst(==(d), DC)
    if i === nothing
      push!(contract_dims, d)
      dimsym = Symbol(:r_, length(Rinds))
      push!(Rinds, dimsym)
    else
      dimsym = Cinds[i]
    end
    push!(Aref.args, dimsym)
  end
  Bref = Expr(:ref, :B)
  for d in DB
    i = findfirst(==(d), DC)
    if i === nothing
      r = findfirst(==(d), contract_dims)
      @assert r !== nothing "contract dims must occur in both `A` and `B`"
      dimsym = Rinds[r]
    else
      @assert d ∉ DA "non contracting elementwise mul not currently supported"
      dimsym = Cinds[i]
    end
    push!(Bref.args, dimsym)
  end
  q = :(Caccum += $Aref * $Bref)
  for (i, d) in enumerate(contract_dims)
    da = findfirst(==(d), DA)
    db = findfirst(==(d), DB)
    range = lao ? :(axes(B, $db)) : :(indices((A, B), ($da, $db)))
    q = :(
      for $(Rinds[i]) in $range
        $q
      end
    )
  end
  Cinit = update ? Cref : :(zero(eltype(C)))
  q = if lao
    Arefo = copy(Aref)
    Arefo.args[end] = :K
    quote
      Caccum = $Cinit
      $q
      $Cref = Caccum + $Arefo
    end
  else
    quote
      Caccum = $Cinit
      $q
      $Cref = Caccum
    end
  end
  for (i, d) in enumerate(DC)
    a = findfirst(==(d), DA)
    b = findfirst(==(d), DB)
    range = if a === nothing
      @assert b !== nothing "should not be possible"
      :(indices((B, C), ($b, $i)))
    else
      @assert b === nothing "should not be possible"
      :(indices((A, C), ($a, $i)))
    end
    q = :(
      for $(Cinds[i]) in $range
        $q
      end
    )
  end
  q = :(@turbo $q)
  if lao
    cd::Int = findfirst(==(first(contract_dims)), DA)
    Expr(:block, :(K = lastindex(A, StaticInt{$cd}())), q)
  else
    q
  end
end

@generated function contract!(
  C::PtrArray{<:Any,<:Any,TC},
  A::PtrArray{<:Any,<:Any,TA},
  B::PtrArray{<:Any,<:Any,TB},
  ::Val{DC},
  ::Val{DA},
  ::Val{DB},
  ::Val{LAO},
  ::Val{U},
) where {TC<:NativeTypes,TA<:NativeTypes,TB<:NativeTypes,DC,DA,DB,LAO,U}
  dc = Vector{Int}(undef, length(DC))
  for i in eachindex(DC)
    dc[i] = DC[i]
  end
  da = Vector{Int}(undef, length(DA))
  for i in eachindex(DA)
    da[i] = DA[i]
  end
  db = Vector{Int}(undef, length(DB))
  for i in eachindex(DB)
    db[i] = DB[i]
  end
  contract_loops(dc, da, db, LAO, U)
end
@generated function contract!(
  C::PtrArray{<:Any,DDC,TC},
  A::PtrArray{<:Any,DDA,TA},
  B::PtrArray{<:Any,<:Any,TB},
  ::Val{DC},
  ::Val{DA},
  ::Val{DB},
  ::Val{LAO},
  ::Val{U},
) where {
  T,
  P,
  TC<:ForwardDiff.Dual{T,<:Any,P},
  TA<:ForwardDiff.Dual{T,<:Any,P},
  TB,
  DDC,
  DDA,
  DC,
  DA,
  DB,
  LAO,
  U,
}
  if (DDC[1] & DDA[1]) & (DC[1] == DA[1]) & Bool(is_column_major(A)) & Bool(is_column_major(C))
    r = reinterpret_dual
    DCN = DC
    DAN = DA
  else
    r = reinterpret_reshape_dual
    dimC::Int = Int(length(DC))
    new_dim = ((Int(length(DA))::Int + Int(length(DB))::Int - dimC) >>> 1) + dimC
    DCN = (new_dim, DC...)
    DAN = (new_dim, DA...)
  end
  quote
    contract!($r(C), $r(A), B, Val{$DCN}(), Val{$DAN}(), Val{$DB}(), Val{$LAO}(), Val{$U}())
  end
end
@generated function contract!(
  C::PtrArray{<:Any,DDC,TC},
  A::PtrArray{<:Any,<:Any,TA},
  B::PtrArray{<:Any,DDB,TB},
  ::Val{DC},
  ::Val{DA},
  ::Val{DB},
  ::Val{LAO},
  ::Val{U},
) where {
  T,
  P,
  TC<:ForwardDiff.Dual{T,<:Any,P},
  TA,
  TB<:ForwardDiff.Dual{T,<:Any,P},
  DDC,
  DDB,
  DC,
  DA,
  DB,
  LAO,
  U,
}

  r = reinterpret_reshape_dual
  dimC::Int = Int(length(DC))
  new_dim = ((Int(length(DA))::Int + Int(length(DB))::Int - dimC) >>> 1) + dimC
  DCN = (new_dim, DC...)
  DBN = (new_dim, DB...)
  quote
    contract!($r(C), A, $r(B), Val{$DCN}(), Val{$DA}(), Val{$DBN}(), Val{$LAO}(), Val{$U}())
  end
end


function view_d1_first(A::AbstractArray{<:Any,N}) where {N}
  view(A, firstindex(A, static(1)), ntuple(_ -> (:), Val(N - 1))...)
end
_increment_first(r::CloseOpen) = CloseOpen(r.lower + static(1), r.upper)
_increment_first(r::AbstractUnitRange) = first(r)+static(1):last(r)

function view_d1_notfirst(A::AbstractArray{<:Any,N}) where {N}
  r = _increment_first(axes(A, static(1)))
  view(A, r, ntuple(_ -> (:), Val(N - 1))...)
end
_decrement_last(r::CloseOpen) = CloseOpen(r.lower, r.upper - static(1))
_decrement_last(r::AbstractUnitRange) = CloseOpen(first(r), last(r))
function view_dlast_front(A::AbstractArray{<:Any,N}) where {N}
  r = _decrement_last(axes(A, static(N)))
  view(A, ntuple(_ -> (:), Val(N - 1))..., r)
end

@generated function contract!(
  C::PtrArray{<:Any,DDC,TC},
  A::PtrArray{<:Any,DDA,TA},
  B::PtrArray{<:Any,DDB,TB},
  ::Val{DC},
  ::Val{DA},
  ::Val{DB},
  ::Val{LAO},
  ::Val{U},
) where {
  T,
  P,
  TC<:ForwardDiff.Dual{T,<:Any,P},
  TA<:ForwardDiff.Dual{T,<:Any,P},
  TB<:ForwardDiff.Dual{T,<:Any,P},
  DDC,
  DDA,
  DDB,
  DC,
  DA,
  DB,
  LAO,
  U,
}

  rr = reinterpret_reshape_dual
  q = quote
    rB = $rr(B)
  end
  dimC::Int = Int(length(DC))
  new_dim = ((Int(length(DA))::Int + Int(length(DB))::Int - dimC) >>> 1) + dimC
  let
    if (DDC[1] & DDA[1]) & (DC[1] == DA[1])
      r1 = reinterpret_dual
      DCN = DC
      DAN = DA
    else
      r1 = reinterpret_reshape_dual
      DCN = (new_dim, DC...)
      DAN = (new_dim, DA...)
    end
    push!(
      q.args,
      :(contract!(
        $r1(C),
        $r1(A),
        view_d1_first(rB),
        Val{$DCN}(),
        Val{$DAN}(),
        Val{$DB}(),
        Val{$LAO}(),
        Val{$U}(),
      )),
    )
  end
  let
    DCN = (new_dim, DC...)
    DBN = (new_dim, DB...)
    if LAO
      Aexpr = :($rr(view_dlast_front(A)))
    else
      Aexpr = :($rr(A))
    end
    push!(
      q.args,
      :(
        contract!(
          view_d1_notfirst($rr(C)),
          view_d1_first($Aexpr),
          view_d1_notfirst(rB),
          Val{$DCN}(),
          Val{$DA}(),
          Val{$DBN}(),
          Val{false}(),
          Val{true}(),
        ),
      ),
    )
  end
  q
end

using StrideArraysCore: PtrVector, PtrMatrix
function matmul!(
  C::PtrVector{<:Any,<:Any,D},
  A::PtrMatrix,
  B::PtrVector,
  ::True,
) where {D<:ForwardDiff.Dual}
  contract!(C, A, B, Val{(0,)}(), Val{(0, 1)}(), Val{(1,)}(), Val{true}(), Val{false}())
end
function matmul!(
  C::PtrMatrix{<:Any,<:Any,D},
  A::PtrMatrix,
  B::PtrMatrix,
  ::True,
) where {D<:ForwardDiff.Dual}
  contract!(C, A, B, Val{(0, 1)}(), Val{(0, 2)}(), Val{(2, 1)}(), Val{true}(), Val{false}())
end
function matmul!(
  C::PtrVector{<:Any,<:Any,D},
  A::PtrMatrix{},
  B::PtrVector,
  ::False,
) where {D<:ForwardDiff.Dual}
  contract!(C, A, B, Val{(0,)}(), Val{(0, 1)}(), Val{(1,)}(), Val{false}(), Val{false}())
end
function matmul!(
  C::PtrMatrix{<:Any,<:Any,D},
  A::PtrMatrix,
  B::PtrMatrix,
  ::False,
) where {D<:ForwardDiff.Dual}
  contract!(
    C,
    A,
    B,
    Val{(0, 1)}(),
    Val{(0, 2)}(),
    Val{(2, 1)}(),
    Val{false}(),
    Val{false}(),
  )
end

function matmul!(
  C::PtrMatrix{<:Any,<:Any,D},
  A::PtrMatrix,
  B::PtrVector,
  bias::StaticBool,
) where {D<:ForwardDiff.Dual}
  matmul!(vec(C), A, B, bias)
end
function matmul!(
  C::PtrVector{<:Any,<:Any,D},
  A::PtrMatrix,
  B::PtrMatrix,
  bias::StaticBool,
) where {D<:ForwardDiff.Dual}
  matmul!(C, A, vec(B), bias)
end

function matmul!(C, A, B, bias::StaticBool)
  GC.@preserve C A B matmul!(PtrArray(C), PtrArray(A), PtrArray(B), bias)
end

function dense!(
  f::F,
  Cdual::PtrArray{<:Any,<:Any,D},
  A::AbstractMatrix,
  B::PtrArray,
  ::BT,
  ::FF,
) where {F,BT<:StaticBool,FF,T,P,D<:ForwardDiff.Dual{<:Any,T,P}}
  matmul!(Cdual, A, B, BT())
  dualeval!(f, Cdual)
end
