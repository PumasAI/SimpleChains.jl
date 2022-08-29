
_type_sym(x) = __type_sym(x)
@generated __type_sym(::T) where {T} = QuoteNode(Symbol(T))

function task_local_memory(sc)::Vector{UInt8}
  (
    get!(task_local_storage(), _type_sym(sc)) do
      UInt8[]
    end
  )::Vector{UInt8}
end

@inline function with_stack_memory(f::F, ::StaticInt{N}, sc, args::Vararg{Any,K}) where {F,N,K}
  stack_memory = Ref{NTuple{N,UInt8}}()
  p = Base.unsafe_convert(Ptr{UInt8}, stack_memory)
  ret = GC.@preserve stack_memory f(
    sc,
    align(p),
    args...,
  )
  VectorizationBase.lifetime_end!(p, Val{N}())
  return ret
end
function get_heap_memory(sc, num_bytes)
  heap_memory = task_local_memory(sc)
  length(heap_memory) >= num_bytes || resize!(empty!(heap_memory), Int(num_bytes))
  return heap_memory
end
function with_heap_memory(f::F, sc, num_bytes, args::Vararg{Any,K}) where {F,K}
  heap_memory = get_heap_memory(sc, num_bytes)
  GC.@preserve heap_memory f(
    sc,
    align(Base.unsafe_convert(Ptr{UInt8}, heap_memory)),
    args...,
  ),
  heap_memory
end
_static_max_16384(::StaticInt{N}) where {N} = StaticInt{N}()
_static_max_16384(_) = StaticInt{16384}()
@inline function with_memory(f::F, sc, num_bytes, args::Vararg{Any,K}) where {F,K}
  if num_bytes <= 16384
    with_stack_memory(f, _static_max_16384(num_bytes), sc, args...)
  else
    first(with_heap_memory(f, sc, num_bytes, args...))
  end
end

function required_bytes(::Val{T}, layers, sx, additional = static(0)) where {T}
  output_size(Val(T), layers, sx) + additional + static(63)
end
function required_bytes(
  ::Val{T},
  layers,
  sx,
  additional,
  additional_per_thread,
  nthread,
) where {T}
  base_mem_per_thread = output_size(Val(T), layers, sx) + additional_per_thread
  base_mem_per_thread, additional + base_mem_per_thread * nthread + static(63)
end
function required_forward_bytes(::Val{T}, layers, sx, additional = static(0)) where {T}
  forward_output_size(Val(T), layers, sx) + additional + static(63)
end


