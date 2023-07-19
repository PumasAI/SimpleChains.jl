
_static_max_stack(::StaticInt{N}) where {N} = StaticInt{N}()
_static_max_stack(_) = StaticInt{MAXSTACK}()

_type_sym(x) = __type_sym(remove_loss(getchain(x)))
@generated __type_sym(::T) where {T} = QuoteNode(Symbol(T))

function task_local_memory(sc)::Vector{UInt8}
  (
    get!(task_local_storage(), _type_sym(sc)) do
      UInt8[]
    end
  )::Vector{UInt8}
end
@inline function with_stack_memory(
  f::F,
  ::StaticInt{N},
  sc,
  args::Vararg{Any,K}
) where {F,N,K}
  stack_memory = Ref{NTuple{N,UInt8}}()
  # stack_memory = pointer(NOTSTACKMEM) + (Threads.threadid()-1)*MAXSTACK
  GC.@preserve stack_memory begin
    p = Base.unsafe_convert(Ptr{UInt8}, stack_memory)
    ret = f(sc, align(p), args...)
    VectorizationBase.lifetime_end!(p, Val{N}())
  end
  return ret
end
function get_heap_memory(sc, num_bytes)
  heap_memory = task_local_memory(sc)
  length(heap_memory) < num_bytes &&
    resize!(empty!(heap_memory), Int(num_bytes))
  return heap_memory
end
function with_heap_memory(f::F, sc, num_bytes, args::Vararg{Any,K}) where {F,K}
  heap_memory = get_heap_memory(sc, num_bytes)
  p = align(Base.unsafe_convert(Ptr{UInt8}, heap_memory))
  GC.@preserve heap_memory f(sc, p, args...), heap_memory
end
@inline function with_memory(
  f::F,
  sc,
  num_bytes,
  args::Vararg{Any,K}
) where {F,K}
  if num_bytes <= MAXSTACK
    with_stack_memory(f, _static_max_stack(num_bytes), sc, args...)
  else
    first(with_heap_memory(f, sc, num_bytes, args...))
  end
end

function required_bytes(::Val{T}, layers, sx) where {T}
  required_bytes(Val{T}(), layers, sx, static(0))
end
function required_bytes(::Val{T}, layers, sx, additional) where {T}
  # we add 63 extra bytes to make sure we can bring the pointer to 64 byte alignment.
  output_size(Val(T), layers, sx) + additional + static(63)
end
function required_bytes(
  ::Val{T},
  layers,
  sx,
  additional,
  additional_per_thread,
  nthread
) where {T}
  base_mem_per_thread = output_size(Val(T), layers, sx) + additional_per_thread
  # we add 63 extra bytes to make sure we can bring the pointer to 64 byte alignment.
  base_mem_per_thread, additional + base_mem_per_thread * nthread + static(63)
end
function required_forward_bytes(
  ::Val{T},
  layers,
  sx,
  additional = static(0)
) where {T}
  # we add 63 extra bytes to make sure we can bring the pointer to 64 byte alignment.
  forward_output_size(Val(T), layers, sx) + additional + static(63)
end
