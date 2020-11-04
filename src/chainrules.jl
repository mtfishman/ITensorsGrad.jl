#
# NDTensors rrules
#

function rrule(::Type{<:Dense}, data::AbstractVector)
  function Dense_pullback(ȳ)
    return (NO_FIELDS, ȳ)
  end
  return Dense(data), Dense_pullback
end

function rrule(::Type{<:Diag}, data)
  function Diag_pullback(ȳ)
    return (NO_FIELDS, ȳ)
  end
  return Diag(data), Diag_pullback
end

function rrule(::Type{<:Tensor},
               is::IndexSet, store::TensorStorage)
  function Tensor_pullback(ȳ)

    println()
    println("In Tensor_pullback")
    @show typeof(ȳ)
    println()

    #return (NO_FIELDS, DoesNotExist(), ȳ.store)
    return (NO_FIELDS, ȳ.inds, ȳ.store)
  end

  println()
  println("In rrule(::Type{<:Tensor}, args...)")
  @show is
  @show typeof(store)
  @show store
  println()

  return tensor(store, is), Tensor_pullback
end

# TODO: may not be needed
#function rrule(::typeof(mapprime), A, B) 
#  function mapprime_pullback(ȳ)
#    return (NO_FIELDS, ȳ, DoesNotExist())
#  end
#  return mapprime(A, B), mapprime_pullback
#end

#
# ITensors rrules
#

function rrule(::Type{<:Index}, x::Vararg{<:Any, N}) where {N}
  function Index_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return Index(x...), Index_pullback
end

function rrule(::Type{<:IndexSet}, x::Vararg{<:Any, N}) where {N}
  function IndexSet_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return IndexSet(x...), IndexSet_pullback
end

function rrule(::Type{<:ITensor},
               is::IndexSet, store::TensorStorage)
  function ITensor_pullback(ȳ)

    println()
    println("In ITensor_pullback")
    @show typeof(ȳ)
    println()

    return (NO_FIELDS, ȳ.inds, ȳ.store)
  end
  #function ITensor_pullback(ȳ::Base.RefValue)
  #  return (NO_FIELDS, DoesNotExist(), ȳ[].store.store)
  #end
  #function ITensor_pullback(ȳ::ITensor)
  #  return (NO_FIELDS, DoesNotExist(), ȳ.store)
  #end
  #function ITensor_pullback(ȳ::Tensor)
  #  return (NO_FIELDS, DoesNotExist(), ȳ.store)
  #end
  #function ITensor_pullback(ȳ::AbstractArray)
  #  return (NO_FIELDS, DoesNotExist(), ȳ)
  #end
  return itensor(store, is), ITensor_pullback
end

#function rrule(::typeof(itensor), A::Array,
#               i::Vararg{Index, N}) where {N}
#
#  println()
#  println("rrule(::typeof(itensor), args...)")
#  @show A
#  @show i
#  println()
#
#  function itensor_pullback(ΔΩ)
#
#    @show ΔΩ
#
#    return (NO_FIELDS, ΔΩ,
#            ntuple(_ -> DoesNotExist(), Val(N))...)
#  end
#  return itensor(A, i...), itensor_pullback
#end

function rrule(::typeof(*), A::ITensor, B::ITensor)

  println()
  println("In rrule(::typeof(*), ::ITensor...)")
  @show A
  @show B
  println()

  function times_pullback(ΔΩ)

    println()
    println("In times_pullback(ΔΩ)")
    @show typeof(ΔΩ)
    @show ΔΩ
    println()

    ∂A = ΔΩ * B
    ∂B = A * ΔΩ
    return (NO_FIELDS, ∂A, ∂B)
  end

  function times_pullback(ΔΩ::Base.RefValue)

    println()
    println("In times_pullback(ΔΩ)")
    @show ΔΩ
    @show ΔΩ[]
    @show typeof(ΔΩ[].store)
    @show ΔΩ[].store
    @show typeof(ΔΩ[].inds)
    @show ΔΩ[].inds
    println()

    return times_pullback(itensor(ΔΩ[].store, ΔΩ[].inds))
  end

  #function times_pullback(ȳ::Base.RefValue{Any})
  #  if ȳ[].store isa Union{Tensor, AbstractArray}
  #    T̄ = itensor(ȳ[].store)
  #  elseif ȳ[].store isa TensorStorage && ȳ[].inds isa IndexSet
  #    T̄ = itensor(ȳ[].store, ȳ[].inds)
  #  else
  #    error("No times_pullback defined")
  #  end
  #  return (NO_FIELDS, T̄ * T2, T1 * T̄)
  #end

  return A * B, times_pullback
end

# To get around Zygote keyword argument issue
import ITensors: dag
function dag(T::ITensor)
  TT = conj(tensor(T))
  return itensor(store(TT), dag(inds(T)))
end

# XXX TODO: This is needed for conj(::NamedTuple) error
function rrule(::typeof(dag), T::ITensor)
  #function dag_pullback(ȳ::Base.RefValue)
  #  return (NO_FIELDS, conj(ȳ[].store))
  #end
  function dag_pullback(ȳ::ITensor)
    return (NO_FIELDS, ȳ)
  end
  return dag(T), dag_pullback
end

# TODO: are these needed?
#function (A::Base.RefValue{Any} + B::ITensor)
#  return itensor(A[].store, inds(B)) + B
#end

# TODO: are these needed?
#function (A::ITensor + B::Base.RefValue{Any})
#  return A + itensor(B[].store, inds(A))
#end

# TODO: define this
#+(::ITensor{2}, ::ITensor{0})

