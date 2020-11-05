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

#function rrule(::Type{<:IndexSet}, x::Vararg{<:Any, N}) where {N}
#  function IndexSet_pullback(::Any)
#    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
#  end
#  return IndexSet(x...), IndexSet_pullback
#end

function rrule(::Type{<:ITensor},
               is::IndexSet, store::TensorStorage)
  function ITensor_pullback(ȳ)

    println()
    println("In ITensor_pullback")
    @show typeof(ȳ)
    @show ȳ
    println()

    return (NO_FIELDS, ȳ.inds, ȳ.store)
  end

  function ITensor_pullback(ȳ::Base.RefValue)
    return (NO_FIELDS, ȳ[].inds, ȳ[].store)
  end

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

function _rrule_itensor(::typeof(*), A, B)
  function times_pullback(ΔΩ)
    ∂A = ΔΩ * B
    ∂B = A * ΔΩ
    return (NO_FIELDS, ∂A, ∂B)
  end
  function times_pullback(ΔΩ::Base.RefValue)
    return times_pullback(itensor(ΔΩ[].store, ΔΩ[].inds))
  end
  return A * B, times_pullback
end

rrule(::typeof(*), A, B::ITensor) =
  _rrule_itensor(*, A, B)

rrule(::typeof(*), A::ITensor, B) =
  _rrule_itensor(*, A, B)

rrule(::typeof(*), A::ITensor, B::ITensor) =
  _rrule_itensor(*, A, B)

function rrule(::typeof(+), A::ITensor, B::ITensor)

  println()
  println("In rrule(::typeof(+), ::ITensor...)")
  @show A
  @show B
  println()

  function plus_pullback(ΔΩ)
    return (NO_FIELDS, ΔΩ, ΔΩ)
  end
  return A + B, plus_pullback
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
# TODO: define this as (A::Base.RefValue + B::ITensor) = A[] + B
# and then define conversion of NamedTuple to ITensor
function (A::Base.RefValue + B::ITensor)

  @show A[]
  @show B

  return itensor(A[].store, A[].inds) + B
end

# TODO: are these needed?
(A::ITensor + B::Base.RefValue) = B + A

# TODO: is this needed?
adjoint(A::Base.RefValue) = adjoint(A[])

adjoint(A::NamedTuple{(:store, :inds), Tuple{T, Nothing}} where {T <: TensorStorage}) = A.store

# TODO: define this
#+(::ITensor{2}, ::ITensor{0})

