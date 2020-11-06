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

# XXX Not sure about this definition
function rrule(::Type{<:Tensor},
               is::IndexSet, st::TensorStorage)
  function Tensor_pullback(ΔΩ)
    return (NO_FIELDS, DoesNotExist(), itensor(store(ΔΩ), is))
  end
  return tensor(st, is), Tensor_pullback
end

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
    return (NO_FIELDS, DoesNotExist(), setinds(ȳ, is))
  end
  ITensor_pullback(ΔΩ::Base.RefValue) = ITensor_pullback(ΔΩ[])
  return itensor(store, is), ITensor_pullback
end

function rrule(::typeof(itensor), A::Array,
               i::Vararg{<:Any, N}) where {N}
  function itensor_pullback(ΔΩ)
    return (NO_FIELDS, array(ΔΩ),
            ntuple(_ -> DoesNotExist(), Val(N))...)
  end

  itensor_pullback(ΔΩ::NamedTuple{(:store, :inds), Tuple{T, Nothing}} where {T}) = itensor_pullback(ΔΩ.store)

  itensor_pullback(ΔΩ::Base.RefValue) =
    itensor_pullback(ΔΩ[])

  return itensor(A, i...), itensor_pullback
end

function _rrule_itensor(::typeof(*), A, B)
  function times_pullback(ΔΩ)
    ∂A = ΔΩ * B
    ∂B = A * ΔΩ
    return (NO_FIELDS, ∂A, ∂B)
  end
  function times_pullback(ΔΩ::Base.RefValue)
    return times_pullback(itensor(ΔΩ[].store))
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
  function plus_pullback(ΔΩ)
    return (NO_FIELDS, ΔΩ, ΔΩ)
  end
  return A + B, plus_pullback
end

# XXX TODO: This is needed for conj(::NamedTuple) error
# Maybe instead define overload of conj(::NamedTuple)
function rrule(::typeof(dag), T::ITensor)
  function dag_pullback(ȳ::ITensor)
    return (NO_FIELDS, ȳ)
  end
  return dag(T), dag_pullback
end

