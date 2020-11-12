#
# NDTensors rrules
#

function rrule(::Type{<:Dense}, v::AbstractVector)

  function Dense_pullback(ΔΩ)
    return (NO_FIELDS, data(ΔΩ))
  end

  return Dense(v), Dense_pullback
end

function rrule(::Type{<:Combiner}, args::Vararg{<:Any, N}) where {N}
  function Combiner_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return Combiner(args...), Combiner_pullback
end
 
function rrule(::Type{<:Diag}, v)

  function Diag_pullback(ΔΩ)
    return (NO_FIELDS, data(ΔΩ))
  end

  #function Diag_pullback(ΔΩ::ITensor)
  #  return (NO_FIELDS, data(store(ΔΩ)))
  #end

  return Diag(v), Diag_pullback
end

# XXX Not sure about this definition
function rrule(::Type{<:Tensor},
               is::IndexSet, st::TensorStorage)

  function Tensor_pullback(ΔΩ)
    return (NO_FIELDS, DoesNotExist(), store(ΔΩ))
  end

  return tensor(st, is), Tensor_pullback
end

#
# ITensors rrules
#

# TODO XXX: this isn't being called by Zygote for some reason
#function rrule(::typeof(permutedims), T::Tensor, perm; kwargs...)
#  function permutedims_pullback(ΔΩ)
#    return (NO_FIELDS, permutedims(ΔΩ, invperm(perm)), DoesNotExist())
#  end
#  return permutedims(T, perm; kwargs...), permutedims_pullback
#end

@adjoint function permutedims(T::Tensor, perm; kwargs...)
  indsT = inds(T)

  function permutedims_pullback(ΔΩ)
    return (permutedims(ΔΩ, invperm(perm)), nothing)
  end

  function permutedims_pullback(ΔΩ::NamedTupleITensor)
    return permutedims_pullback(tensor(ΔΩ.store, indsT))
  end

  return permutedims(T, perm; kwargs...), permutedims_pullback
end

# XXX T.inds = ... is not working right now, need to
# figure out why. Maybe need to overload this?
#function rrule(::typeof(setproperty!), T::ITensor, field, val)
#  indsT = inds(T)
#  @show inds(T)
#  @show field
#  @show val
#  function setproperty!_pullback(ΔΩ)
#    @show ΔΩ
#    #error("setproperty!_pullback")
#    if field == :inds
#      setproperty!(ΔΩ, field, indsT)
#    else
#      error("No setproperty!_pullback for field $field")
#    end
#    return (NO_FIELDS, ΔT, DoesNotExist(), DoesNotExist())
#  end
#  return setproperty!(T, field, val), setproperty!_pullback
#end

# XXX TODO: need to implement block sparse version
function rrule(::typeof(svd), X::Tensor{<:Real, 2})
  U, S, V, spec = svd(X)
  function svd_pullback(Ȳ)
    # `getproperty` on `Composite`s ensures we have no thunks.
    Uₐ = array(U)
    Sₐ = convert(Diagonal, S)
    # Need this to account for Julia SVD convention
    Vₐ = Matrix(array(V)')
    F = SVD(Uₐ, parent(Sₐ), Vₐ)
    ΔU = array(tensor(Ȳ[1].store, size(Uₐ)))
    ΔS = Ȳ[2] isa Zero ? Zero() : array(tensor(Ȳ[2].store, size(Sₐ)))
    # Need this to account for Julia SVD convention
    ΔV = Matrix(array(tensor(Ȳ[3].store, size(Vₐ)))')
    ∂X = ChainRules.svd_rev(F, ΔU, ΔS, ΔV')
    ∂T = tensor(Dense(vec(∂X)), inds(X))
    return (NO_FIELDS, ∂T)
  end
  return (U, S, V, spec), svd_pullback
end

function rrule(::Type{<:TagSet}, args::Vararg{<:Any, N}) where {N}
  function TagSet_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return TagSet(args...), TagSet_pullback
end

function rrule(::Type{<:Index}, args::Vararg{<:Any, N}) where {N}
  function Index_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return Index(args...), Index_pullback
end

function rrule(::Type{<:IndexSet}, args::Vararg{<:Any, N}) where {N}
  function IndexSet_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return IndexSet(args...), IndexSet_pullback
end

function rrule(::typeof(unioninds), args::Vararg{<:Any, N}) where {N}
  function unioninds_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return unioninds(args...), unioninds_pullback
end

function rrule(::typeof(commoninds), args::Vararg{<:Any, N};
               kwargs...) where {N}
  function commoninds_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return commoninds(args...; kwargs...), commoninds_pullback
end

function rrule(::typeof(uniqueinds), args::Vararg{<:Any, N}) where {N}
  function uniqueinds_pullback(::Any)
    return (DoesNotExist(), ntuple(_ -> DoesNotExist(), Val(N))...)
  end
  return uniqueinds(args...), uniqueinds_pullback
end

function rrule(::typeof(setinds!), ::ITensor, args...)
  error("Differentiating `setinds!` is currently not available, use out-of-place versions (like `setinds`, `settags`, etc.) instead.")
end

# XXX: this isn't being called
#function rrule(::typeof(tr), ::ITensor)
#  error("Differentiating `tr(::ITensor)` is currently not supported, use contractions with δ instead.")
#end

@adjoint function tr(A::ITensor; kwargs...)
  error("Differentiating `tr(::ITensor)` is currently not supported, use contractions with δ instead.")
end

function rrule(::typeof(setinds),
               T::ITensor, is)
  indsT = inds(T)

  function setinds_pullback(ΔΩ)
    ΔT = setinds(ΔΩ, indsT)
    return (NO_FIELDS, ΔT, DoesNotExist())
  end

  setinds_pullback(ΔΩ::Base.RefValue) =
    setinds_pullback(ΔΩ[])

  function setinds_pullback(ΔΩ::Union{<:Composite,
                                      <:NamedTupleITensor})
    ΔT = itensor(ΔΩ.store, indsT)
    return (NO_FIELDS, ΔT, DoesNotExist())
  end

  return setinds(T, is), setinds_pullback
end

function rrule(::Type{<:ITensor},
               is::IndexSet, st::TensorStorage)

  function ITensor_pullback(ΔΩ::TensorStorage)
    return (NO_FIELDS, DoesNotExist(), ΔΩ)
  end

  # TODO: maybe return Dense(ΔΩ)?
  function ITensor_pullback(ΔΩ::AbstractArray)
    return (NO_FIELDS, DoesNotExist(), ΔΩ)
  end

  function ITensor_pullback(ΔΩ)
    return ITensor_pullback(ΔΩ.store)
  end

  ITensor_pullback(ΔΩ::Base.RefValue) = ITensor_pullback(ΔΩ[])

  return itensor(st, is), ITensor_pullback
end

function rrule(::typeof(itensor), A::Array,
               i::Vararg{<:Any, N}) where {N}
  is = IndexSet(i...)

  function itensor_pullback(ΔΩ::ITensor)
    return (NO_FIELDS, array(ΔΩ),
            ntuple(_ -> DoesNotExist(), Val(N))...)
  end

  function itensor_pullback(ΔΩ::Union{<:Array, <:TensorStorage})
    return itensor_pullback(itensor(ΔΩ, is))
  end

  function itensor_pullback(ΔΩ::NamedTuple{(:store, :inds), Tuple{T, Nothing}} where {T})
    return itensor_pullback(ΔΩ.store)
  end

  itensor_pullback(ΔΩ::Base.RefValue) =
    itensor_pullback(ΔΩ[])

  return itensor(A, i...), itensor_pullback
end


function _rrule_itensor(::typeof(*), A, B)
  if A isa Number
    indsΔΩ = inds(B)
  elseif B isa Number
    indsΔΩ = inds(A)
  else
    indsΔΩ = noncommoninds(A, B)
  end

  function times_pullback(ΔΩ::ITensor)
    ∂A = ΔΩ * B
    ∂B = A * ΔΩ
    return (NO_FIELDS, ∂A, ∂B)
  end

  times_pullback(ΔΩ::Base.RefValue) =
    times_pullback(itensor(ΔΩ[].store, indsΔΩ))

  times_pullback(ΔΩ::Composite) =
    times_pullback(itensor(ΔΩ.store, indsΔΩ))

  return A * B, times_pullback
end

rrule(::typeof(*), A, B::ITensor) =
  _rrule_itensor(*, A, B)

rrule(::typeof(*), A::ITensor, B) =
  _rrule_itensor(*, A, B)

rrule(::typeof(*), A::ITensor, B::ITensor) =
  _rrule_itensor(*, A, B)

function rrule(::typeof(+), A::ITensor, B::ITensor)
  indsA = inds(A)
  indsB = inds(B)

  function plus_pullback(ΔΩ::ITensor)
    return (NO_FIELDS, setinds(ΔΩ, indsA), setinds(ΔΩ, indsB))
  end

  function plus_pullback(ΔΩ::NamedTupleITensor)
    return (NO_FIELDS, itensor(ΔΩ.store, indsA),
                       itensor(ΔΩ.store, indsB))
  end

  plus_pullback(ΔΩ::Base.RefValue) =
    plus_pullback(ΔΩ[])

  return A + B, plus_pullback
end

# XXX TODO: This is needed for conj(::NamedTuple) error
# Maybe instead define overload of conj(::NamedTuple)
function rrule(::typeof(dag), T::ITensor)
  indsT = inds(T)

  function dag_pullback(ΔΩ::ITensor)
    # XXX TODO: double check this definition
    return (NO_FIELDS, dag(ΔΩ))
  end

  function dag_pullback(ΔΩ::NamedTuple{(:store, :inds),
                                       Tuple{T, Nothing}} where {T <: TensorStorage})
    return dag_pullback(itensor(ΔΩ.store, indsT))
  end

  dag_pullback(ΔΩ::Base.RefValue) = dag_pullback(ΔΩ[])

  return dag(T), dag_pullback
end

# TODO: for some reason this version isn't being used
# by Zygote
#function rrule(::typeof(adjoint), T::ITensor)
#  indsT = inds(T)
#
#  function adjoint_pullback(ΔΩ::ITensor)
#    return (NO_FIELDS, setinds(ΔΩ, indsT))
#  end
#
#  function adjoint_pullback(ΔΩ::NamedTuple{(:store, :inds), Tuple{StoreT, Nothing}} where {StoreT <: TensorStorage})
#    return adjoint_pullback(itensor(ΔΩ.store, indsT))
#  end
#
#  adjoint_pullback(ΔΩ::Base.RefValue) = adjoint_pullback(ΔΩ[])
#
#  return prime(T), adjoint_pullback
#end

# TODO: this is needed to overload the Zygote version
# of adjoint
@adjoint function Base.adjoint(T::ITensor)
  indsT = inds(T)

  function adjoint_pullback(ΔΩ::ITensor)
    return (setinds(ΔΩ, indsT),)
  end

  function adjoint_pullback(ΔΩ::NamedTuple{(:store, :inds), Tuple{StoreT, Nothing}} where {StoreT <: TensorStorage})
    return adjoint_pullback(itensor(ΔΩ.store, indsT))
  end

  adjoint_pullback(ΔΩ::Base.RefValue) = adjoint_pullback(ΔΩ[])

  return prime(T), adjoint_pullback
end

