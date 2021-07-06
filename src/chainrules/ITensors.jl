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
    return (NO_FIELDS, ntuple(_ -> NoTangent(), Val(N))...)
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
    return (NO_FIELDS, NoTangent(), store(ΔΩ))
  end
  return tensor(st, is), Tensor_pullback
end

function rrule(T::Type{<:Tensor}, as::NDTensors.AllowAlias, st::TensorStorage, is)
  function Tensor_pullback(ΔΩ)
    return (NO_FIELDS, NoTangent(), storage(ΔΩ), NoTangent())
  end
  return Tensor(as, st, is), Tensor_pullback
end

function rrule(::typeof(tensor), T::ITensor)
  indsT = inds(T)
  function tensor_pullback(ΔΩ)
    return NO_FIELDS, itensor(ΔΩ.storage, indsT)
  end
  return tensor(T), tensor_pullback
end

function rrule(::Type{ITensor}, as::ITensors.AliasStyle, T::Tensor)
  indsT = inds(T)
  function ITensor_pullback(ΔΩ)
    return (NO_FIELDS, setinds(Tensor(as, ΔΩ), indsT))
  end
  return ITensor(as, T), ITensor_pullback
end

function rrule(::Type{ITensor}, T::Tensor)
  indsT = inds(T)
  function ITensor_pullback(ΔΩ)
    return (NO_FIELDS, setinds(Tensor(ΔΩ), indsT))
  end
  return ITensor(T), ITensor_pullback
end

function rrule(::Type{ITensor}, as::ITensors.AliasStyle, st::TensorStorage, is)
  function ITensor_pullback(ΔΩ)
    return (NO_FIELDS, NoTangent(), storage(ΔΩ), NoTangent())
  end
  return ITensor(as, st, is), ITensor_pullback
end

function rrule(::Type{ITensor}, args...)
  @show args
  error("In rrule for ITensor, not implemented yet.")
  function ITensor_pullback(ΔΩ)
    return (NO_FIELDS, NoTangent(), storage(ΔΩ), NoTangent())
  end
  return ITensor(as, st, is), ITensor_pullback
end
#
# ITensors rrules
#

# TODO XXX: this isn't being called by Zygote for some reason
# Probably related to: https://github.com/FluxML/Zygote.jl/issues/811
#function rrule(::typeof(permutedims), T::Tensor, perm; kwargs...)
#  function permutedims_pullback(ΔΩ)
#    return (NO_FIELDS, permutedims(ΔΩ, invperm(perm)), NoTangent())
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
#    return (NO_FIELDS, ΔT, NoTangent(), NoTangent())
#  end
#  return setproperty!(T, field, val), setproperty!_pullback
#end

# For order-0 ITensor
function rrule(::typeof(getindex), T::ITensor)
  indsT = inds(T)

  function getindex_pullback(ΔΩ::ITensor)
    return NO_FIELDS, setinds(ΔΩ, indsT)
  end

  getindex_pullback(ΔΩ::Number) =
    getindex_pullback(ITensor(ΔΩ, indsT))

  return T[], getindex_pullback
end

#dense(T::TensorT) where {TensorT<:DiagTensor}

function rrule(::typeof(dense), T::DiagTensor)
  indsT = inds(T)

  function dense_pullback(ΔΩ::Tensor)
    ΔT = Tensor(Diag(diag(array(ΔΩ))), indsT)
    return NO_FIELDS, ΔT
  end

  dense_pullback(ΔΩ::TensorStorage) =
    dense_pullback(tensor(ΔΩ, indsT))

  dense_pullback(ΔΩ::Tangent) =
    dense_pullback(ΔΩ.store)

  return dense(T), dense_pullback
end

# XXX TODO: need to implement block sparse version
function rrule(::typeof(svd), X::Tensor{<:Real, 2}; kwargs...)
  U, S, V, spec = svd(X; kwargs...)
  function svd_pullback(ΔΩ)
    # `getproperty` on `Tangent`s ensures we have no thunks.
    Uₐ = array(U)
    Sₐ = convert(Diagonal, S)
    # Need this to account for Julia SVD convention
    Vtₐ = Matrix(array(V)')
    F = SVD(Uₐ, parent(Sₐ), Vtₐ)
    ΔU = array(tensor(ΔΩ[1].store, size(Uₐ)))
    # XXX TODO: should this be just a vector?
    #ΔS = ΔΩ[2] isa Zero ? Zero() : array(tensor(ΔΩ[2].store, size(Sₐ)))
    ΔS = ΔΩ[2] isa Zero ? Zero() : data(ΔΩ[2].store)
    # Need this to account for Julia SVD convention
    ΔVt = Matrix(array(tensor(ΔΩ[3].store, size(Vtₐ)))')

    #@show size(F.U)
    #@show size(F.S)
    #@show size(F.Vt)
    #@show size(ΔU)
    #@show size(ΔS)
    #@show size(ΔVt)

    ∂X = ChainRules.svd_rev(F, ΔU, ΔS, ΔVt)
    ∂T = tensor(Dense(vec(∂X)), inds(X))
    return (NO_FIELDS, ∂T)
  end
  return (U, S, V, spec), svd_pullback
end

#@adjoint function LinearAlgebra.eigen(A::LinearAlgebra.RealHermSymComplexHerm)
#  dU = eigen(A)
function _eigen_pullback(dU::Eigen, Δd, ΔU)
  d, U = dU
  if ΔU === nothing
    P = Diagonal(Δd)
  else
    F = inv.(d' .- d)
    P = F .* (U' * ΔU)

    @show typeof(P)
    @show size(P)
    @show typeof(Δd)
    @show size(Δd)

    if Δd === nothing
      P[diagind(P)] .= 0
    else
      P[diagind(P)] = Δd
    end
  end
  return U * P * U'
end


#function rrule(T::Type{<:Hermitian}, T::Tensor{<:Any, 2})
#    Ω = T(A, uplo)
#    function Hermitian_pullback(ΔΩ)
#        return (NO_FIELDS, _symherm_back(T, ΔΩ, Ω.uplo), NoTangent())
#    end
#    return Ω, HermOrSym_pullback
#end

# XXX TODO: need to implement block sparse version
@adjoint function eigen(T::Union{<:TensorT, Hermitian{ElT, <:TensorT}};
                        kwargs...) where {ElT <: Real, TensorT <: Tensor{ElT, 2}}
  indsT = inds(T)
  D, U, spec = eigen(T; kwargs...)
  function eigen_pullback(ΔΩ)
    @show typeof(ΔΩ)
    #error("In eigen_pullback")
    Dₐ = convert(Diagonal, D)
    Uₐ = array(U)
    F = Eigen(parent(Dₐ), Uₐ)
    #ΔD = ΔΩ[2] isa Zero ? Zero() : array(tensor(ΔΩ[1].store, size(Dₐ)))
    ΔD = ΔΩ[2] isa Zero ? Zero() : data(ΔΩ[1].store)
    ΔU = array(tensor(ΔΩ[2].store, size(Uₐ)))
    ∂X = _eigen_pullback(F, ΔD, ΔU)
    ∂T = tensor(Dense(vec(∂X)), indsT)
    return (NO_FIELDS, ∂T)
  end
  return (D, U, spec), eigen_pullback
end

function rrule(::Type{<:TagSet}, args::Vararg{<:Any, N}) where {N}
  function TagSet_pullback(::Any)
    return (NO_FIELDS, ntuple(_ -> NoTangent(), Val(N))...)
  end
  return TagSet(args...), TagSet_pullback
end

function rrule(::Type{<:Index}, args::Vararg{<:Any, N}) where {N}
  function Index_pullback(::Any)
    return (NO_FIELDS, ntuple(_ -> NoTangent(), Val(N))...)
  end
  return Index(args...), Index_pullback
end

function rrule(::typeof(addtags), ts::TagSet, tsadd)
  function addtags_pullback(::Any)
    return (NO_FIELDS, (NoTangent(), NoTangent()))
  end
  return addtags(ts, tsadd), addtags_pullback
end

function rrule(::Type{<:IndexSet}, args::Vararg{<:Any, N}) where {N}
  function IndexSet_pullback(::Any)
    return (NO_FIELDS, ntuple(_ -> NoTangent(), Val(N))...)
  end
  return IndexSet(args...), IndexSet_pullback
end

function rrule(::typeof(unioninds), args::Vararg{<:Any, N}) where {N}
  function unioninds_pullback(::Any)
    return (NO_FIELDS, ntuple(_ -> NoTangent(), Val(N))...)
  end
  return unioninds(args...), unioninds_pullback
end

function rrule(::typeof(commoninds), args::Vararg{<:Any, N};
               kwargs...) where {N}
  function commoninds_pullback(::Any)
    return (NO_FIELDS, ntuple(_ -> NoTangent(), Val(N))...)
  end
  return commoninds(args...; kwargs...), commoninds_pullback
end

function rrule(::typeof(uniqueinds), args::Vararg{<:Any, N}) where {N}
  function uniqueinds_pullback(::Any)
    return (NO_FIELDS, ntuple(_ -> NoTangent(), Val(N))...)
  end
  return uniqueinds(args...), uniqueinds_pullback
end

function rrule(::typeof(replaceinds), is::IndexSet,
               args::Vararg{<:Any, N}) where {N}
  function replaceinds_pullback(::Any)
    return (NO_FIELDS, NoTangent(),
            ntuple(_ -> NoTangent(), Val(N))...)
  end
  return replaceinds(is, args...), replaceinds_pullback
end

function rrule(::typeof(setinds!), ::ITensor, args...)
  error("Differentiating `setinds!` is currently not available, use out-of-place versions (like `setinds`, `settags`, etc.) instead.")
end

# XXX: this isn't being called
# Maybe related to https://github.com/FluxML/Zygote.jl/issues/811
#function rrule(::typeof(tr), ::ITensor)
#  error("Differentiating `tr(::ITensor)` is currently not supported, use contractions with δ instead.")
#end

@adjoint function tr(A::ITensor; kwargs...)
  error("Differentiating `tr(::ITensor)` is currently not supported, use contractions with δ instead.")
end

function rrule(::typeof(replaceind),
               T::ITensor, args::Vararg{<:Any, N}) where {N}
  indsT = inds(T)

  function replaceind_pullback(ΔΩ)
    ΔT = setinds(ΔΩ, indsT)
    return (NO_FIELDS, ΔT, ntuple(_ -> NoTangent(), Val(N))...)
  end

  replaceind_pullback(ΔΩ::Base.RefValue) =
    replaceind_pullback(ΔΩ[])

  function replaceind_pullback(ΔΩ::Union{<:Tangent,
                                         <:NamedTupleITensor})
    ΔT = itensor(ΔΩ.store, indsT)
    return (NO_FIELDS, ΔT, ntuple(_ -> NoTangent(), Val(N))...)
  end

  return replaceind(T, args...), replaceind_pullback
end

function rrule(::typeof(setinds),
               T::ITensor, is)
  indsT = inds(T)

  function setinds_pullback(ΔΩ)
    ΔT = setinds(ΔΩ, indsT)
    return (NO_FIELDS, ΔT, NoTangent())
  end

  setinds_pullback(ΔΩ::Base.RefValue) =
    setinds_pullback(ΔΩ[])

  function setinds_pullback(ΔΩ::Union{<:Tangent,
                                      <:NamedTupleITensor})
    ΔT = itensor(ΔΩ.store, indsT)
    return (NO_FIELDS, ΔT, NoTangent())
  end

  return setinds(T, is), setinds_pullback
end

function rrule(::Type{<:ITensor},
               is::IndexSet, st::TensorStorage)

  function ITensor_pullback(ΔΩ::TensorStorage)
    return (NO_FIELDS, NoTangent(), ΔΩ)
  end

  # TODO: maybe return Dense(ΔΩ)?
  function ITensor_pullback(ΔΩ::AbstractArray)
    return (NO_FIELDS, NoTangent(), ΔΩ)
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
            ntuple(_ -> NoTangent(), Val(N))...)
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

function rrule(::typeof(itensor), args...)
  @show args
  error("rrule for itensor, not implemented yet")
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

  times_pullback(ΔΩ::Base.RefValue) = times_pullback(ΔΩ[])

  function times_pullback(ΔΩ::Tangent)
    @show ΔΩ
    return times_pullback(itensor(ΔΩ.store, indsΔΩ))
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

