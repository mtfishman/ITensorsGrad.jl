
const NamedTupleITensor{TensorStorageT, IndexSetT} =
  NamedTuple{(:store, :inds), Tuple{TensorStorageT, IndexSetT}}

function similar(D::Diag, ::Type{T}) where {T}
  return Diag(similar(data(D), T))
end

function convert(::Type{Diagonal}, T::Tensor{<:Any, 2, <:Diag})
  return Diagonal(data(T))
end

# TODO: shows up in f(β) = (Aᵦ = A(β); tr(product(Aᵦ, Aᵦ)))
# Consider supporting in ITensor directly
itensor(A::Diagonal, is) = itensor(Matrix(A), is)

ITensor(st::Combiner, is) = itensor(st, is)

function similar(T::Tensor{<:Any}, ::Type{ITensor})
  T̃ = similar(T)
  return itensor(T̃)
end

(a::Float64 + b::ITensor) = ITensor(a) + b

#function real(T::ITensor)
#  dataᵣ = real(store(T))
#  if store(T) isa Dense
#    Tᵣ = itensor(Dense(dataᵣ), inds(T))
#  else
#    error("real(::ITensor) only implemented for Dense storage right now")
#  end
#  return Tᵣ
#end

#function similar(T::Tensor{<:Any, N, <:Dense},
#                 ::Type{ITensor{N}}) where {N}
#  T̃ = similar(T)
#  return itensor(T̃)
#end

## function convert(::Type{Diagonal}, T::ITensor{2})
##   return convert(Diagonal, tensor(T))
## end
## 
## function convert(::Type{Array}, T::ITensor)
##   return convert(Array, tensor(T))
## end
## 
## function convert(::Type{Tensor}, A::Array)
##   return Tensor(A, size(A))
## end
## 
## # To get around Zygote keyword argument issue
## function dag(T::ITensor)
##   TT = conj(tensor(T))
##   return itensor(store(TT), dag(inds(T)))
## end

## # TODO: these are all weird definitions
## (A::NamedTupleITensor{<:TensorStorage, Nothing} + B::ITensor) =
##   itensor(A.store, inds(B)) + B
## 
## (::NamedTupleITensor{Nothing} + B::ITensor) = B
## 
## (A::TensorStorage + B::ITensor) = itensor(A, inds(B)) + B
## 
## (A::ITensor + B::TensorStorage) = A + itensor(B, inds(A))
## 
## # TODO: are these needed?
## # TODO: define this as (A::Base.RefValue + B::ITensor) = A[] + B
## # and then define conversion of NamedTuple to ITensor
## (A::Base.RefValue + B::ITensor) = A[] + B
## 
## # TODO: are these needed?
## (A::ITensor + B::Base.RefValue) = B + A

