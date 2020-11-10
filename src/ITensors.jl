
# To get around Zygote keyword argument issue
function dag(T::ITensor)
  TT = conj(tensor(T))
  return itensor(store(TT), dag(inds(T)))
end

const NamedTupleITensor{TensorStorageT, IndexSetT} =
  NamedTuple{(:store, :inds), Tuple{TensorStorageT, IndexSetT}}

# TODO: these are all weird definitions
(A::NamedTupleITensor{<:TensorStorage, Nothing} + B::ITensor) =
  itensor(A.store, inds(B)) + B

(::NamedTupleITensor{Nothing} + B::ITensor) = B

(A::TensorStorage + B::ITensor) = itensor(A, inds(B)) + B

(A::ITensor + B::TensorStorage) = A + itensor(B, inds(A))

# TODO: are these needed?
# TODO: define this as (A::Base.RefValue + B::ITensor) = A[] + B
# and then define conversion of NamedTuple to ITensor
(A::Base.RefValue + B::ITensor) = A[] + B

# TODO: are these needed?
(A::ITensor + B::Base.RefValue) = B + A

# TODO: shows up in f(β) = (Aᵦ = A(β); tr(product(Aᵦ, Aᵦ)))
# Consider supporting in ITensor directly
itensor(A::Diagonal, is) = itensor(Matrix(A), is)

ITensor(st::Combiner, is) = itensor(st, is)

