
import Zygote: @adjoint, @nograd

@adjoint Tensor{ElT,
                N,
                StoreT,
                IndsT}(inds, store) where {ElT,
                                           N,
                                           StoreT,
                                           IndsT} = 
  Tensor{ElT, N, StoreT, IndsT}(inds, store), 
  t̄ -> (nothing, t̄.store)

@adjoint ITensor{N}(inds, store) where {N} = 
  ITensor{N}(inds, store), 
  t̄ -> (nothing, t̄.store)

@adjoint function Dense{ElT, VecT}(data) where {ElT, VecT}
  Dense{ElT, VecT}(data), s̄ -> (s̄, )
end

@adjoint function Base.:*(T1::ITensor, T2::ITensor)
  T1 * T2, t̄ -> (t̄ * T2, T1 * t̄)
end

function Base.:*(A::Base.RefValue{Any}, B::ITensor)
  return B
end

function Base.:*(A::ITensor, B::Base.RefValue{Any})
  return A
end

@adjoint function Base.:+(T1::ITensor, T2::ITensor)
  T1 + T2, t̄ -> (t̄, t̄)
end

function Base.:+(A::Base.RefValue{Any}, B::ITensor)
  return B
end

function Base.:+(A::ITensor, B::Base.RefValue{Any})
  return A
end

@adjoint function Base.copyto!(T::ITensor,
                               bc::Broadcast.Broadcasted)
  error("Adjoint for copyto!(T::ITensor, bc::Broadcast.Broadcasted) not implemented yet")
end

@adjoint function Base.map!(f::Function,
                            R::ITensor{N},
                            Ts::ITensor{N}...) where {N}
  error("Adjoint for map!(f::Function, R::ITensor, Ts::ITensor...) not implemented yet")
end

@nograd Index

@nograd IndexSet

