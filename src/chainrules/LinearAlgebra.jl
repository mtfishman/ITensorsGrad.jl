
#struct ZeroAdder end
#Base.:+(a, zero::ZeroAdder) = a
#Base.:+(zero::ZeroAdder, a) = a
#Base.:-(a, zero::ZeroAdder) = a
#Base.:-(zero::ZeroAdder, a) = -a
#Base.:-(zero::ZeroAdder) = zero
#
#mpow2(a::AbstractArray) = a .^ 2
#
#
#function svd_back(U::Tensor, S::Tensor, V::Tensor,
#                  dU, dS, dV; kwargs...)
#  dUₐ = dU isa Zero ? Zero() : array(dU)
#  dSₐ = dS isa Zero ? Zero() : array(dS)
#  dVₐ = dV isa Zero ? Zero() : array(dV)
#  return svd_back(array(U), array(S), array(V),
#                  dUₐ, dSₐ, dVₐ, kwargs...)
#end
#
#"""
#    svd_back(U, S, V, dU, dS, dV)
#
#Adjoint for SVD decomposition.
#
#References:
#    https://j-towns.github.io/papers/svd-derivative.pdf
#    https://giggleliu.github.io/2019/04/02/einsumbp.html
#"""
#function svd_back(U::AbstractArray, S::AbstractArray{T}, V,
#                  dU, dS, dV; η::Real = 1e-40) where T
#  @show typeof(dU)
#  @show typeof(dS)
#  @show typeof(dV)
#
#  all(x -> x isa Zero, (dU, dS, dV)) && return Zero()
#  η = T(η)
#  NS = length(S)
#  S2 = mpow2(S)
#  Sinv = @. S/(S2+η)
#  F = S2' .- S2
#  F ./= (mpow2(F) .+ η)
#
#  res = ZeroAdder()
#  if !(dU isa Nothing)
#    UdU = U'*dU
#    J = F.*(UdU)
#    res += (J+J')*LinearAlgebra.Diagonal(S) + LinearAlgebra.Diagonal(1im*imag(LinearAlgebra.diag(UdU)) .* Sinv)
#  end
#  if !(dV isa Zero)
#    VdV = V'*dV
#    K = F.*(VdV)
#    res += LinearAlgebra.Diagonal(S) * (K+K')
#  end
#  if !(dS isa Zero)
#    res += LinearAlgebra.Diagonal(dS)
#  end
#
#  res = U*res*V'
#
#  if !(dU isa Zero) && size(U, 1) != size(U, 2)
#    res += (dU - U* (U'*dU)) * LinearAlgebra.Diagonal(Sinv) * V'
#  end
#
#  if !(dV isa Zero) && size(V, 1) != size(V, 2)
#    res = res + U * LinearAlgebra.Diagonal(Sinv) * (dV' - (dV'*V)*V')
#  end
#
#  @show res
#
#  return res
#end

#"""
#		qr(A) -> Tuple{AbstractMatrix, AbstractMatrix}
#private QR method, call LinearAlgebra.qr to achieve forward calculation, while
#return Tuple type
#"""
#function qr(A)
#	res = LinearAlgebra.qr(A)
#	Matrix(res.Q), res.R
#end
#
#"""
#    qr(A, pivot) -> Tuple{AbstractMatrix, AbstractMatrix, AbstractVector}
#"""
#function qr(A::AbstractMatrix, pivot::Val{true})
#  res = LinearAlgebra.qr(A, pivot)
#  Q, R, P = Matrix(res.Q), res.R, res.P
#end
#
#"""
#    lq(A) -> Tuple{AbstractMatrix, AbstractMatrix}
#"""
#function lq(A)
#	res = LinearAlgebra.lq(A)
#  res.L, Matrix(res.Q)
#end

#trtrs!(c1::Char, c2::Char, c3::Char, r::AbstractMatrix, b::AbstractVecOrMat) = LinearAlgebra.LAPACK.trtrs!(c1, c2, c3, r, b)
#
#"""
#    copyltu!(A::AbstractMatrix) -> AbstractMatrix
#copy the lower triangular to upper triangular.
#"""
#function copyltu!(A::AbstractMatrix)
#  m, n = size(A)
#  for i=1:m
#    A[i,i] = real(A[i,i])
#    for j=i+1:n
#      @inbounds A[i,j] = conj(A[j,i])
#    end
#  end
#  A
#end
#
#"""
#    qr_back_fullrank(q, r, dq, dr) -> Matrix
#backward for QR decomposition, for input matrix (in forward pass) with M > N.
#References:
#    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
#"""
#@generated function qr_back_fullrank(q, r, dq, dr)
#  dqnot0 = !(dq <: Nothing)
#  drnot0 = !(dr <: Nothing)
#  (!dqnot0 && !drnot0) && return :(nothing)
#  ex = drnot0 && dqnot0 ? :(r*dr' - dq'*q) : (dqnot0 ? :(-dq'*q) : :(r*dr'))
#  :(b = $(dqnot0 ? :(dq) : :(ZeroAdder())) + q*copyltu!($ex);
#  trtrs!('U', 'N', 'N', r, do_adjoint(b))')
#end
#
#do_adjoint(A::Matrix) = Matrix(A')
#
#"""
#    qr_back(A, q, r, dq, dr) -> Matrix
#backward for QR decomposition, for an arbituary shaped input matrix.
#References:
#    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
#    Differentiable Programming Tensor Networks, Hai-Jun Liao, Jin-Guo Liu, Lei Wang, Tao Xiang
#"""
#@generated function qr_back(A, q, r, dq, dr)
#  dqnot0 = !(dq <: Nothing)
#  drnot0 = !(dr <: Nothing)
#  (!dqnot0 && !drnot0) && return :(nothing)
#  ex = quote
#    size(r, 1) == size(r, 2) && return qr_back_fullrank(q, r, dq ,dr)
#    M, N = size(r)
#    B = view(A,:,M+1:N)
#    U = view(r,:,1:M)
#    D = view(r,:,M+1:N)
#    $(if drnot0
#      :(dD = view(dr,:,M+1:N);
#      da = qr_back_fullrank(q, U, $(dqnot0 ? :(dq+B*dD') : :(B*dD')), view(dr,:,1:M));
#      db = q*dD)
#    else
#      :(da = qr_back_fullrank(q, U, dq, nothing);
#      db = zero(B))
#    end)
#    hcat(da, db)
#  end
#end
#
#"""
#    lq_back(A, l, q, dl, dq) -> Matrix
#backward for LQ decomposition, for an arbituary shaped input matrix.
#References:
#    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
#    Differentiable Programming Tensor Networks, Hai-Jun Liao, Jin-Guo Liu, Lei Wang, Tao Xiang
#"""
#function lq_back_fullrank(L, Q, dL, dQ)
#  M = ZeroAdder()
#  dL === nothing || (M += L'*dL)
#  dQ === nothing || (M -= dQ*Q')
#  C = copyltu!(M)*Q
#  if dQ !== nothing
#    C += dQ
#  end
#  #inv(L)' * C
#  trtrs!('L', 'C', 'N', L, C)
#end
#
#"""
#    lq_back(A, L, Q, dL, dQ) -> Matrix
#backward for QR decomposition, for an arbituary shaped input matrix.
#References:
#    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
#    HaiJun's paper.
#"""
#@generated function lq_back(A, L, Q, dL, dQ)
#  dunot0 = !(dQ <: Nothing)
#  dlnot0 = !(dL <: Nothing)
#  (!dunot0 && !dlnot0) && return :(nothing)
#  ex = quote
#    N, M = size(L)
#    M == N && return lq_back_fullrank(L, Q, dL, dQ)
#    B = view(A,M+1:N,:)
#    U = view(L,1:M,:)
#    D = view(L,M+1:N,:)
#    $(if dlnot0
#      :(dD = view(dL,M+1:N,:);
#      da = lq_back_fullrank(U, Q, view(dL,1:M,:), $(dunot0 ? :(dQ+dD'*B) : :(dD'*B)));
#      db = dD*Q)
#    else
#      :(da = lq_back_fullrank(U, Q, nothing, dQ);
#      db = zero(B))
#    end)
#    vcat(da, db)
#  end
#end
#
#function symeigen(A::AbstractMatrix)
#    E, U = LinearAlgebra.eigen(A)
#    E, Matrix(U)
#end
#
#"""
#References:
#    * Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
#"""
#function symeigen_back(E::AbstractVector{T}, U, dE, dU; η=1e-40) where T
#  all(x->x isa Nothing, (dU, dE)) && return nothing
#  η = T(η)
#  if dU === nothing
#    D = LinearAlgebra.Diagonal(dE)
#  else
#    F = E .- E'
#    F .= F./(F.^2 .+ η)
#    dUU = dU' * U .* F
#    D = (dUU + dUU')/2
#    if dE !== nothing
#      D = D + LinearAlgebra.Diagonal(dE)
#    end
#  end
#  U * D * U'
#end

