using ITensors
using ITensorsGrad
using Zygote
using LinearAlgebra

#T(β) = [ exp(β) exp(-β)
#        exp(-β)  exp(β)]

T(β) = [β ^ 2 β ^ 2
        β ^ 2 β ^ 2]

Z(β) = tr(T(β) * T(β))
#Z(β) = (T(β) + 2 * T(β) * T(β))[1, 1]
#Z(β) = (2 * T(β) * T(β))[1, 1]
#Z(β) = tr(2 * T(β) * T(β))
#Z(β) = tr(T(β) * T(β) * T(β))
#Z(β) = (2 * T(β))[1, 1]
#Z(β) = (T(β) * 3)[1, 1]
#Z(β) = (T(β) + 2 * T(β))[1, 1]

β = 3.5

@show β

function Zit(β)
  i = Index(2, "i")
  A = itensor(T(β), i', dag(i))

  #return A[1, 1]

  #return (A + 2 * A)[1, 1]

  #return (A * 3)[1, 1]

  #return (A * δ(dag(i'), i))[]

  #return (A * A)[]

  #return (2 * A * A)[]
 
  #return (A * dag(A))[]
  
  # XXX Broken
  return (swapprime(A, 0 => 1) * A)[]

  #return (mapprime(A', 1 => 2 => 0) * A)[]
  
  # XXX Broke
  #return (mapprime(A'', 3 => 0) * A' * A)[]

  # XXX Broken
  #return (A' * A * δ(dag(i)'', i))[]

  #return tr(A' * A; plev = 0 => 2)

  #return (mapprime(A' * A, 2 => 1) * δ(dag(i)'', i))[]

  #return tr(product(A, A))
end

@show Z(β)
@show Zit(β)
println()

@show Z'(β)
@show Zit'(β)
println()

