using ITensors
using ITensorsGrad
using Zygote
using LinearAlgebra

#T(β) = [ exp(β) exp(-β)
#        exp(-β)  exp(β)]

T(β) = [β ^ 2 β ^ 2
        β ^ 2 β ^ 2]

N = 2
Z(β) = tr(T(β)^N)

β = 3.5

@show β

function Zit(β)
  i = Index(2, "i")
  A = itensor(T(β), i', dag(i))

  #return A[1, 1]

  #return (A * δ(dag(i'), i))[]

  #return (A * A)[]

  #return (A * dag(A))[]
  
  return (mapprime(A, 1 => 0, 0 => 2) * A' * A)[]

  # Broken
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

