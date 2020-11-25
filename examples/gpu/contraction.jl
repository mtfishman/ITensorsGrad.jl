using CUDA
using ITensors
using ITensorsGrad
using ITensorsGPU
using Zygote
using Random: seed!

CUDA.allowscalar(true)

seed!(1234)

gpu = cu
#gpu = identity

i = Index(2, "i")
A = randomITensor(i', i) |> gpu

function f(A)
  return (A * A)[]
end

@show f(A)
dA = @show f'(A)

