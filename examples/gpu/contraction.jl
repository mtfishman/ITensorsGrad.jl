using CUDA
using ITensors
using ITensorsGrad
using ITensorsGPU
using Zygote
using Random: seed!

CUDA.allowscalar(true)

seed!(1234)

device = cu
#device = identity

i = Index(2, "i")
A = device(randomITensor(i', i))

function f(A)
  return (A * A)[]
end

@show f(A)
dA = @show f'(A)
