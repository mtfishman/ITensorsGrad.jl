using ITensors
using ITensorsGrad
using ITensorsGPU
using OptimKit
using Random
using Zygote

using ITensors: data

gpu = cu

Random.seed!(1234)

N = 4
s = siteinds("S=1/2", N)

h = -1.0

a = AutoMPO()
for b in 1:N-1
  a .+= -1, "Sz", b, "Sz", b+1
end
for n in 1:N
  a .+= h, "Sx", n
end
H = gpu(MPO(a, s))

χmax = 4
ψ₀ = gpu(randomMPS(s, χmax))

# Versions that are just vectors
H̃ = data(H)
ψ̃₀ = data(ψ₀)

function E(H, ψ)
  N = length(ψ)
  ψdag = prime.(dag.(ψ))
  e = ψ[1] * H[1] * ψdag[1]
  for n in 2:N
    e = e * ψ[n] * H[n] * ψdag[n]
  end
  ψdag = noprime.(ψdag, "Site")
  norm = ψ[1] * ψdag[1]
  for n in 2:N
    norm = norm * ψ[n] * ψdag[n]
  end
  return e[] / norm[]
end
E(ψ) = E(H̃, ψ)

algorithm = GradientDescent(maxiter = 100,
                            gradtol = 1e-3,
                            verbosity = 2)
ψ̃, _ = optimize(ψ -> (E(ψ), E'(ψ)), ψ̃₀, algorithm)

@show E(ψ̃)

sweeps = Sweeps(5)
maxdim!(sweeps, χmax)
_E, ψ = dmrg(H, ψ₀, sweeps)

@show E(ψ)

