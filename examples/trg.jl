using ITensors
using ITensorsGrad
using Zygote

examples_src_dir = joinpath(ITensors.examples_dir(), "src")
include(joinpath(examples_src_dir, "trg.jl"))
include(joinpath(examples_src_dir, "2d_classical_ising.jl"))

# Compute the partition function per site κᴺ = Z
# using TRG as a function of inverse temperature β
# and field h
function κ(β, h; χmax = 5, nsteps = 12)
  d = 2
  s = Index(d)
  sₕ = addtags(s, "horiz")
  sᵥ = addtags(s, "vert")
  Tᵦ = ising_mpo(sₕ, sᵥ, β, h)
  κᵦ, T = trg(Tᵦ; χmax = χmax, nsteps = nsteps)
  return κᵦ
end

β = 1.1 * βc
h = 0.0

κᵦ = κ(β, h)

κₑₓ = exp(-β * ising_free_energy(β))
@show κᵦ, κₑₓ
@show abs(κᵦ - κₑₓ)

∂ₕκᵦ = gradient(h -> κ(β, h), h)[1]

@show ∂ₕκᵦ

mᵦ = -∂ₕκᵦ / β

mₑₓ = ising_magnetization(β)
@show mᵦ, mₑₓ

