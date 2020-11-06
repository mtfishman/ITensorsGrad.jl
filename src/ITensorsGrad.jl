module ITensorsGrad

using ChainRulesCore
using ITensors
using NDTensors
using LinearAlgebra

# Use for truncated SVD
#using BackwardsLinalg

# Use for testing
#using FiniteDifferences

using ITensors: setinds

import Base: +, adjoint
import ChainRulesCore: rrule
import ITensors: itensor, dag, prime

include("ITensors.jl")
include("chainrules.jl")

end
