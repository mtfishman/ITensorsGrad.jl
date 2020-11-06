module ITensorsGrad

using BackwardsLinalg
using ChainRulesCore
using ITensors
using LinearAlgebra
using NDTensors
using ZygoteRules # This is needed for adjoint (ITensor priming)

import Base: +, adjoint
import ChainRulesCore: rrule
import ITensors: itensor, ITensor, dag, prime, setinds

include("ITensors.jl")
include("chainrules.jl")

end
