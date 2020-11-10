module ITensorsGrad

#using BackwardsLinalg
using ChainRulesCore
using ITensors
using LinearAlgebra
using NDTensors
using Reexport
using ZygoteRules # This is needed for adjoint (ITensor priming)

import Base: +, adjoint
import ChainRulesCore: rrule
import ITensors: itensor, ITensor, dag, prime, setinds

include("ITensors.jl")
include("zygoterules/ITensors.jl")
include("chainrules/ITensors.jl")
include("chainrules/LinearAlgebra.jl")

end
