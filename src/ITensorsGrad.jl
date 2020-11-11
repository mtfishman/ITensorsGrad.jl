module ITensorsGrad

#using BackwardsLinalg
using ChainRules
using ChainRulesCore
using ITensors
using LinearAlgebra
using NDTensors
using Reexport
using ZygoteRules # This is needed for adjoint (ITensor priming)

import Base: +, adjoint, convert, similar
import ChainRulesCore: rrule
import ITensors: itensor, ITensor, dag, prime, setinds

include("ITensors.jl")
include("chainrules/ITensors.jl")
include("chainrules/LinearAlgebra.jl")

end
