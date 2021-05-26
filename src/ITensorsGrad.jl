module ITensorsGrad

#using BackwardsLinalg
using ChainRules
using ChainRulesCore
using ITensors
using ITensors.NDTensors
using LinearAlgebra
using Reexport
using ZygoteRules # This is needed for adjoint (ITensor priming)

using ITensors: setinds!

import Base: +, adjoint, convert, similar, real
import ChainRulesCore: rrule
import ITensors: itensor, ITensor, dag, prime, setinds

include("ITensors.jl")
include("chainrules/ITensors.jl")
include("chainrules/LinearAlgebra.jl")

end
