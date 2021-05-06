module QuadraticAssignmentProblem

include("solvers/ip.jl")
include("solvers/faq.jl")
include("solvers/mms.jl")

include("util/generators.jl")
include("util/graphmatching.jl")
include("util/metrics.jl")

export qap_ip
export faq, sgm
export mms

end
