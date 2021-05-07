module QuadraticAssignmentProblem

include("solvers/ip.jl")
include("solvers/lp.jl")
include("solvers/faq.jl")
include("solvers/mms.jl")
include("solvers/random.jl")

include("util/generators.jl")
include("util/graphmatching.jl")
include("util/metrics.jl")

export qap_ip, lap_ip
export lap_relaxation
export faq
export mms
export qap_random

export uniform_matrix
export metric_matrix

export qap_objective

end
