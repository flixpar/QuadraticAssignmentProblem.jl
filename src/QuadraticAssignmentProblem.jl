module QuadraticAssignmentProblem

include("solvers/exact.jl")
include("solvers/lp.jl")
include("solvers/faq.jl")
include("solvers/mms.jl")
include("solvers/ns.jl")
include("solvers/random.jl")

include("util/generators.jl")
include("util/graphmatching.jl")
include("util/metrics.jl")

export qap_exact
export qap_linearization, qap_lprounding
export faq
export qap_mms
export qap_ns, qap_starpacking, qap_densemapping
export qap_random

export uniform_matrix
export metric_matrix
export zeroone_matrix
export generate_qap

export qap_objective
export gm_objective

end
