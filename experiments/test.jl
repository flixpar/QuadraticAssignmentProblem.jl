using QuadraticAssignmentProblem

N = 8
alg = qap_ns

@show N
@show alg

A, B, optperm = generate_qap(N)

time = @elapsed P, perm = alg(A, B, :max)

@show time
@show perm
@show optperm

@show qap_objective(A, B, perm)
@show qap_objective(A, B, optperm)
