var documenterSearchIndex = {"docs":
[{"location":"solvers/#Solvers","page":"Solvers","title":"Solvers","text":"","category":"section"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"qap_exact\nqap_linearization\nqap_lprounding\nqap_random\nqap_mms\nqap_ns\nfaq","category":"page"},{"location":"solvers/#QuadraticAssignmentProblem.qap_exact","page":"Solvers","title":"QuadraticAssignmentProblem.qap_exact","text":"Solve the QAP using quadratic integer progamming.\n\n\n\n\n\n","category":"function"},{"location":"solvers/#QuadraticAssignmentProblem.qap_linearization","page":"Solvers","title":"QuadraticAssignmentProblem.qap_linearization","text":"Approximately solve the QAP by linearizing the problem and solving the resulting linear integer program.\n\n\n\n\n\n","category":"function"},{"location":"solvers/#QuadraticAssignmentProblem.qap_lprounding","page":"Solvers","title":"QuadraticAssignmentProblem.qap_lprounding","text":"Approximately solve the QAP by linearizing and relaxing the problem and solving the resulting LIP with linear programming and rounding via the Hungarian algorithm.\n\n\n\n\n\n","category":"function"},{"location":"solvers/#QuadraticAssignmentProblem.qap_random","page":"Solvers","title":"QuadraticAssignmentProblem.qap_random","text":"Approximately solve the QAP through random sampling.\n\n\n\n\n\n","category":"function"},{"location":"solvers/#QuadraticAssignmentProblem.qap_mms","page":"Solvers","title":"QuadraticAssignmentProblem.qap_mms","text":"Run the algorithm from \"Maximum Quadratic Assignment Problem: Reduction from Maximum Label Cover and LP-based Approximation Algorithm\" by Makarychev, Manokaran, and Sviridenko (MMS) to approximately solve the QAP.\n\n\n\n\n\n","category":"function"},{"location":"solvers/#QuadraticAssignmentProblem.qap_ns","page":"Solvers","title":"QuadraticAssignmentProblem.qap_ns","text":"Approximate the maxQAP using the complete algorithm from \"On the Maximum Quadratic Assignment Problem\" by Nagarajan and Sviridenko (NS).\n\n\n\n\n\nApproximate the 0-1 QAP using the complete algorithm from \"On the Maximum Quadratic Assignment Problem\" by Nagarajan and Sviridenko (NS).\n\n\n\n\n\n","category":"function"},{"location":"solvers/#QuadraticAssignmentProblem.faq","page":"Solvers","title":"QuadraticAssignmentProblem.faq","text":"Perform the FAQ algorithm from \"Fast Approximate Quadratic Programming for Graph Matching\" to approximately solve the QAP.\n\n\n\n\n\n","category":"function"},{"location":"generators/#Instance-Generators","page":"Instance Generators","title":"Instance Generators","text":"","category":"section"},{"location":"generators/","page":"Instance Generators","title":"Instance Generators","text":"uniform_matrix\nmetric_matrix\nzeroone_matrix\ngenerate_qap","category":"page"},{"location":"generators/#QuadraticAssignmentProblem.uniform_matrix","page":"Instance Generators","title":"QuadraticAssignmentProblem.uniform_matrix","text":"Generate an N×N matrix with uniformly random entries in [0,1].\n\n\n\n\n\n","category":"function"},{"location":"generators/#QuadraticAssignmentProblem.metric_matrix","page":"Instance Generators","title":"QuadraticAssignmentProblem.metric_matrix","text":"Generate a random N×N matrix with entries satisfying the triangle inequality.\n\n\n\n\n\n","category":"function"},{"location":"generators/#QuadraticAssignmentProblem.zeroone_matrix","page":"Instance Generators","title":"QuadraticAssignmentProblem.zeroone_matrix","text":"Generate a random 0-1 N×N matrix with edge probability p.\n\n\n\n\n\n","category":"function"},{"location":"generators/#QuadraticAssignmentProblem.generate_qap","page":"Instance Generators","title":"QuadraticAssignmentProblem.generate_qap","text":"Generate a pair of N×N matricies with known optimal permutation using Algorithm 4 from \"Generating Quadratic Assignment Test Problems with Known Optimal Permutations\" by Li and Pardalos.\n\n\n\n\n\n","category":"function"},{"location":"#QuadraticAssignmentProblem.jl","page":"QuadraticAssignmentProblem.jl","title":"QuadraticAssignmentProblem.jl","text":"","category":"section"},{"location":"","page":"QuadraticAssignmentProblem.jl","title":"QuadraticAssignmentProblem.jl","text":"Pages = [\"index.md\", \"solvers.md\", \"generators.md\", \"util.md\"]","category":"page"},{"location":"","page":"QuadraticAssignmentProblem.jl","title":"QuadraticAssignmentProblem.jl","text":"Algorithms for (approximately) solving the quadratic assignment problem (QAP) implemented in Julia.","category":"page"},{"location":"","page":"QuadraticAssignmentProblem.jl","title":"QuadraticAssignmentProblem.jl","text":"Exact algorithms:","category":"page"},{"location":"","page":"QuadraticAssignmentProblem.jl","title":"QuadraticAssignmentProblem.jl","text":"Quadratic integer programming\nLinearization","category":"page"},{"location":"","page":"QuadraticAssignmentProblem.jl","title":"QuadraticAssignmentProblem.jl","text":"Approximation algorithms:","category":"page"},{"location":"","page":"QuadraticAssignmentProblem.jl","title":"QuadraticAssignmentProblem.jl","text":"\"On the Maximum Quadratic Assignment Problem\" by Nagarajan and Sviridenko (NS)\n\"Maximum Quadratic Assignment Problem: Reduction from Maximum Label Cover and LP-based Approximation Algorithm\" by Makarychev, Manokaran, and Sviridenko (MMS)","category":"page"},{"location":"","page":"QuadraticAssignmentProblem.jl","title":"QuadraticAssignmentProblem.jl","text":"Heuristic algorithms:","category":"page"},{"location":"","page":"QuadraticAssignmentProblem.jl","title":"QuadraticAssignmentProblem.jl","text":"Fast Approximate QAP\nRandom search\nLP + LAP rounding","category":"page"},{"location":"util/#Utility-Functions","page":"Utility Functions","title":"Utility Functions","text":"","category":"section"},{"location":"util/","page":"Utility Functions","title":"Utility Functions","text":"qap_objective\nqap_set_optimizer!","category":"page"},{"location":"util/#QuadraticAssignmentProblem.qap_objective","page":"Utility Functions","title":"QuadraticAssignmentProblem.qap_objective","text":"Compute the QAP objective function value.\n\n\n\n\n\n","category":"function"},{"location":"util/#QuadraticAssignmentProblem.qap_set_optimizer!","page":"Utility Functions","title":"QuadraticAssignmentProblem.qap_set_optimizer!","text":"Set optimizer for all quadratic and linear programming subproblems.\n\n\n\n\n\n","category":"function"}]
}