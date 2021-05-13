using Documenter
using QuadraticAssignmentProblem

makedocs(
	sitename="QuadraticAssignmentProblem.jl",
	pages = [
		"index.md",
		"solvers.md",
		"generators.md",
		"util.md",
	],
)

deploydocs(
	repo = "github.com/flixpar/QuadraticAssignmentProblem.jl.git",
	devbranch = "main",
)
