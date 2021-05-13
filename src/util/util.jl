OPTIMIZER = nothing

"""
Set optimizer for all quadratic and linear programming subproblems.
"""
function qap_set_optimizer!(o)
	global OPTIMIZER = o
end

function get_optimizer()
	if isnothing(OPTIMIZER)
		error("Need to call set_optimizer! before using this solver.")
	else
		return OPTIMIZER
	end
end
