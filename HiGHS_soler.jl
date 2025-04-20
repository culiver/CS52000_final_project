using Pkg
Pkg.activate(".")  # Activate current folder as the environment

using MatrixDepot, SparseArrays, JuMP, HiGHS

# -------------------------------------------------------------------
# Data Structures and Conversion Function (as specified)
# -------------------------------------------------------------------

mutable struct IplpProblem
  c::Vector{Float64}           # Objective vector for the original problem
  A::SparseMatrixCSC{Float64}  # Constraint matrix for the original problem
  b::Vector{Float64}           # Right-hand side (b) for the original problem
  lo::Vector{Float64}          # Lower bounds for the original variables
  hi::Vector{Float64}          # Upper bounds for the original variables
end

function convert_matrixdepot(P::MatrixDepot.MatrixDescriptor)
  # Convert the MatrixDepot descriptor into IplpProblem format.
  return IplpProblem(vec(P.c), P.A, vec(P.b), vec(P.lo), vec(P.hi))
end

# -------------------------------------------------------------------
# HiGHS-based LP Solver Using JuMP
# -------------------------------------------------------------------

function solve_lp_highs(prob::IplpProblem)
    model = Model(HiGHS.Optimizer)  # Use HiGHS as the solver
    set_optimizer_attribute(model, "solver", "simplex")  # set HiGHS to use simplex
    
    n = length(prob.c)
    
    # Define variables with bounds using the converted problem data.
    @variable(model, prob.lo[i] <= x[i=1:n] <= prob.hi[i])
    
    # Add equality constraints: A*x = b
    for i in 1:size(prob.A, 1)
        @constraint(model, sum(prob.A[i, j] * x[j] for j in 1:n) == prob.b[i])
    end
    
    # Objective: Minimize c'x
    @objective(model, Min, sum(prob.c[j] * x[j] for j in 1:n))
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        x_opt = value.(x)
        obj_val = objective_value(model)
        println("  Optimal solution found. Objective value: ", obj_val)
        return x_opt, obj_val
    else
        println("  Solver failed or did not converge.")
        return nothing
    end
end

# -------------------------------------------------------------------
# Test All LPnetlib Matrices Using HiGHS (now wrapped in a function)
# -------------------------------------------------------------------

function run_tests()
    # List of LPnetlib matrix names (as given in the project description)
    matrix_names = ["afiro", "brandy", "fit1d", "adlittle", "agg", "ganges", "stocfor1", "25fv47", "chemcom"]

    for name in matrix_names
        # For matrices starting with a digit (like "25fv47"), use "LPnetlib/<n>"
        # Otherwise, prepend "lp_" (e.g. "afiro" becomes "LPnetlib/lp_afiro")
        matrix_id = (name == "chemcom") ? "LPnetlib/lpi_" * name : "LPnetlib/lp_" * name
        
        println("-----------------------------------------------------")
        println("Processing matrix: ", matrix_id)
        
        md = nothing  # Declare md outside the try-catch block
        try
            md = mdopen(matrix_id)
        catch e
            println("  ERROR loading ", matrix_id, ": ", e)
            continue
        end

        # If for some reason md is still nothing, skip this matrix.
        if md === nothing
            println("  md is undefined for ", matrix_id)
            continue
        end

        println("  Matrix size: ", size(md.A))
        
        # Convert the MatrixDepot descriptor to our problem format.
        prob = convert_matrixdepot(md)
        
        # Solve the LP using HiGHS (simplex).
        result = solve_lp_highs(prob)
        if result !== nothing
            x, obj = result
            println("  Finished ", matrix_id, " with objective: ", obj)
        end
    end
end

# Uncomment the line below to run the tests
# run_tests()