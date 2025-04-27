include("iplp_solver.jl")

# =============================================================================
# Test the Setup Using a MatrixDepot LPnetlib Problem
# =============================================================================
feasible_matrix_names = ["afiro", "brandy", "fit1d", "adlittle", "agg", "ganges", "stocfor1", "25fv47"]
infeasible_matrix_names = ["chemcom", "woodw", "bgdbg1", "bgetam", "bgindy", "bgprtr", "box1"]
matrix_names = vcat(feasible_matrix_names, infeasible_matrix_names)

for name in matrix_names
  # For matrices starting with a digit (like "25fv47"), use "LPnetlib/<n>"
  # Otherwise, prepend "lp_" (e.g. "afiro" becomes "LPnetlib/lp_afiro")
  matrix_id = (name in infeasible_matrix_names) ? "LPnetlib/lpi_" * name : "LPnetlib/lp_" * name
  
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

  # Convert the downloaded matrix descriptor to our problem format.
  problem = convert_matrixdepot(md)

  # Set the tolerance for convergence.
  tol = 1e-8

  # Call the interior-point LP solver.
  solution_iplp = iplp(problem, tol; maxit=100)

  if solution_iplp.flag
    println("Interior-point solver converged!")

    # Import the HiGHS solver function without running tests
    include("HiGHS_soler.jl")
    # Solve the problem using HiGHS
    solution_HiGHS = solve_lp_highs(problem)
    x_HiGHS, obj_HiGHS = solution_HiGHS

    # Calculate objective value for iplp (make sure to use original problem objective)
    obj_iplp = dot(problem.c, solution_iplp.x)

    # Print solution comparison
    println("\n========== SOLUTION COMPARISON ==========")
    println("IPLP objective value: ", obj_iplp)
    println("HiGHS objective value: ", obj_HiGHS)
    println("Objective difference (iplp - HiGHS): ", obj_iplp - obj_HiGHS)
    println("Relative objective difference: ", abs(obj_iplp - obj_HiGHS) / max(abs(obj_HiGHS), 1e-10))

    # Compare solution vectors
    x_diff = solution_iplp.x - x_HiGHS
    x_diff_norm = norm(x_diff)
    println("\nSolution difference (L2 norm): ", x_diff_norm)
    println("Relative solution difference: ", x_diff_norm / max(norm(x_HiGHS), 1e-10))

    # Check constraint satisfaction
    constr_viol_iplp = norm(problem.A * solution_iplp.x - problem.b)
    constr_viol_HiGHS = norm(problem.A * x_HiGHS - problem.b)
    println("\nConstraint violation for IPLP: ", constr_viol_iplp)
    println("Constraint violation for HiGHS: ", constr_viol_HiGHS)

    # Print the 5 largest differences in solution components
    if length(x_diff) > 0
        abs_diff = abs.(x_diff)
        sorted_indices = sortperm(abs_diff, rev=true)
        
        println("\nTop 5 largest differences in solution components:")
        for i in 1:min(5, length(sorted_indices))
            idx = sorted_indices[i]
            println("  x[$idx]: IPLP = $(solution_iplp.x[idx]), HiGHS = $(x_HiGHS[idx]), diff = $(x_diff[idx])")
        end
    end
  else
    println("Interior-point solver did not converge.")
  end

  
end
