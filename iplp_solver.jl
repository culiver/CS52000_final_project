using MatrixDepot, SparseArrays, LinearAlgebra, Printf
using LDLFactorizations

# =============================================================================
# Define Data Structures for the LP Solver Interface
# =============================================================================

mutable struct IplpSolution
  x::Vector{Float64}           # The solution vector (original problem)
  flag::Bool                   # True if the solution converged, false otherwise
  cs::Vector{Float64}          # Objective vector in standard form
  As::SparseMatrixCSC{Float64} # Constraint matrix in standard form
  bs::Vector{Float64}          # Right-hand side (b) in standard form
  xs::Vector{Float64}          # The solution vector in standard form
  lam::Vector{Float64}         # Lagrange multipliers for equality constraints
  s::Vector{Float64}           # Slack variables in standard form
end

mutable struct IplpProblem
  c::Vector{Float64}           # Objective vector for the original problem
  A::SparseMatrixCSC{Float64}  # Constraint matrix for the original problem
  b::Vector{Float64}           # Right-hand side (b) for the original problem
  lo::Vector{Float64}          # Lower bounds for the original variables
  hi::Vector{Float64}          # Upper bounds for the original variables
end

# =============================================================================
# Conversion Function: MatrixDepot → IplpProblem
# =============================================================================

function convert_matrixdepot(P::MatrixDepot.MatrixDescriptor)
  # Converts a MatrixDepot descriptor to the IplpProblem format.
  return IplpProblem(vec(P.c), P.A, vec(P.b), vec(P.lo), vec(P.hi))
end

# =============================================================================
# Interior-Point LP Solver Implementation
# =============================================================================

"""
soln = iplp(Problem, tol; maxit=100) solves the linear program:

   minimize c'*x   subject to   A*x = b  and  lo <= x <= hi

where the input problem is stored in the following fields:

   Problem.A, Problem.c, Problem.b, Problem.lo, Problem.hi

The output is an IplpSolution containing:

  [x, flag, cs, As, bs, xs, lam, s]

Interpretation:
  - x      : the solution for the original problem
  - flag   : true if the solver converged, false otherwise
  - cs     : objective vector in standard form
  - As     : constraint matrix in standard form
  - bs     : right-hand side in standard form
  - xs     : solution vector in standard form (solving: minimize cs'*xs with As*xs = bs, xs >= 0)
  - lam, s : associated Lagrange multipliers in standard form

Convergence is achieved when the duality measure (xs'*s)/n <= tol and the normalized residual
norm([As'*lam + s - cs; As*xs - bs; xs.*s]) / norm([bs; cs]) <= tol,
or if the iteration count exceeds maxit.
"""
function iplp(Problem::IplpProblem, tol; maxit=100)
  println("Starting interior-point method with tolerance: $tol, max iterations: $maxit")
  
  # ---------------------------------------------------------------------------
  # Step 1: Problem Setup and Conversion to Standard Form
  # ---------------------------------------------------------------------------
  
  # Extract original data.
  A = Problem.A
  b_orig = Problem.b
  c_orig = Problem.c
  lo = copy(Problem.lo)
  hi = copy(Problem.hi)
  n = length(c_orig)
  m = size(A, 1)
  
  println("Problem dimensions: $n variables, $m constraints")
  if n == 0 || m == 0
    println("Error: Empty problem detected")
    return IplpSolution(Float64[], false, Float64[], spzeros(0,0), Float64[], Float64[], Float64[], Float64[])
  end
  
  # ---------------------------------------------------------------------------
  # Treat Large Bounds as Infinity (HiGHS convention)
  # Any upper bound ≥ 1e20 will be treated as +Inf.
  # Any lower bound ≤ -1e20 will be treated as -Inf.
  threshold = 1e20
  for i in 1:n
    if hi[i] ≥ threshold
      hi[i] = Inf
    end
    if lo[i] ≤ -threshold
      lo[i] = -Inf
    end
  end
  
  # Determine how many extra columns and extra equality rows we need.
  # The transformations are as follows:
  # 1. Bounded (finite lo and hi): need 2 columns (x' and slack) and 1 extra equality constraint.
  # 2. Lower only: need 1 column.
  # 3. Upper only: need 1 column.
  # 4. Free: need 2 columns (x⁺ and x⁻).
  num_extra_eq = 0
  num_std_vars = 0
  # var_map[i] will be:
  #   (:bounded, col_x, col_slack)
  #   (:lower, col)
  #   (:upper, col)
  #   (:free, col_plus, col_minus)
  var_map = Vector{Any}(undef, n)
  
  for i in 1:n
    if isfinite(lo[i]) && isfinite(hi[i])
      num_std_vars += 2
      num_extra_eq += 1
      var_map[i] = (:bounded, num_std_vars - 1, num_std_vars)
    elseif isfinite(lo[i])
      num_std_vars += 1
      var_map[i] = (:lower, num_std_vars)
    elseif isfinite(hi[i])
      num_std_vars += 1
      var_map[i] = (:upper, num_std_vars)
    else
      num_std_vars += 2
      var_map[i] = (:free, num_std_vars - 1, num_std_vars)
    end
  end
  
  total_eq_rows = m + num_extra_eq  # original m eq. plus one extra eq per bounded variable.
  
  println("Converting to standard form: $num_std_vars standard variables, $total_eq_rows equality constraints")
  
  # Initialize standard form arrays.
  As = spzeros(total_eq_rows, num_std_vars)
  cs = zeros(num_std_vars)
  bs = zeros(total_eq_rows)
  
  # ---------------------------------------------------------------------------
  # Adjust the right-hand side for the original m constraints.
  # For any variable with a finite lower bound, subtract A[:, i] * lo[i]
  # For any variable with only an upper bound, subtract A[:, i] * hi[i]
  b_adj = copy(b_orig)
  for i in 1:n
    if isfinite(lo[i])
      b_adj .-= A[:, i] * lo[i]
    elseif (!isfinite(lo[i])) && isfinite(hi[i])
      b_adj .-= A[:, i] * hi[i]
    end
    # For bounded variables, subtract the lower bound only.
  end
  bs[1:m] = b_adj  # original constraints go into the first m rows.
  
  # ---------------------------------------------------------------------------
  # Fill in the columns corresponding to each variable.
  #
  # We loop over each variable i and insert the appropriate columns into As,
  # set the corresponding entries of cs, and record the transformation.
  #
  # For the original constraints (rows 1:m):
  # - Bounded & lower: multiply by +1.
  # - Upper only: multiply by -1.
  # - Free: split into positive and negative parts.
  #
  # Then, for each bounded variable, add an extra equality row enforcing:
  #   x'_i + slack_i = hi[i] - lo[i].
  extra_eq_row = m  # next available extra equality row index.
  for i in 1:n
    trans = var_map[i][1]
    if trans == :bounded
      # Transformation: x = x' + lo, with 0 <= x' <= hi - lo.
      # In original constraints, we use the column for x'.
      col_x = var_map[i][2]   # column index for the shifted variable x'
      As[1:m, col_x] = A[:, i]
      cs[col_x] = c_orig[i]
      # Add extra equality row for the upper bound.
      extra_eq_row += 1
      As[extra_eq_row, col_x] = 1.0
      # The slack variable column is the next one.
      col_slack = var_map[i][3]
      As[extra_eq_row, col_slack] = 1.0
      bs[extra_eq_row] = hi[i] - lo[i]
      cs[col_slack] = 0.0   # slack variable has zero cost
    elseif trans == :lower
      # Lower bound only: x = x' + lo, so use the column directly.
      col = var_map[i][2]
      As[1:m, col] = A[:, i]
      cs[col] = c_orig[i]
    elseif trans == :upper
      # Upper bound only: x = hi - x', so we use -A.
      col = var_map[i][2]
      As[1:m, col] = -A[:, i]
      cs[col] = -c_orig[i]
    elseif trans == :free
      # Free variable: x = x⁺ - x⁻.
      col_plus = var_map[i][2]
      col_minus = var_map[i][3]
      As[1:m, col_plus] = A[:, i]
      As[1:m, col_minus] = -A[:, i]
      cs[col_plus] = c_orig[i]
      cs[col_minus] = -c_orig[i]
    end
  end
  
  println("Standard form: $(size(As,2)) variables, $(size(As,1)) equality constraints")
  
  # ---------------------------------------------------------------------------
  # Step 2: Robust Initialization of the Interior-Point Method
  # ---------------------------------------------------------------------------

  println("Initializing interior point with robust method")
  n_std = size(As, 2)

  # --- Robust Initialization ---
  # Initialize primal (xs), dual (lam), and slack (s) variables
  xs = ones(n_std)
  lam = zeros(size(As, 1))
  s = ones(n_std)

  # Regularization parameter (epsilon) helps stabilize ill-conditioned systems.
  reg_param = 1e-8
  W_max  = 1e8

  # Compute dual variables robustly with a regularized system.
  # This solves (A * A' + εI) λ = A * c, improving the conditioning.
  try
    dual_matrix = As * As' + reg_param * I(size(As, 1))
    lam = dual_matrix \ (As * cs)
  catch e
    println("Warning: Regularized dual solver failed; using zeros for dual variables")
    lam = zeros(size(As, 1))
  end

  # Compute initial slack variables:
  # s = c - A' * λ. A floor value is imposed to keep s strictly positive.
  s = cs .- As' * lam
  s = max.(s, 1e-1)

  # --- Optional: Primal Correction ---
  # Evaluate how well the initial x satisfies the equality constraints.
  primal_res = As * xs - bs
  if norm(primal_res) > 1e-8
    # Compute a correction term using a similar regularized system.
    try
      correction = (As * As' + reg_param * I(size(As, 1))) \ primal_res
      xs = xs - As' * correction
    catch
      println("Warning: Primal correction failed, proceeding with initial xs")
    end
  end

  # Ensure that both xs and s are strictly positive.
  xs = max.(xs, 1e-12)
  s = max.(s, 1e-12)

  # Compute the initial duality measure (complementarity gap).
  initial_mu = dot(xs, s) / n_std
  println("Initial duality measure: $initial_mu")
  
  # ---------------------------------------------------------------------------
  # Step 3: Main Loop of Mehrotra's Predictor-Corrector Method
  # ---------------------------------------------------------------------------
  
  alpha_safety = 0.9995
  converged = false
  iter = 0
  best_x = copy(xs)
  best_residual = Inf
  
  println("Starting Mehrotra's predictor-corrector iterations")
  println("Iter | Primal Res | Dual Res | Gap | Step P | Step D")
  println("------------------------------------------------------")
  
  while iter < maxit
    iter += 1
    r_p = As * xs .- bs                   # primal residual
    r_d = cs .- As' * lam .- s            # dual residual
    mu = dot(xs, s) / n_std
    
    primal_res_norm = norm(r_p) / max(1.0, norm(bs))
    dual_res_norm = norm(r_d) / max(1.0, norm(cs))
    total_res_norm = norm([r_p; r_d; xs .* s]) / max(1.0, norm([bs; cs]))

    cur_size = sum(abs, xs) + sum(abs, s)

    # ---- Blow-up / infeasibility test ---------------------------------------
    if cur_size > W_max
        println("Iter $iter : ‖(x,s)‖₁ = $cur_size  > W_max = $W_max.  ")
        println("Blow-up condition triggered ⇒ problem is (primal OR dual) infeasible.")
        converged = false
        break
    end
    # -------------------------------------------------------------------------

    
    if total_res_norm <= tol && mu <= tol
      converged = true
      @printf("%4d | %.2e | %.2e | %.2e | CONVERGED\n", iter, primal_res_norm, dual_res_norm, mu)
      break
    end
    
    if total_res_norm < best_residual
      best_residual = total_res_norm
      best_x = copy(xs)
    end
    
    ### Predictor step.
    # Step 1: Predictor step
    xs = max.(xs, 1e-12)
    s  = max.(s, 1e-12) 
    delta_xs_aff, delta_lam_aff, delta_s_aff = solve_kkt_augmented(As, xs, lam, s, cs, bs, 0.0)

    alpha_aff_primal = compute_step_length(xs, delta_xs_aff)
    alpha_aff_dual   = compute_step_length(s, delta_s_aff)

    xs_aff = xs .+ alpha_aff_primal * delta_xs_aff
    s_aff  = s .+ alpha_aff_dual   * delta_s_aff
    mu     = dot(xs, s) / length(xs)
    mu_aff = dot(xs_aff, s_aff) / length(xs)

    sigma = (mu_aff / mu)^3

    # Step 2: Compute correction RHS
    r_c = As' * lam + s - cs
    r_b = As * xs - bs
    comp_corr = clamp.(delta_xs_aff .* delta_s_aff, -1e6, 1e6)
    r_μ = xs .* s .+ comp_corr .- sigma * mu * ones(length(xs))
    rhs_corr = -vcat(r_c, r_b, r_μ)

    # Step 3: Corrector step with computed sigma
    xs = max.(xs, 1e-12)
    s  = max.(s, 1e-12)
    delta_xs, delta_lam, delta_s = solve_kkt_augmented(As, xs, lam, s, cs, bs, sigma, rhs_corr)

    # Step 4: Compute step lengths
    alpha_primal = alpha_safety * compute_step_length(xs, delta_xs)
    alpha_dual   = alpha_safety * compute_step_length(s, delta_s)

    
    @printf("%4d | %.2e | %.2e | %.2e | %.4f | %.4f\n", iter, primal_res_norm, dual_res_norm, mu, alpha_primal, alpha_dual)
    
    # println("Before update xs max: $(maximum(xs)) min: $(minimum(xs))")
    xs = xs .+ (alpha_primal * delta_xs)
    # println("After update xs max: $(maximum(xs)) min: $(minimum(xs))")
    lam = lam .+ (alpha_dual * delta_lam)
    # println("Before update s max: $(maximum(s)) min: $(minimum(s))")
    s = s .+ (alpha_dual * delta_s)
    # println("After update s max: $(maximum(s)) min: $(minimum(s))")
    
    if any(xs .<= 0) || any(s .<= 0)
      println("Warning: Numerical issues; enforcing positivity")
      xs = max.(xs, 1e-10)
      s = max.(s, 1e-10)
    end
  end
  
  if !converged && iter >= maxit
    println("Maximum iterations reached. Using best solution with residual: $best_residual")
    xs = best_x
  end
  
  # ---------------------------------------------------------------------------
  # Step 4: Recover the Original Solution
  # ---------------------------------------------------------------------------
  
  x = zeros(n)
  println("Recovering solution in original variable space")
  for i in 1:n
    trans = var_map[i][1]
    if trans == :bounded
      # x = x' + lo.
      x[i] = xs[var_map[i][2]] + lo[i]
    elseif trans == :lower
      x[i] = xs[var_map[i][2]] + lo[i]
    elseif trans == :upper
      x[i] = hi[i] - xs[var_map[i][2]]
    elseif trans == :free
      x[i] = xs[var_map[i][2]] - xs[var_map[i][3]]
    end
  end
  
  # Check constraint and bound violations.
  original_constraint_violation = norm(A * x - b_orig) / max(1.0, norm(b_orig))
  bound_violation = 0.0
  for i in 1:n
    if isfinite(lo[i]) && x[i] < lo[i] - 1e-8
      bound_violation = max(bound_violation, lo[i] - x[i])
    end
    if isfinite(hi[i]) && x[i] > hi[i] + 1e-8
      bound_violation = max(bound_violation, x[i] - hi[i])
    end
  end
  
  println("Solution quality in original space:")
  println("  Objective value: $(dot(c_orig, x))")
  println("  Constraint violation: $original_constraint_violation")
  println("  Bound violation: $bound_violation")
  
  # Convert to Vector properly
  lam = vec(Array(lam))
  s = vec(Array(s))
  
  return IplpSolution(x, converged, cs, As, bs, xs, lam, s)
end

# Helper function to compute maximum step length.
function compute_step_length(x, dx)
  alpha = 1.0
  for i in 1:length(x)
    if dx[i] < 0
      alpha = min(alpha, -x[i]/dx[i])
    end
  end
  return alpha
end

function solve_kkt_augmented(As, xs, lam, s, cs, bs, σ, custom_rhs=nothing)
  n = length(xs)
  m = size(As, 1)
  I_n = spdiagm(0 => ones(n))
  I_m = spdiagm(0 => ones(m))
  reg_eps = 1e-10  # Regularization strength

  S = spdiagm(0 => s)
  X = spdiagm(0 => xs)

  # Build the block rows of the KKT matrix
  K_top = [spzeros(n,n) As' I_n]
  K_mid = [As spzeros(m,m) spzeros(m,n)]
  K_bot = [S spzeros(n,m) X]
  K = [K_top; K_mid; K_bot]

  # Add diagonal regularization to entire system
  K += reg_eps * spdiagm(0 => ones(n + m + n))

  # Construct RHS
  if custom_rhs === nothing
      r_b = As * xs - bs
      r_c = As' * lam + s - cs
      μ = dot(xs, s) / n
      r_μ = xs .* s .- σ * μ * ones(n)
      rhs = -vcat(r_c, r_b, r_μ)
  else
      rhs = custom_rhs
  end

  Δ = K \ rhs

  delta_xs = Δ[1:n]
  delta_lam = Δ[n+1:n+m]
  delta_s = Δ[n+m+1:end]

  return delta_xs, delta_lam, delta_s
end


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
