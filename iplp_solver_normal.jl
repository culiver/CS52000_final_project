using MatrixDepot, SparseArrays, LinearAlgebra, Printf, SuiteSparse, SuiteSparse.CHOLMOD

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
    r_b = As * xs .- bs
    r_c = As' * lam .+ s .- cs
    mu = dot(xs, s) / n_std
    
    primal_res_norm = norm(r_b) / max(1.0, norm(bs))
    dual_res_norm = norm(r_c) / max(1.0, norm(cs))
    total_res_norm = norm([r_b; r_c; xs .* s]) / max(1.0, norm([bs; cs]))
    # total_res_norm = norm([r_b; r_c; xs .* s]) / norm([bs; cs])

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
    
    # Predictor step.
    X = Diagonal(xs)
    S = Diagonal(s)
    S_inv = inv(S)
    d = xs ./ s
    D = Diagonal(d) # is the D^2 in the textbook

    # Regularization parameters - now properly defined as scalars
    delta_s_param = 1e-8
    
    r_xs_aff = X * s
    rhs = -r_b .+ As * (-S_inv * X * r_c + S_inv * r_xs_aff) # r_xs = X * s
    reg_matrix = As * D * As' + delta_s_param * I(size(As,1))
    delta_lam_aff = zeros(size(As, 1))
    localF = nothing
    try
      # Use CHOLMOD's sparsifying Cholesky with AMD ordering
      # CHOLMOD automatically selects the best ordering method
      sym_reg_matrix = Symmetric(reg_matrix)
      cholmod_factor = cholesky(sym_reg_matrix)
      localF = cholmod_factor
      delta_lam_aff = cholmod_factor \ rhs
    catch e
      println("  Warning: CHOLMOD Cholesky failed, using QR")
      delta_lam_aff = reg_matrix \ rhs
    end
    
    delta_s_aff = -r_c .- As' * delta_lam_aff
    delta_xs_aff = -S_inv * (r_xs_aff + X * delta_s_aff)
    
    alpha_primal_aff = compute_step_length(xs, delta_xs_aff)
    alpha_dual_aff = compute_step_length(s, delta_s_aff)
    
    xs_aff = xs .+ alpha_primal_aff * delta_xs_aff
    s_aff = s .+ alpha_dual_aff * delta_s_aff
    mu_aff = dot(xs_aff, s_aff) / n_std
    
    sigma = (mu_aff / mu)^3
    
    r_xs_cor = delta_xs_aff .* delta_s_aff .- sigma * mu
    rhs_cor = -r_b + As * (-S_inv * X * r_c + S_inv * r_xs_cor)
    
    delta_lam_cor = zeros(size(As, 1))
    try
      if localF !== nothing
        delta_lam_cor = localF \ rhs_cor
      else
        # If we don't have a factorization, try to create one with fill-reducing ordering
        sym_reg_matrix = Symmetric(reg_matrix)
        cholmod_factor = cholesky(sym_reg_matrix)
        delta_lam_cor = cholmod_factor \ rhs_cor
      end
    catch e
      println("  Warning: Failed to solve with Cholesky, using direct solver")
      delta_lam_cor = reg_matrix \ rhs_cor
    end
    
    delta_s_cor = -r_c - As' * delta_lam_cor
    delta_xs_cor = -S_inv * (r_xs_cor + X * delta_s_cor)
    
    delta_xs = delta_xs_aff .+ delta_xs_cor
    delta_lam = delta_lam_aff .+ delta_lam_cor
    delta_s = delta_s_aff .+ delta_s_cor
    
    alpha_primal = alpha_safety * compute_step_length(xs, delta_xs)
    alpha_dual = alpha_safety * compute_step_length(s, delta_s)
    
    @printf("%4d | %.2e | %.2e | %.2e | %.4f | %.4f\n", iter, primal_res_norm, dual_res_norm, mu, alpha_primal, alpha_dual)
    
    xs = xs .+ (alpha_primal * delta_xs)
    lam = lam .+ (alpha_dual * delta_lam)
    s = s .+ (alpha_dual * delta_s)
    
    xs = max.(xs, 1e-12)
    s = max.(s, 1e-12)
  end
  
  if !converged
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
