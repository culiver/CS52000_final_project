using MatrixDepot, SparseArrays, LinearAlgebra, Printf

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
  lo = Problem.lo
  hi = Problem.hi
  n = length(c_orig)
  m = size(A, 1)
  
  println("Problem dimensions: $n variables, $m constraints")
  if n == 0 || m == 0
    println("Error: Empty problem detected")
    return IplpSolution(Float64[], false, Float64[], spzeros(0,0), Float64[], Float64[], Float64[], Float64[])
  end
  
  # We now convert the problem to standard form:
  #   min cs'*xs   subject to   As * xs = bs, xs >= 0
  # For variables in the original problem, we use different transformations:
  #
  # 1. Bounded variable (finite lo and hi): 
  #      x = x' + lo, with 0 <= x' <= hi - lo.
  #      To enforce the upper bound, we add an extra equality: 
  #           x' + s = hi - lo,  s >= 0.
  # 2. Lower-bound only (finite lo): 
  #      x = x' + lo, with x' >= 0.
  # 3. Upper-bound only (finite hi): 
  #      x = hi - x', with x' >= 0.
  # 4. Free variable (no finite bounds): 
  #      x = x⁺ - x⁻, with x⁺, x⁻ >= 0.
  #
  # We also build a mapping (var_map) so that later we can recover the original variable x.
  # For each original variable i, var_map[i] will hold:
  #   (:bounded, col)  -- for finite lo and hi, where col is the column index of x'
  #   (:lower, col)    -- for lower bound only
  #   (:upper, col)    -- for upper bound only
  #   (:free, col_plus, col_minus)  -- for free variable.
  
  extra_rows = 0  # additional rows for bounded variables’ upper-bound constraints
  num_std_vars = 0
  var_map = Vector{Any}(undef, n)
  
  for i in 1:n
    if isfinite(lo[i]) && isfinite(hi[i])
      extra_rows += 1          # one extra equality row for the upper-bound constraint
      num_std_vars += 2        # one column for the shifted variable and one for its slack
      var_map[i] = (:bounded, num_std_vars - 1)  # store index of the shifted variable (x')
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
  
  total_rows = m + extra_rows  # original m equations plus one for each bounded variable
  
  println("Converting to standard form: $num_std_vars standard variables, $total_rows equality constraints")
  
  # Initialize standard form arrays.
  As = spzeros(total_rows, num_std_vars)
  cs = zeros(num_std_vars)
  bs = zeros(total_rows)
  
  # For the original m constraint rows, adjust b for lower/upper shifts.
  b_adj = copy(b_orig)
  for i in 1:n
    if isfinite(lo[i])
      # For both bounded and lower-only variables, subtract A[:, i] * lo[i]
      b_adj .-= A[:, i] * lo[i]
    end
    if (!isfinite(lo[i])) && isfinite(hi[i])
      # For upper-bound–only variables: x = hi - x', so subtract A[:, i] * hi[i]
      b_adj .-= A[:, i] * hi[i]
    end
  end
  bs[1:m] = b_adj  # first m rows are from the original constraints
  
  # Now fill in the columns for each variable.
  extra_row_index = m  # next available extra row is at index m+1
  next_var_idx = 1     # next available standard variable column
  
  for i in 1:n
    if var_map[i][1] == :bounded
      # For variables with both finite lo and hi:
      # Transformation: x = x' + lo, with 0 <= x' <= hi - lo,
      # and enforce x' + s = hi - lo.
      local_col = var_map[i][2]  # index for x' (shifted variable)
      # Main constraint: add A[:,i] * x'
      As[1:m, local_col] = A[:, i]
      cs[local_col] = c_orig[i]
      # Now add the extra row constraint for the upper bound:
      extra_row_index += 1
      # In the extra row, we enforce: x' + slack = hi - lo.
      # x' is in column "local_col", and the slack is the next column.
      As[extra_row_index, local_col] = 1.0
      As[extra_row_index, local_col+1] = 1.0
      bs[extra_row_index] = hi[i] - lo[i]
      cs[local_col+1] = 0.0   # slack variable has zero cost
      next_var_idx = max(next_var_idx, local_col+2)
    elseif var_map[i][1] == :lower
      # Lower-bound only: x = x' + lo, with x' >= 0.
      local_col = var_map[i][2]
      As[1:m, local_col] = A[:, i]
      cs[local_col] = c_orig[i]
      next_var_idx = max(next_var_idx, local_col+1)
    elseif var_map[i][1] == :upper
      # Upper-bound only: x = hi - x', with x' >= 0.
      local_col = var_map[i][2]
      As[1:m, local_col] = -A[:, i]  # note the negative sign
      cs[local_col] = -c_orig[i]
      next_var_idx = max(next_var_idx, local_col+1)
    elseif var_map[i][1] == :free
      # Free variable: x = x⁺ - x⁻, with both components >= 0.
      local_plus = var_map[i][2]
      local_minus = var_map[i][3]
      As[1:m, local_plus] = A[:, i]
      As[1:m, local_minus] = -A[:, i]
      cs[local_plus] = c_orig[i]
      cs[local_minus] = -c_orig[i]
      next_var_idx = max(next_var_idx, local_minus+1)
    end
  end
  
  # Truncate cs if we allocated extra space.
  cs = cs[1:next_var_idx-1]
  As = As[:, 1:next_var_idx-1]
  println("Standard form: $(size(As,2)) variables, $(size(As,1)) equality constraints")
  
  # ---------------------------------------------------------------------------
  # Step 2: Initialization of the Interior-Point Method
  # ---------------------------------------------------------------------------
  
  println("Initializing interior point")
  # Scaling for numerical stability
  row_norms = zeros(m)
  for i in 1:m
    row_norm = norm(As[i, :])
    row_norms[i] = row_norm
    if row_norm > 1e-10
      As[i, :] ./= row_norm
      bs[i] /= row_norm
    end
  end
  
  n_std = size(As, 2)
  # (Note: we do not scale extra rows in this snippet, though one could scale them separately.)
  col_norms = zeros(n_std)
  for j in 1:n_std
    col_norm = norm(As[:, j])
    col_norms[j] = col_norm
    if col_norm > 1e-10
      As[:, j] ./= col_norm
      cs[j] *= col_norm  # adjust objective coefficients
    end
  end
  
  # Initialization for variables:
  xs = ones(n_std)
  lam = zeros(size(As, 1))
  s = ones(n_std)
  
  # Use a heuristic to initialize dual variables based on the objective:
  try
    lam = (As * As') \ (As * cs)
  catch
    println("Warning: Could not compute optimal initial dual variables; using zeros")
  end
  s = max.(cs .- As' * lam, 1e-1)
  
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
    
    if total_res_norm <= tol && mu <= tol
      converged = true
      @printf("%4d | %.2e | %.2e | %.2e | CONVERGED\n", iter, primal_res_norm, dual_res_norm, mu)
      break
    end
    
    if total_res_norm < best_residual
      best_residual = total_res_norm
      best_x = copy(xs)
    end
    
    # Predictor step
    X = Diagonal(xs)
    S = Diagonal(s)
    d = xs ./ max.(s, 1e-8)
    D = Diagonal(d)
    
    delta_x = 1e-8
    delta_s = 1e-8
    
    rhs = r_p .- As * (D * r_d)
    reg_matrix = As * D * As' + delta_s * I
    delta_lam_aff = zeros(size(As, 1))
    localF = nothing
    try
      localF = cholesky(Symmetric(reg_matrix))
      delta_lam_aff = localF \ rhs
    catch e
      println("  Warning: Cholesky factorization failed, using QR")
      delta_lam_aff = reg_matrix \ rhs
    end
    
    delta_s_aff = r_d .+ As' * delta_lam_aff
    delta_xs_aff = -D * delta_s_aff
    
    alpha_primal_aff = compute_step_length(xs, delta_xs_aff)
    alpha_dual_aff = compute_step_length(s, delta_s_aff)
    
    xs_aff = xs .+ alpha_primal_aff * delta_xs_aff
    s_aff = s .+ alpha_dual_aff * delta_s_aff
    mu_aff = dot(xs_aff, s_aff) / n_std
    
    sigma = (mu_aff / mu)^3
    
    comp_correction = delta_xs_aff .* delta_s_aff .- sigma * mu
    rhs_cor = -As * (D * comp_correction)
    
    delta_lam_cor = zeros(size(As, 1))
    try
      if localF !== nothing
        delta_lam_cor = localF \ rhs_cor
      else
        delta_lam_cor = reg_matrix \ rhs_cor
      end
    catch e
      delta_lam_cor = reg_matrix \ rhs_cor
    end
    
    delta_s_cor = As' * delta_lam_cor
    delta_xs_cor = -D * (delta_s_cor .+ comp_correction)
    
    delta_xs = delta_xs_aff .+ delta_xs_cor
    delta_lam = delta_lam_aff .+ delta_lam_cor
    delta_s = delta_s_aff .+ delta_s_cor
    
    alpha_primal = alpha_safety * compute_step_length(xs, delta_xs)
    alpha_dual = alpha_safety * compute_step_length(s, delta_s)
    
    min_step = 1e-5
    if alpha_primal < min_step && alpha_dual < min_step
      println("  Warning: Step sizes too small, taking centering step")
      sigma = 0.8
      comp_correction = -sigma * mu
      rhs_cor = -As * (D * comp_correction)
      try
        if localF !== nothing
          delta_lam = localF \ rhs_cor
        else
          delta_lam = reg_matrix \ rhs_cor
        end
      catch
        delta_lam = reg_matrix \ rhs_cor
      end
      delta_s = As' * delta_lam
      delta_xs = -D * (delta_s .+ comp_correction)
      alpha_primal = alpha_safety * compute_step_length(xs, delta_xs)
      alpha_dual = alpha_safety * compute_step_length(s, delta_s)
    end
    
    @printf("%4d | %.2e | %.2e | %.2e | %.4f | %.4f\n", iter, primal_res_norm, dual_res_norm, mu, alpha_primal, alpha_dual)
    
    xs += alpha_primal * delta_xs
    lam += alpha_dual * delta_lam
    s += alpha_dual * delta_s
    
    if any(xs .<= 0) || any(s .<= 0)
      println("Warning: Numerical issues detected, variables not strictly positive")
      xs = max.(xs, 1e-10)
      s = max.(s, 1e-10)
    end
  end
  
  if !converged
    println("Maximum iterations reached without convergence.")
    println("Using best solution with residual: $best_residual")
    xs = best_x
  end
  
  # ---------------------------------------------------------------------------
  # Step 4: Recover the Original Solution
  # ---------------------------------------------------------------------------
  
  x = zeros(n)
  
  println("Recovering solution in original variable space")
  for i in 1:n
    if var_map[i][1] == :bounded
      # For bounded variable, x = x' + lo.
      # (The extra slack variable is not part of the original variable.)
      idx = var_map[i][2]
      x[i] = xs[idx] + lo[i]
    elseif var_map[i][1] == :lower
      idx = var_map[i][2]
      x[i] = xs[idx] + lo[i]
    elseif var_map[i][1] == :upper
      idx = var_map[i][2]
      x[i] = hi[i] - xs[idx]
    elseif var_map[i][1] == :free
      idx_plus = var_map[i][2]
      idx_minus = var_map[i][3]
      x[i] = xs[idx_plus] - xs[idx_minus]
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
  
  # Ensure output types are correct.
  lam = Vector(lam)
  s = Vector(s)
  
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

# =============================================================================
# Test the Setup Using a MatrixDepot LPnetlib Problem
# =============================================================================

# Download an LP problem from LPnetlib.
md = mdopen("LPnetlib/lp_afiro")
println("LPnetlib/lp_afiro matrix size: ", size(md.A))

# Convert the downloaded matrix descriptor to our problem format.
problem = convert_matrixdepot(md)

# Set the tolerance for convergence.
tol = 1e-6

# Call the interior-point LP solver.
solution = iplp(problem, tol; maxit=100)

if solution.flag
  println("Interior-point solver converged!")
  println("Optimal solution x: ", solution.x)
else
  println("Interior-point solver did not converge.")
end