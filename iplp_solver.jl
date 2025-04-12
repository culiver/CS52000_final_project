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
  
  # Step 1: Problem Setup and Standardization
  # Extract problem data
  A = Problem.A
  b = Problem.b
  c = Problem.c
  lo = Problem.lo
  hi = Problem.hi
  
  n = length(c)
  m = size(A, 1)
  
  println("Problem dimensions: $n variables, $m constraints")
  
  # Check for basic problem validity
  if n == 0 || m == 0
    println("Error: Empty problem detected")
    return IplpSolution(Float64[], false, Float64[], spzeros(0,0), Float64[], Float64[], Float64[], Float64[])
  end
  
  # Convert to standard form (min c'x s.t. Ax = b, x ≥ 0)
  # Create arrays for the standardized problem
  num_bound_constraints = count(isfinite, lo) + count(isfinite, hi)
  num_std_vars = n + num_bound_constraints  # Original vars + slack vars
  
  println("Converting to standard form (estimated $num_std_vars variables)")
  
  # Initialize standard form components
  As = spzeros(m, num_std_vars)
  cs = zeros(num_std_vars)
  bs = copy(b)
  
  # Set up variable mapping from original to standard form
  var_map = zeros(Int, n)
  next_var_idx = 1
  
  # Process each variable and handle bounds
  for i in 1:n
    if isfinite(lo[i]) && lo[i] != 0
      # Shift variable: x_i = x_i' + lo[i]
      bs .-= A[:,i] * lo[i]
    end
    
    if isfinite(lo[i]) && isfinite(hi[i])
      # Bounded variable: lo[i] ≤ x_i ≤ hi[i]
      # Substitute x_i = x_i' + lo[i] where 0 ≤ x_i' ≤ hi[i] - lo[i]
      As[:,next_var_idx] = A[:,i]
      cs[next_var_idx] = c[i]
      var_map[i] = next_var_idx
      next_var_idx += 1
    elseif isfinite(lo[i])
      # Lower bound only: x_i ≥ lo[i]
      As[:,next_var_idx] = A[:,i]
      cs[next_var_idx] = c[i]
      var_map[i] = next_var_idx
      next_var_idx += 1
    elseif isfinite(hi[i])
      # Upper bound only: x_i ≤ hi[i]
      # Substitute x_i = hi[i] - x_i' where x_i' ≥ 0
      As[:,next_var_idx] = -A[:,i]
      cs[next_var_idx] = -c[i]
      bs .-= A[:,i] * hi[i]
      var_map[i] = next_var_idx
      next_var_idx += 1
    else
      # Free variable: -∞ < x_i < ∞
      # Substitute x_i = x_i⁺ - x_i⁻ where x_i⁺, x_i⁻ ≥ 0
      As[:,next_var_idx] = A[:,i]
      cs[next_var_idx] = c[i]
      var_map[i] = next_var_idx
      next_var_idx += 1
      
      As[:,next_var_idx] = -A[:,i]
      cs[next_var_idx] = -c[i]
      next_var_idx += 1
    end
  end
  
  # Adjust dimensions if needed
  As = As[:, 1:next_var_idx-1]
  cs = cs[1:next_var_idx-1]
  
  # Check if the standardized problem is well-formed
  n_std = size(As, 2)  # Number of variables in standard form
  println("Standard form: $n_std variables, $m equality constraints")
  
  if n_std == 0
    println("Error: No variables in standard form")
    return IplpSolution(Float64[], false, Float64[], spzeros(0,0), Float64[], Float64[], Float64[], Float64[])
  end
  
  # Step 2: Initialization
  println("Initializing interior point")
  
  # Apply scaling to improve numerical stability
  println("Scaling problem data for numerical stability")
  
  # Scale rows of As to have unit norm
  row_norms = zeros(m)
  for i in 1:m
    row_norms[i] = norm(As[i,:])
    if row_norms[i] > 1e-10
      As[i,:] ./= row_norms[i]
      bs[i] /= row_norms[i]
    end
  end
  
  # Scale columns of As to have unit norm where possible
  col_norms = zeros(n_std)
  for j in 1:n_std
    col_norms[j] = norm(As[:,j])
    if col_norms[j] > 1e-10
      As[:,j] ./= col_norms[j]
      cs[j] *= col_norms[j]  # Adjust objective coefficients
    end
  end
  
  # Step 2: Initialization
  println("Initializing interior point")
  
  # Compute a feasible starting point
  # Try to find a point close to the central path
  xs = ones(n_std)
  
  # Set dual variables based on the objective
  lam = zeros(m)
  s = ones(n_std)
  
  # Apply Mehrotra's initial point heuristic
  # Solve for lam to minimize ||As'*lam - c||
  try
    lam = (As * As') \ (As * cs)
  catch
    println("Warning: Could not compute optimal initial dual variables, using zeros")
  end
  
  # Compute initial slack variables
  s = max.(cs .- As' * lam, 1e-1)
  
  # Adjust primal variables to be compatible with slacks
  xs = ones(n_std)
  
  # Optional: Compute a better initial point if needed
  initial_mu = dot(xs, s) / n_std
  println("Initial duality measure: $initial_mu")
  
  # Step 3: Set Parameters
  # Safety factor for step length
  alpha_safety = 0.9995
  
  # Step 4: Main Loop
  converged = false
  iter = 0
  
  # Track best solution found
  best_x = copy(xs)
  best_residual = Inf
  
  println("Starting Mehrotra's predictor-corrector iterations")
  println("Iter | Primal Res | Dual Res | Gap | Step P | Step D")
  println("------------------------------------------------------")
  
  while iter < maxit
    iter += 1
    
    # 4.1 Compute residuals
    r_p = As * xs .- bs   # Primal residual
    r_d = cs .- As' * lam .- s  # Dual residual
    mu = dot(xs, s) / n_std    # Complementarity gap
    
    # Calculate residual norms for reporting
    primal_res_norm = norm(r_p) / max(1.0, norm(bs))
    dual_res_norm = norm(r_d) / max(1.0, norm(cs))
    total_res_norm = norm([r_p; r_d; xs .* s]) / max(1.0, norm([bs; cs]))
    
    # Check convergence
    if total_res_norm <= tol && mu <= tol
      converged = true
      @printf("%4d | %.2e | %.2e | %.2e | CONVERGED\n", 
              iter, primal_res_norm, dual_res_norm, mu)
      break
    end
    
    # Track best solution
    if total_res_norm < best_residual
      best_residual = total_res_norm
      best_x = copy(xs)
    end
    
    # 4.2 Compute affine scaling (predictor) direction
    # Form diagonal matrices
    X = Diagonal(xs)
    S = Diagonal(s)
    
    # Compute the scaling matrix D = X/S with safeguards
    d = xs ./ max.(s, 1e-8)
    D = Diagonal(d)
    
    # Add regularization to improve conditioning
    delta_x = 1e-8
    delta_s = 1e-8
    
    # Solve the normal equations
    # (As*D*As' + delta_s*I) dlam = rhs
    rhs = r_p .- As * (D * r_d)
    
    # Create the regularized normal equations matrix
    reg_matrix = As * D * As' + delta_s * I
    
    # Use try-catch for numerical stability issues
    delta_lam_aff = zeros(m)
    local F = nothing
    try
      # Add regularization to the normal equations matrix
      F = cholesky(Symmetric(reg_matrix))
      delta_lam_aff = F \ rhs
    catch e
      # If Cholesky fails, try a more robust but slower method
      println("  Warning: Cholesky factorization failed, using QR")
      delta_lam_aff = reg_matrix \ rhs
    end
    
    # Recover the other directions with regularization
    delta_s_aff = r_d .+ As' * delta_lam_aff
    delta_xs_aff = -D * delta_s_aff
    
    # 4.3 Compute maximum step lengths
    alpha_primal_aff = compute_step_length(xs, delta_xs_aff)
    alpha_dual_aff = compute_step_length(s, delta_s_aff)
    
    # 4.4 Calculate affine duality gap
    xs_aff = xs .+ alpha_primal_aff * delta_xs_aff
    s_aff = s .+ alpha_dual_aff * delta_s_aff
    mu_aff = dot(xs_aff, s_aff) / n_std
    
    # 4.5 Set centering parameter (Mehrotra's heuristic)
    sigma = (mu_aff / mu)^3
    
    # 4.6 Compute corrector direction
    # Compute correction term for complementarity
    comp_correction = delta_xs_aff .* delta_s_aff .- sigma * mu
    
    # Modify the right-hand side for the corrector step
    rhs_cor = -As * (D * comp_correction)
    
    # Solve the corrector system using the same factorization
    delta_lam_cor = zeros(m)
    try
      # Use the same regularized matrix if F exists
      if F !== nothing
        delta_lam_cor = F \ rhs_cor
      else
        delta_lam_cor = reg_matrix \ rhs_cor
      end
    catch e
      # If reusing factorization fails, solve again
      delta_lam_cor = reg_matrix \ rhs_cor
    end
    
    # Recover the corrector directions
    delta_s_cor = As' * delta_lam_cor
    delta_xs_cor = -D * (delta_s_cor .+ comp_correction)
    
    # 4.7 Combine directions
    delta_xs = delta_xs_aff .+ delta_xs_cor
    delta_lam = delta_lam_aff .+ delta_lam_cor
    delta_s = delta_s_aff .+ delta_s_cor
    
    # 4.8 Compute step lengths with safety factor
    alpha_primal = alpha_safety * compute_step_length(xs, delta_xs)
    alpha_dual = alpha_safety * compute_step_length(s, delta_s)
    
    # Ensure minimum step size to make progress
    min_step = 1e-5
    if alpha_primal < min_step && alpha_dual < min_step
      # If both step lengths are too small, use Mehrotra's heuristic
      # to make a pure centering step
      println("  Warning: Step sizes too small, taking centering step")
      sigma = 0.8  # Heavy centering
      comp_correction = .-sigma * mu
      
      # Solve for a pure centering direction
      rhs_cor = -As * (D * comp_correction)
      try
        if F !== nothing
          delta_lam = F \ rhs_cor
        else
          delta_lam = reg_matrix \ rhs_cor
        end
      catch
        delta_lam = reg_matrix \ rhs_cor
      end
      
      delta_s = As' * delta_lam
      delta_xs = -D * (delta_s .+ comp_correction)
      
      # Recompute step lengths
      alpha_primal = alpha_safety * compute_step_length(xs, delta_xs)
      alpha_dual = alpha_safety * compute_step_length(s, delta_s)
    end
    
    # Print iteration status
    @printf("%4d | %.2e | %.2e | %.2e | %.4f | %.4f\n", 
            iter, primal_res_norm, dual_res_norm, mu, alpha_primal, alpha_dual)
    
    # 4.9 Update iterates
    xs = xs .+ alpha_primal * delta_xs
    lam = lam .+ alpha_dual * delta_lam
    s = s .+ alpha_dual * delta_s
    
    # Additional check to prevent numerical issues
    if any(xs .<= 0) || any(s .<= 0)
      println("Warning: Numerical issues detected, variables not strictly positive")
      # Fix negative values
      xs = max.(xs, 1e-10)
      s = max.(s, 1e-10)
    end
  end
  
  # If we didn't converge, use the best solution found
  if !converged
    println("Maximum iterations reached without convergence.")
    println("Using best solution with residual: $best_residual")
    xs = best_x
  end
  
  # Step 5: Recover original solution
  # Map xs back to original variables
  x = zeros(n)
  
  println("Recovering solution in original variable space")
  
  for i in 1:n
    if var_map[i] > 0
      x[i] = xs[var_map[i]]
      
      # Apply any shift if the variable was bounded from below
      if isfinite(lo[i]) && lo[i] != 0
        x[i] += lo[i]
      end
      
      # If variable was x = hi - x', reverse the transformation
      if isfinite(hi[i]) && !isfinite(lo[i])
        x[i] = hi[i] .- x[i]
      end
    else
      # Handle split free variables if necessary (x = x⁺ - x⁻)
      pos_idx = findfirst(j -> j > 0 && As[:,j] == A[:,i], 1:n_std)
      neg_idx = findfirst(j -> j > 0 && As[:,j] == -A[:,i], 1:n_std)
      
      if pos_idx !== nothing && neg_idx !== nothing
        x[i] = xs[pos_idx] .- xs[neg_idx]
      end
    end
  end
  
  # Check solution feasibility in original space
  original_constraint_violation = norm(A * x .- b) / max(1.0, norm(b))
  bound_violation = 0.0
  for i in 1:n
    if isfinite(lo[i]) && x[i] < lo[i] - 1e-8
      bound_violation = max(bound_violation, lo[i] .- x[i])
    end
    if isfinite(hi[i]) && x[i] > hi[i] + 1e-8
      bound_violation = max(bound_violation, x[i] .- hi[i])
    end
  end
  
  # Report solution quality
  println("Solution quality in original space:")
  println("  Objective value: $(dot(c, x))")
  println("  Constraint violation: $original_constraint_violation")
  println("  Bound violation: $bound_violation")
  
  # Ensure all output values are the correct type
  if isa(lam, SparseMatrixCSC)
    lam = Vector(vec(lam))
  end
  if isa(s, SparseMatrixCSC)
    s = Vector(vec(s))
  end
  
  # Return solution
  return IplpSolution(x, converged, cs, As, bs, xs, lam, s)
end

# Helper function to compute maximum step length
function compute_step_length(x, dx)
  # Find the maximum alpha in [0,1] such that x + alpha*dx >= 0
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
# The matrix identifier "LPnetlib/lp_afiro" should appear in the available list.
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