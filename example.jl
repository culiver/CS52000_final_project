using Pkg
Pkg.activate(".")  # activate current folder as the environment

using MatrixDepot, SparseArrays
md = mdopen("LPnetlib/lp_afiro")

# Show the matrix metadata
println("Size: ", size(md.A))

using JuMP, Clp

# function solve_lp_clp(md)
#     model = Model(Clp.Optimizer)
#     n = size(md.A, 2)

#     # Define variables with bounds
#     @variable(model, md.lo[i] <= x[i=1:n] <= md.hi[i])

#     # Add constraints: A*x = b
#     for i in 1:size(md.A, 1)
#         @constraint(model, sum(md.A[i, j] * x[j] for j in 1:n) == md.b[i])
#     end

#     # Objective: Minimize c'x
#     @objective(model, Min, sum(md.c[j] * x[j] for j in 1:n))

#     optimize!(model)

#     if termination_status(model) == MOI.OPTIMAL
#         x_opt = value.(x)
#         obj_val = objective_value(model)
#         println("Optimal solution found.")
#         println("Objective value: ", obj_val)
#         return x_opt, obj_val
#     else
#         println("Solver failed or did not converge.")
#         return nothing
#     end
# end

# x, obj = solve_lp_clp(md)