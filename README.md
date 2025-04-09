# CS52000_final_project

This project demonstrates solving linear programming problems using Julia with the JuMP modeling language and Clp solver.

## Environment Setup

### Prerequisites
- Julia 1.6 or later installed on your system
- Git (for cloning the repository)

### Installation Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd CS52000_final_project
```

2. Start Julia and activate the project environment:
```bash
julia --project=.
```

3. Install required packages:
```julia
using Pkg
Pkg.add("MatrixDepot")
Pkg.add("Clp")
Pkg.add("JuMP")
Pkg.add("SparseArrays")  # should already be in stdlib
```

## Running the Code

1. Start Julia in the project directory:
```bash
julia --project=.
```

2. Run the example code:
```julia
include("example.jl")
```

## Project Structure

- `example.jl`: Main example file demonstrating linear programming problem solving
- `Project.toml`: Project dependencies and configuration
- `Manifest.toml`: Exact versions of all dependencies

## Dependencies

- MatrixDepot: For accessing test matrices
- Clp: Linear programming solver
- JuMP: Mathematical optimization modeling language
- SparseArrays: Standard library for sparse matrix operations

## Notes

- The example uses the "lp_afiro" test problem from the LPnetlib collection
- The code demonstrates how to:
  - Load a test matrix
  - Set up a linear programming problem
  - Solve it using the Clp solver
