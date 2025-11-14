# AI Search and Optimization Project

A comprehensive, production-ready Python framework for comparing **Firefly Algorithm** (FA) with classical optimization methods on continuous and discrete benchmark problems.

## 🎯 Project Overview

This project implements and benchmarks multiple optimization algorithms with:

- ✅ **Full type hints** for all functions and classes
- ✅ **Comprehensive error handling** with actionable error messages
- ✅ **>80% test coverage** with edge case testing
- ✅ **Parallel execution** support for faster benchmarking
- ✅ **Academic-grade visualizations** following metaheuristic best practices
- ✅ **Reproducible results** with fixed seeds
- ✅ **Statistical analysis** with Wilcoxon and Friedman tests

### Algorithms Implemented

#### Swarm Intelligence
- **Firefly Algorithm (FA)** - Bio-inspired optimization
  - Continuous optimization variant
  - Discrete Knapsack variant with repair strategies

#### Classical Baselines
- **Hill Climbing (HC)** - Greedy local search with restart
- **Simulated Annealing (SA)** - Probabilistic local search with temperature scheduling
- **Genetic Algorithm (GA)** - Evolutionary optimization with elitism

### Benchmark Problems

#### Continuous Functions
- **Rastrigin** - Highly multimodal with many local minima
  - Dimensions: d=10, 30, 50
  - Global optimum: f(0,...,0) = 0
  - Domain: [-5.12, 5.12]^d
  - Three test configurations: quick convergence, multimodal escape, scalability

#### Discrete Problems
- **0/1 Knapsack** - Maximize value within capacity constraint
  - Sizes: n=50, 100, 200 items
  - 4 instance types: uncorrelated, weakly correlated, strongly correlated, subset-sum
  - DP optimal solution available for n ≤ 100
  - Multiple random seeds for statistical robustness

## 📁 Project Structure

```

CSTTNT_DA1/
├── src/
│   ├── core/                      # Base classes and utilities
│   │   ├── base_optimizer.py      # Abstract optimizer interface
│   │   ├── problem_base.py        # Abstract problem interface
│   │   └── utils.py               # Helper functions
│   │
│   ├── problems/
│   │   ├── continuous/            # Continuous benchmark functions
│   │   │   └── rastrigin.py       # Rastrigin function
│   │   └── discrete/              # Discrete optimization problems
│   │       └── knapsack.py        # 0/1 Knapsack problem with DP solver
│   │
│   ├── swarm/                     # Swarm intelligence algorithms
│   │   └── fa.py                  # Firefly Algorithm (continuous & discrete)
│   │
│   ├── classical/                 # Classical baseline algorithms
│   │   ├── hill_climbing.py       # Hill Climbing with random restart
│   │   ├── simulated_annealing.py # Simulated Annealing
│   │   └── genetic_algorithm.py   # Genetic Algorithm
│   │
│   └── utils/                     # Utility modules
│       └── visualization.py       # Academic-grade plotting functions
│
├── test/                          # Unit tests (>80% coverage)
│   ├── test_problems.py           # Tests for Rastrigin and Knapsack
│   ├── test_firefly_algorithm.py  # Tests for Firefly Algorithm
│   ├── test_classical_algorithms.py
│   ├── test_edge_cases.py
│   ├── test_parallel_execution.py
│   ├── run_all_tests.py
│   └── README.md
│
├── benchmark/                     # Comprehensive benchmark suite
│   ├── config.py                  # Centralized benchmark configurations
│   ├── instance_generator.py      # Knapsack instance generation
│   ├── run_rastrigin.py           # Rastrigin benchmark runner
│   ├── run_knapsack.py            # Knapsack benchmark runner
│   ├── analyze_results.py         # Statistical analysis (Wilcoxon, Friedman)
│   ├── visualize.py               # Generate all plots
│   ├── run_all.py                 # Master script (parallel execution)
│   ├── run_all.sh                 # Shell script wrapper
│   ├── test_benchmarks.py         # Benchmark integration tests
│   ├── results/                   # Auto-generated results
│   │   ├── rastrigin/             # Rastrigin results by config
│   │   ├── knapsack/              # Knapsack results by instance
│   │   ├── plots/                 # All visualizations
│   │   ├── logs/                  # Execution logs
│   │   ├── summaries/             # Statistical summaries
│   │   ├── rastrigin_summary.csv
│   │   └── knapsack_summary.csv
│   └── README.md                  # Detailed benchmark documentation
│
├── results/                       # Legacy results directory (deprecated)
├── demo.py                        # Quick demonstration script
├── environment.yml                # Conda environment
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── QUICKSTART.md                  # Quick start guide

````

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- NumPy
- SciPy (for statistical tests)
- Matplotlib (for visualization)
- Pandas (for data analysis)
- pytest (for testing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/1234quan1234/CSTTNT_DA1.git
cd CSTTNT_DA1
````

2. Install dependencies using conda (recommended):

```bash
conda env create -f environment.yml
conda activate aisearch
```

Or using pip:

```bash
pip install -r requirements.txt
```

### Quick Start

#### Option 1: Run Complete Benchmark Suite (Recommended)

```bash
# Fast mode with parallel execution (use all CPU cores)
python benchmark/run_all.py --quick --jobs -1

# Full benchmark with 4 parallel workers
python benchmark/run_all.py --full --jobs 4
```

This will:

* Run all Rastrigin configurations (quick_convergence, multimodal_escape, scalability)
* Run all Knapsack instances (n=50, 100, 200 with 4 instance types)
* Perform 30 independent runs per configuration
* Generate statistical analysis (mean, std, median, Wilcoxon tests, Friedman tests)
* Create all visualizations in `benchmark/results/plots/`

#### Option 2: Run Individual Benchmarks

**Rastrigin Benchmark:**

```bash
# Quick convergence test (d=10, ~2 minutes with 4 cores)
python benchmark/run_rastrigin.py --config quick_convergence --jobs 4

# Multimodal escape test (d=30, ~5 minutes)
python benchmark/run_rastrigin.py --config multimodal_escape --jobs -1

# Scalability test (d=50, ~10 minutes)
python benchmark/run_rastrigin.py --config scalability --jobs -1
```

**Knapsack Benchmark:**

```bash
# Small instances (n=50, ~5 minutes with 4 cores)
python benchmark/run_knapsack.py --size 50 --jobs 4

# Medium instances with DP optimal (n=100, ~15 minutes)
python benchmark/run_knapsack.py --size 100 --jobs -1

# Large instances (n=200, ~30 minutes)
python benchmark/run_knapsack.py --size 200 --jobs 4
```

#### Option 3: Analysis and Visualization Only

If you already have benchmark results:

```bash
# Generate statistical analysis
python benchmark/analyze_results.py --problem all

# Generate all plots
python benchmark/visualize.py
```

### Quick Test

Test individual algorithm implementations:

```bash
# Test Firefly Algorithm
python src/swarm/fa.py

# Test problem definitions
python src/problems/continuous/rastrigin.py
python src/problems/discrete/knapsack.py

# Test classical algorithms
python src/classical/hill_climbing.py
python src/classical/simulated_annealing.py
python src/classical/genetic_algorithm.py
```

## 💡 Usage Examples

### Example 1: Rastrigin Function with Firefly Algorithm

```python
import numpy as np
from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer

# Create problem instance
problem = RastriginProblem(dim=10)

# Create optimizer with parameters tuned for multimodal problems
optimizer = FireflyContinuousOptimizer(
    problem=problem,
    n_fireflies=40,
    alpha=0.3,      # Higher randomization for exploration
    beta0=1.0,      # Base attractiveness
    gamma=1.0,      # Light absorption coefficient
    seed=42         # For reproducibility
)

# Run optimization
best_solution, best_fitness, history, trajectory = optimizer.run(max_iter=100)

print(f"Best fitness: {best_fitness:.6f}")
print(f"Error to optimum: {abs(best_fitness):.6f}")
print(f"Convergence: {history[0]:.4f} -> {history[-1]:.4f}")
```

### Example 2: Knapsack with Discrete Firefly Algorithm

```python
import numpy as np
from src.problems.discrete.knapsack import KnapsackProblem
from src.swarm.fa import FireflyKnapsackOptimizer

# Create Knapsack instance (uncorrelated type)
rng = np.random.RandomState(42)
n_items = 50
values = rng.randint(10, 100, n_items)
weights = rng.randint(1, 50, n_items)
capacity = int(0.5 * np.sum(weights))

problem = KnapsackProblem(values, weights, capacity)

# Compute DP optimal for comparison (feasible for n ≤ 100)
dp_optimal = problem.solve_dp()
print(f"DP Optimal: {dp_optimal}")

# Create optimizer with REPAIR strategy (fair comparison)
optimizer = FireflyKnapsackOptimizer(
    problem=problem,
    n_fireflies=60,
    alpha_flip=0.2,
    max_flips_per_move=3,
    constraint_handling="repair",  # 'repair' or 'penalty'
    seed=42
)

# Run optimization
best_sol, best_fit, history, trajectory = optimizer.run(max_iter=166)

total_value = -best_fit  # Negate for actual value
total_weight = np.sum(best_sol * weights)
optimality_gap = (dp_optimal - total_value) / dp_optimal * 100

print(f"Best value: {total_value:.0f} (DP: {dp_optimal:.0f})")
print(f"Optimality gap: {optimality_gap:.2f}%")
print(f"Weight: {total_weight:.0f}/{capacity} ({total_weight/capacity*100:.1f}%)")
print(f"Items selected: {np.sum(best_sol)}/{n_items}")
```

**Constraint Handling Strategies:**

The framework supports two strategies for handling Knapsack capacity constraints:

1. **Repair Strategy** (`constraint_handling="repair"`):

   * Infeasible solutions are repaired using greedy removal
   * Ensures all solutions are feasible
   * Fair comparison across all algorithms
   * Recommended for benchmarking

2. **Penalty Strategy** (`constraint_handling="penalty"`):

   * Infeasible solutions receive large penalty
   * Allows exploration of infeasible space
   * May find better solutions by exploring infeasible regions
   * Corresponds to the original implementation behavior

**Run benchmarks with both strategies:**

```bash
# Repair strategy only (fair comparison)
python benchmark/run_knapsack.py --size 50 --constraint repair

# Penalty strategy only (original behavior)
python benchmark/run_knapsack.py --size 50 --constraint penalty

# Both strategies for comparison
python benchmark/run_knapsack.py --size 50 --constraint both
```

### Example 3: Compare All Algorithms

```python
from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer
from src.classical.simulated_annealing import SimulatedAnnealingOptimizer
from src.classical.hill_climbing import HillClimbingOptimizer
from src.classical.genetic_algorithm import GeneticAlgorithmOptimizer

problem = RastriginProblem(dim=10)
max_iter = 125  # Same budget for fair comparison

# Firefly Algorithm
fa = FireflyContinuousOptimizer(problem, n_fireflies=40, alpha=0.3, seed=42)
_, fa_fit, fa_hist, _ = fa.run(max_iter=max_iter)

# Simulated Annealing
sa = SimulatedAnnealingOptimizer(problem, initial_temp=100, cooling_rate=0.95, seed=42)
_, sa_fit, sa_hist, _ = sa.run(max_iter=max_iter)

# Hill Climbing with restart
hc = HillClimbingOptimizer(problem, num_neighbors=20, restart_interval=50, seed=42)
_, hc_fit, hc_hist, _ = hc.run(max_iter=max_iter)

# Genetic Algorithm
ga = GeneticAlgorithmOptimizer(problem, pop_size=40, crossover_rate=0.8, seed=42)
_, ga_fit, ga_hist, _ = ga.run(max_iter=max_iter)

print(f"FA: {abs(fa_fit):.6f}")
print(f"SA: {abs(sa_fit):.6f}")
print(f"HC: {abs(hc_fit):.6f}")
print(f"GA: {abs(ga_fit):.6f}")
```

## 📊 Benchmark Configurations

### Rastrigin Configurations

| Config Name         | Dimension | Budget (evals) | Max Iter | Success Thresholds                      | Purpose                  |
| ------------------- | --------- | -------------- | -------- | --------------------------------------- | ------------------------ |
| `quick_convergence` | 10        | 10,000         | 250      | Gold: 1.0, Silver: 10.0, Bronze: 30.0    | Fast convergence test    |
| `multimodal_escape` | 30        | 30,000         | 500      | Gold: 5.0, Silver: 25.0, Bronze: 50.0   | Escape local minima      |
| `scalability`       | 50        | 50,000         | 625      | Gold: 10.0, Silver: 50.0, Bronze: 80.0 | High-dimensional scaling |

**Success Levels:**

* **Gold** 🥇: Very close to global optimum (toughest threshold)
* **Silver** 🥈: Escaped bad regions, found good local minimum
* **Bronze** 🥉: Escaped worst regions (most lenient threshold)

**Algorithm Parameters:**

* **FA**: n_fireflies=40, α=0.3, β₀=1.0, γ=1.0
* **SA**: T₀=100, cooling=0.95, step=0.5
* **HC**: neighbors=20, step=0.5, restart=50
* **GA**: pop=40, crossover=0.8, mutation=0.1

### Knapsack Configurations

| n Items | Instance Types | Seeds        | Budget (evals) | Max Iter (FA/GA) | Max Iter (SA/HC) | DP Optimal? |
| ------- | -------------- | ------------ | -------------- | ---------------- | ---------------- | ----------- |
| 50      | All 4 types    | 42, 123, 999 | 10,000         | 166              | 10,000           | ✓ Yes       |
| 100     | All 4 types    | 42, 123, 999 | 15,000         | 250              | 15,000           | ✓ Yes       |
| 200     | Uncorr, Weak   | 42, 123, 999 | 30,000         | 500              | 30,000           | ✗ No        |

**Instance Types:**

1. **Uncorrelated**: Random values and weights
2. **Weakly Correlated**: values ≈ weights ± noise
3. **Strongly Correlated**: values = weights + 100
4. **Subset-Sum**: values = weights (hardest)

**Algorithm Parameters:**

* **FA**: n_fireflies=60, α_flip=0.2, max_flips=3, repair="greedy_remove"
* **SA**: T₀=1000, cooling=0.95
* **HC**: neighbors=20, restart=100
* **GA**: pop=60, crossover=0.8, mutation=1/n, elitism=0.1

## 📈 Output Format

All benchmark results are saved in JSON format for reproducibility.

### Rastrigin Results

**File naming:** `benchmark/results/rastrigin/rastrigin_{config}_{algo}_{scenario}_{timestamp}.json`

```json
{
  "metadata": {
    "problem": "rastrigin",
    "config_name": "quick_convergence",
    "algorithm": "FA",
    "scenario": "out_of_the_box",
    "timestamp": "20251110T200402",
    "dimension": 10,
    "budget": 10000,
    "max_iter": 250,
    "pop_size": 40,
    "problem_seed": 42,
    "n_runs": 30,
    "n_successful": 28,
    "n_failed": 2,
    "status_breakdown": {
      "ok": 28,
      "timeout": 1,
      "nan": 1
    },
    "thresholds_used": {
      "gold": 1.0,
      "silver": 5.0,
      "bronze": 10.0
    },
    "avg_budget_utilization": 0.998
  },
  "all_results": [
    {
      "algorithm": "FA",
      "seed": 0,
      "algo_seed": 0,
      "problem_seed": 42,
      "best_fitness": 8.4567,
      "history": [45.6, 34.2, 23.1, 15.8, 8.4567],
      "elapsed_time": 2.15,
      "evaluations": 10000,
      "budget": 10000,
      "budget_utilization": 1.0,
      "success_levels": {
        "gold": {
          "success": false,
          "threshold": 1.0,
          "hit_evaluations": null
        },
        "silver": {
          "success": false,
          "threshold": 5.0,
          "hit_evaluations": null
        },
        "bronze": {
          "success": true,
          "threshold": 10.0,
          "hit_evaluations": 8200
        }
      },
      "status": "ok",
      "error_type": null,
      "error_msg": null
    }
  ]
}
```

**Highlights:**

* **Metadata**: Centralized configuration tracking, status breakdown, budget utilization metrics
* **Tracking**: Every run has `status`, `error_type`, `error_msg` for error investigation
* **Hit time**: `hit_evaluations` records when target threshold was achieved (null if never hit)
* **Budget control**: `budget_utilization` ensures algorithm stayed within evaluation budget
* **Problem seed**: `problem_seed` enables problem reproducibility

### Knapsack Results

**File naming:** `benchmark/results/knapsack/knapsack_n{size}_{type}_seed{seed}_{algo}_{strategy}_{timestamp}.json`

```json
{
  "metadata": {
    "problem": "knapsack",
    "n_items": 50,
    "instance_type": "uncorrelated",
    "instance_seed": 42,
    "algorithm": "FA",
    "timestamp": "20251110T202419",
    "capacity": 500.0,
    "budget": 10000,
    "n_runs": 30,
    "n_successful": 30,
    "n_failed": 0,
    "status_breakdown": {"ok": 30},
    "dp_optimal": 2450.0,
    "gap_thresholds": {
      "gold": 1.0,
      "silver": 5.0,
      "bronze": 10.0
    },
    "tier_success_rates": {
      "SR_Gold_%": 5.0,
      "SR_Silver_%": 45.0,
      "SR_Bronze_%": 80.0
    },
    "constraint_handling": "repair",
    "avg_gap_%": 2.34,
    "avg_feasibility_rate": 1.0
  },
  "all_results": [
    {
      "algorithm": "FA",
      "seed": 0,
      "algo_seed": 0,
      "instance_seed": 42,
      "best_value": 2387.0,
      "best_fitness": -2387.0,
      "total_weight": 487.5,
      "is_feasible": true,
      "gap_relative": 2.57,
      "gap_tier": "silver",
      "success_levels": {
        "gold": {
          "success": false,
          "threshold": 1.0,
          "hit_evaluations": null
        },
        "silver": {
          "success": true,
          "threshold": 5.0,
          "hit_evaluations": 8200
        },
        "bronze": {
          "success": true,
          "threshold": 10.0,
          "hit_evaluations": 3400
        }
      },
      "history": [-1200.0, -1500.0, ..., -2387.0],
      "elapsed_time": 3.45,
      "items_selected": 18,
      "capacity_utilization": 0.975,
      "status": "ok"
    }
  ]
}
```

**Key Updates:**
- **gap_thresholds**: Multi-tier thresholds (Gold/Silver/Bronze)
- **tier_success_rates**: Success rate at each tier
- **gap_relative**: Relative gap (%) to DP optimal
- **gap_tier**: Best tier achieved ("gold"/"silver"/"bronze"/null)
- **success_levels**: Per-tier hit time tracking (like Rastrigin)

