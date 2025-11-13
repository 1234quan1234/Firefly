# Test Suite

This folder contains comprehensive unit tests for the AI Search & Optimization Framework.

## Test Structure
```
test/
├── __init__.py                      # Package initialization
├── test_continuous_problems.py      # Tests for Rastrigin function
├── test_knapsack_problem.py         # Tests for Knapsack problem
├── test_firefly_algorithm.py        # Tests for Firefly Algorithm (continuous & discrete)
├── test_classical_algorithms.py    # Tests for classical algorithms (HC, SA, GA)
├── test_edge_cases.py              # Edge cases and boundary conditions
├── test_parallel_execution.py      # Parallel execution tests
├── run_all_tests.py                # Main test runner
└── README.md                       # This file
```

## Running Tests

### Run All Tests
```bash
# From project root
python test/run_all_tests.py

# Or with verbose output
python test/run_all_tests.py -v

# With parallel execution (faster)
python test/run_all_tests.py -j 4
```

### Run Specific Test Module
```bash
python test/run_all_tests.py test_continuous_problems
python test/run_all_tests.py test_firefly_algorithm
```

### Run Individual Test File
```bash
python -m unittest test.test_continuous_problems
python -m unittest test.test_knapsack_problem
```

### Run Specific Test Class or Method
```bash
python -m unittest test.test_continuous_problems.TestSphereProblem
python -m unittest test.test_continuous_problems.TestSphereProblem.test_optimum_value
```

## Test Coverage

### Continuous Problems (`test_continuous_problems.py`)
- ✓ Rastrigin function dimension and bounds
- ✓ Optimum value verification (global minimum at origin)
- ✓ Fitness calculation correctness
- ✓ Random solution generation within bounds
- ✓ Multimodality properties

### Knapsack Problem (`test_knapsack_problem.py`)
- ✓ Item weights and values
- ✓ Capacity constraints
- ✓ DP optimal solution verification
- ✓ Random instance generation (4 types: uncorrelated, weakly correlated, strongly correlated, subset-sum)
- ✓ Invalid input handling
- ✓ Greedy repair strategy

### Firefly Algorithm (`test_firefly_algorithm.py`)
- ✓ Continuous optimizer initialization and parameters
- ✓ Output format validation (solution, fitness, history, stats_history)
- ✓ Convergence behavior on Rastrigin
- ✓ Deterministic results with seed
- ✓ Knapsack optimizer initialization
- ✓ Valid binary solution generation
- ✓ Feasibility constraint checking
- ✓ Repair vs penalty constraint handling

### Classical Algorithms (`test_classical_algorithms.py`)
- ✓ Hill Climbing with random restart
- ✓ Simulated Annealing with temperature schedule
- ✓ Genetic Algorithm with crossover, mutation, and elitism
- ✓ Convergence for all algorithms
- ✓ Constraint handling for Knapsack

### Edge Cases (`test_edge_cases.py`)
- ✓ Extreme dimensions (d=1, d=100, d=1000)
- ✓ Zero capacity Knapsack
- ✓ Empty items
- ✓ All items too heavy
- ✓ Single item cases
- ✓ Boundary parameter values
- ✓ Invalid inputs (negative dimensions, invalid parameters)
- ✓ Numerical stability

### Parallel Execution (`test_parallel_execution.py`)
- ✓ Multiprocessing correctness
- ✓ Reproducibility with seeds
- ✓ Performance scaling

## Testing Visualizations

While `src/utils/visualization.py` doesn't have automated tests, you can verify plots manually:

```bash
# Run demo script to generate all plots
python demo.py

# Check results/ folder for generated images
ls -la results/
```

Expected output files:
- `fa_rastrigin_convergence.png`
- `fa_rastrigin_trajectory.png`
- `fa_knapsack_convergence.png`
- `algorithm_comparison.png`
- `algorithm_comparison_log.png`
- `parameter_sensitivity_gamma.png`

## Adding New Tests

To add tests for a new component:

1. Create a new test file: `test_your_component.py`
2. Import unittest and your component
3. Create test class inheriting from `unittest.TestCase`
4. Write test methods (must start with `test_`)
5. Run tests to verify

Example:
```python
import unittest
from src.your_module import YourClass

class TestYourClass(unittest.TestCase):
    def setUp(self):
        self.obj = YourClass()
    
    def test_something(self):
        result = self.obj.method()
        self.assertEqual(result, expected_value)
```

## Test Best Practices

1. **Isolation**: Each test should be independent
2. **Determinism**: Use seeds for reproducible results
3. **Coverage**: Test normal cases, edge cases, and error cases
4. **Clarity**: Use descriptive test names
5. **Speed**: Keep tests fast (use small problem sizes)

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: python test/run_all_tests.py
```

## Troubleshooting

**Import errors**: Make sure you're running from the project root or have the correct PYTHONPATH set.

**Missing dependencies**: Install required packages:
```bash
pip install -r requirements.txt
# Or use conda
conda env create -f environment.yml
```

**Matplotlib backend errors**: If plots don't display, try:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

**Failed tests**: Check the error message and traceback for details. Ensure all source files are present and correct.
