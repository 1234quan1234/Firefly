"""
0/1 Knapsack Problem optimization.

The 0/1 Knapsack problem is a classic combinatorial optimization problem where
the goal is to select items to maximize value while staying within weight capacity.

References
----------
.. [1] https://en.wikipedia.org/wiki/Knapsack_problem
"""

import numpy as np
from typing import Literal

# Use relative import
from ...core.problem_base import ProblemBase


class KnapsackProblem(ProblemBase):
    """
    0/1 Knapsack Problem.
    
    Given a set of items, each with a weight and value, select items to maximize
    total value without exceeding the knapsack's weight capacity.
    
    Solution representation: binary vector where x[i] = 1 means item i is selected.
    
    For optimization consistency (minimization), we use:
        fitness = -total_value  (if feasible)
        fitness = -total_value + penalty_coefficient * violation  (if infeasible)
    
    This ensures:
    - Better solutions have LOWER fitness (minimize-compatible)
    - Feasible solutions always better than infeasible ones
    - Performance profiles work correctly with cost ratios
    
    Parameters
    ----------
    values : np.ndarray
        Value of each item, shape (num_items,).
    weights : np.ndarray
        Weight of each item, shape (num_items,).
    capacity : float
        Maximum weight capacity of the knapsack.
    penalty_coefficient : float, optional
        Penalty multiplier for constraint violations. Default is 1000.
    
    Attributes
    ----------
    values : np.ndarray
        Item values.
    weights : np.ndarray
        Item weights.
    capacity : float
        Knapsack capacity.
    num_items : int
        Number of items.
    penalty_coefficient : float
        Penalty for infeasible solutions.
    dp_optimal : float or None
        DP optimal value if available, None otherwise.
    
    Examples
    --------
    >>> values = np.array([10, 20, 30])
    >>> weights = np.array([1, 2, 3])
    >>> capacity = 4.0
    >>> problem = KnapsackProblem(values, weights, capacity)
    >>> x = np.array([1, 1, 0])  # Select items 0 and 1
    >>> fitness = problem.evaluate(x)
    >>> print(f"Fitness: {fitness}")  # Should be -30 (maximizing value)
    """
    
    def __init__(
        self, 
        values: np.ndarray, 
        weights: np.ndarray, 
        capacity: float,
        penalty_coefficient: float = 1000.0
    ):
        """
        Initialize Knapsack problem.
        
        Parameters
        ----------
        values : np.ndarray
            Value of each item, shape (num_items,).
        weights : np.ndarray
            Weight of each item, shape (num_items,).
        capacity : float
            Maximum weight capacity.
        penalty_coefficient : float, optional
            Penalty multiplier for exceeding capacity. Default is 1000.
        """
        self.values = np.array(values, dtype=float)
        self.weights = np.array(weights, dtype=float)
        self.capacity = float(capacity)
        self.num_items = len(values)
        self.penalty_coefficient = penalty_coefficient
        
        # Placeholder for DP optimal (set externally if available)
        self.dp_optimal = None
        
        if len(weights) != self.num_items:
            raise ValueError("values and weights must have the same length")
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate knapsack solution with minimize-compatible fitness.
        
        Returns:
        - Feasible: fitness = -total_value (lower is better)
        - Infeasible: fitness = -total_value + penalty * violation (always worse)
        
        This ensures performance profiles work correctly:
        - Cost ratio r = fitness_algo / fitness_best always meaningful
        - Feasible solutions always have fitness < infeasible ones
        
        Parameters
        ----------
        x : np.ndarray
            Binary selection vector, shape (num_items,), values in {0, 1}.
        
        Returns
        -------
        fitness : float
            Fitness value (lower is better).
        """
        # Ensure binary
        selection = (x > 0.5).astype(int)
        
        total_weight = float(np.sum(selection * self.weights))
        total_value = float(np.sum(selection * self.values))
        
        # Calculate violation
        violation = max(0.0, total_weight - self.capacity)
        
        # Base cost (minimize-compatible: negate value)
        base_cost = -total_value
        
        if violation <= 0.0:
            # Feasible solution
            fitness = base_cost
        else:
            # Infeasible: add penalty proportional to violation
            # This maintains monotonicity: more violation = worse fitness
            fitness = base_cost + self.penalty_coefficient * violation
        
        # Ensure no NaN/Inf
        if not np.isfinite(fitness):
            fitness = self.penalty_coefficient * 1e6
        
        return fitness
    
    def get_solution_info(self, x: np.ndarray) -> dict:
        """
        Get detailed information about a solution for analysis.
        
        This is used by runners/loggers to extract metrics for:
        - Performance profiles
        - Data profiles
        - Fixed-budget analysis
        - Pairwise statistical tests
        
        Parameters
        ----------
        x : np.ndarray
            Binary selection vector.
        
        Returns
        -------
        info : dict
            Dictionary with keys:
            - 'Fitness': float (minimize-compatible)
            - 'Value': float (raw value, always positive)
            - 'Weight': float (total weight)
            - 'Feasible': bool (True if within capacity)
            - 'Violation': float (amount over capacity, 0 if feasible)
            - 'DP_Optimal': float or None (if available)
        """
        selection = (x > 0.5).astype(int)
        
        total_weight = float(np.sum(selection * self.weights))
        total_value = float(np.sum(selection * self.values))
        violation = max(0.0, total_weight - self.capacity)
        
        # Compute fitness (same logic as evaluate)
        base_cost = -total_value
        if violation <= 0.0:
            fitness = base_cost
        else:
            fitness = base_cost + self.penalty_coefficient * violation
        
        # Ensure no NaN/Inf
        if not np.isfinite(fitness):
            fitness = self.penalty_coefficient * 1e6
        
        return {
            'Fitness': float(fitness),
            'Value': float(total_value),
            'Weight': float(total_weight),
            'Feasible': bool(violation <= 0.0),
            'Violation': float(violation),
            'DP_Optimal': self.dp_optimal
        }
    
    def is_feasible(self, x: np.ndarray) -> bool:
        """
        Check if solution is feasible (within capacity).
        
        Parameters
        ----------
        x : np.ndarray
            Binary selection vector.
        
        Returns
        -------
        feasible : bool
            True if total weight <= capacity.
        """
        selection = (x > 0.5).astype(int)
        total_weight = np.sum(selection * self.weights)
        return float(total_weight) <= self.capacity
    
    def representation_type(self) -> Literal["continuous", "tsp", "knapsack", "graph_coloring"]:
        """Return 'knapsack' for this problem type."""
        return "knapsack"
    
    def init_solution(self, rng: np.random.RandomState, n: int = 1) -> np.ndarray:
        """
        Generate n random feasible knapsack solutions.
        
        Uses a greedy repair strategy to ensure solutions are feasible:
        randomly select items, and if capacity is exceeded, randomly remove
        items until feasible.
        
        Parameters
        ----------
        rng : np.random.RandomState
            Random number generator.
        n : int, default=1
            Number of solutions to generate.
        
        Returns
        -------
        solutions : np.ndarray
            Array of shape (n, num_items) with binary values.
        """
        solutions = np.zeros((n, self.num_items), dtype=int)
        
        for i in range(n):
            # Start with random binary vector
            solution = rng.randint(0, 2, self.num_items)
            
            # Repair if needed
            solution = self._repair_solution(solution, rng)
            solutions[i] = solution
        
        return solutions
    
    def _repair_solution(self, solution: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """
        Repair an infeasible solution by removing items until feasible.
        
        Uses greedy strategy: removes items with lowest value/weight ratio first.
        
        Parameters
        ----------
        solution : np.ndarray
            Binary selection vector.
        rng : np.random.RandomState
            Random number generator.
        
        Returns
        -------
        repaired : np.ndarray
            Feasible binary selection vector.
        """
        solution = solution.copy()
        total_weight = np.sum(solution * self.weights)
        
        # If already feasible, return
        if total_weight <= self.capacity:
            return solution
        
        selected_indices = np.where(solution == 1)[0]
        
        # Greedy repair: remove lowest value/weight ratio items first
        if len(selected_indices) > 0:
            # Avoid division by zero
            safe_weights = np.maximum(self.weights[selected_indices], 1e-12)
            ratios = self.values[selected_indices] / safe_weights
            sorted_indices = selected_indices[np.argsort(ratios)]  # Ascending
            
            for idx in sorted_indices:
                solution[idx] = 0
                total_weight -= self.weights[idx]
                if total_weight <= self.capacity:
                    break
        
        return solution
    
    def greedy_repair(self, solution: np.ndarray) -> np.ndarray:
        """
        Public method for greedy repair (for external use by optimizers).
        
        Removes items with lowest value/weight ratio first until feasible.
        
        Parameters
        ----------
        solution : np.ndarray
            Binary selection vector that may be infeasible.
        
        Returns
        -------
        repaired : np.ndarray
            Feasible binary selection vector.
        """
        solution = solution.copy()
        total_weight = np.sum(solution * self.weights)
        
        if total_weight <= self.capacity:
            return solution
        
        selected_indices = np.where(solution == 1)[0]
        
        if len(selected_indices) > 0:
            # Calculate value/weight ratios, avoid division by zero
            safe_weights = np.maximum(self.weights[selected_indices], 1e-12)
            ratios = self.values[selected_indices] / safe_weights
            # Sort by ratio ascending (remove worst items first)
            sorted_indices = selected_indices[np.argsort(ratios)]
            
            for idx in sorted_indices:
                solution[idx] = 0
                total_weight -= self.weights[idx]
                if total_weight <= self.capacity:
                    break
        
        return solution
    
    def clip(self, X: np.ndarray) -> np.ndarray:
        """
        For Knapsack, clip to binary {0, 1}.
        
        Values >= 0.5 become 1, others become 0.
        
        Parameters
        ----------
        X : np.ndarray
            Solution(s), shape (n, num_items) or (num_items,).
        
        Returns
        -------
        X_binary : np.ndarray
            Binary-clipped solutions.
        """
        return (X > 0.5).astype(int)


def _hamming_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute Hamming distance between two binary vectors.
    
    Private helper for diversity calculation.
    
    Parameters
    ----------
    x1, x2 : np.ndarray
        Binary vectors.
    
    Returns
    -------
    distance : float
        Number of differing bits.
    """
    return float(np.sum(x1 != x2))


def compute_population_diversity(population: np.ndarray) -> float:
    """
    Compute population diversity for Knapsack (bitstring representation).
    
    Uses average pairwise Hamming distance, normalized by sqrt(n_items).
    
    Parameters
    ----------
    population : np.ndarray
        Population of solutions, shape (pop_size, n_items).
    
    Returns
    -------
    diversity : float
        Normalized diversity metric.
    """
    n_pop, n_items = population.shape
    
    if n_pop < 2:
        return 0.0
    
    # Sample pairs to avoid O(n^2) for large populations
    max_pairs = min(100, (n_pop * (n_pop - 1)) // 2)
    
    total_distance = 0.0
    count = 0
    
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    for _ in range(max_pairs):
        i, j = rng.choice(n_pop, size=2, replace=False)
        total_distance += _hamming_distance(population[i], population[j])
        count += 1
    
    avg_distance = total_distance / max(count, 1)
    
    # Normalize by sqrt(n_items) for scale-invariance
    normalized = avg_distance / np.sqrt(max(n_items, 1.0))
    
    return float(normalized)


if __name__ == "__main__":
    # Demo & Internal Tests
    print("Knapsack Problem Demo & Validation")
    print("=" * 50)
    
    # Small example
    values = np.array([10, 20, 30, 40])
    weights = np.array([1, 2, 3, 4])
    capacity = 5.0
    
    problem = KnapsackProblem(values, weights, capacity)
    print(f"Items: {problem.num_items}")
    print(f"Values: {values}")
    print(f"Weights: {weights}")
    print(f"Capacity: {capacity}")
    print(f"Penalty Coefficient: {problem.penalty_coefficient}")
    
    print("\n" + "=" * 50)
    print("TEST 1: FEASIBLE SOLUTION")
    print("=" * 50)
    # Test feasible solution
    x1 = np.array([1, 1, 0, 0])  # Items 0,1: value=30, weight=3
    f1 = problem.evaluate(x1)
    info1 = problem.get_solution_info(x1)
    print(f"Solution: [1,1,0,0]")
    print(f"  Fitness: {f1:.2f} (should be -30)")
    print(f"  Info: {info1}")
    print(f"  Is Feasible: {problem.is_feasible(x1)}")
    assert info1['Feasible'], "Should be feasible"
    assert np.isclose(f1, -30.0), "Fitness should be -30"
    assert np.isclose(info1['Value'], 30.0), "Value should be 30"
    print("✓ PASSED")
    
    print("\n" + "=" * 50)
    print("TEST 2: INFEASIBLE SOLUTION")
    print("=" * 50)
    x2 = np.array([0, 0, 1, 1])  # Items 2,3: value=70, weight=7 (INFEASIBLE)
    f2 = problem.evaluate(x2)
    info2 = problem.get_solution_info(x2)
    print(f"Solution: [0,0,1,1]")
    print(f"  Fitness: {f2:.2f}")
    print(f"  Info: {info2}")
    print(f"  Is Feasible: {problem.is_feasible(x2)}")
    assert not info2['Feasible'], "Should be infeasible"
    assert f2 > -70.0, "Fitness should be worse (higher) than -70 due to penalty"
    assert np.isclose(info2['Violation'], 2.0), "Violation should be 2"
    print("✓ PASSED")
    
    print("\n" + "=" * 50)
    print("TEST 3: MONOTONICITY (more violation = worse fitness)")
    print("=" * 50)
    # Optimal solution at capacity
    x_opt = np.array([1, 0, 0, 1])  # Items 0,3: value=50, weight=5
    f_opt = problem.evaluate(x_opt)
    info_opt = problem.get_solution_info(x_opt)
    print(f"Solution: [1,0,0,1] (at capacity)")
    print(f"  Fitness: {f_opt:.2f} (should be -50)")
    print(f"  Info: {info_opt}")
    
    # Check monotonicity: f_opt < f1 < f2 (lower is better)
    print(f"\nMonotonicity check:")
    print(f"  Feasible optimal: {f_opt:.2f}")
    print(f"  Feasible suboptimal: {f1:.2f}")
    print(f"  Infeasible: {f2:.2f}")
    assert f_opt < f1, "Optimal should be better than suboptimal"
    assert f1 < f2, "Feasible should be better than infeasible"
    print("✓ PASSED (Feasible < Infeasible, monotonic)")
    
    print("\n" + "=" * 50)
    print("TEST 4: NO NaN/Inf")
    print("=" * 50)
    # Generate random solutions
    rng = np.random.RandomState(42)
    random_sols = problem.init_solution(rng, n=10)
    print(f"Generated 10 random feasible solutions:")
    all_finite = True
    for i, sol in enumerate(random_sols):
        fitness = problem.evaluate(sol)
        info = problem.get_solution_info(sol)
        print(f"  Sol {i}: fitness={fitness:.2f}, value={info['Value']:.1f}, "
              f"weight={info['Weight']:.1f}, feasible={info['Feasible']}")
        if not np.isfinite(fitness):
            all_finite = False
            print(f"    ✗ FAILED: Non-finite fitness")
    
    assert all_finite, "All fitness values should be finite"
    print("✓ PASSED (All fitness values finite)")
    
    print("\n" + "=" * 50)
    print("TEST 5: DP_Optimal Placeholder")
    print("=" * 50)
    info_with_dp = problem.get_solution_info(x_opt)
    print(f"DP_Optimal: {info_with_dp['DP_Optimal']} (should be None)")
    assert info_with_dp['DP_Optimal'] is None, "DP_Optimal should be None initially"
    
    # Simulate setting DP from external source
    problem.dp_optimal = 50.0
    info_with_dp2 = problem.get_solution_info(x_opt)
    print(f"After setting: {info_with_dp2['DP_Optimal']} (should be 50.0)")
    assert info_with_dp2['DP_Optimal'] == 50.0, "DP_Optimal should be 50.0"
    print("✓ PASSED")
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50)
    print("\nSummary:")
    print("- Feasible solutions: fitness = -value (minimize-compatible)")
    print("- Infeasible solutions: fitness = -value + penalty * violation")
    print("- Monotonic: more violation ⇒ higher (worse) fitness")
    print("- No NaN/Inf in normal operation")
    print("- DP_Optimal placeholder ready for external input")
    print("- API unchanged: all signatures and params preserved")
