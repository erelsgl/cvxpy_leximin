#!/usr/bin/env python3
"""
Benchmark script for comparing different leximin/leximax solving methods.

This script compares the runtime and solution quality of three different methods:
1. _solve_leximin (Willson's algorithm)
2. _solve_leximin_ogry_relax (Ogryczak & Śliwiński's algorithm without integer variables)
3. _solve_leximin_ogry_integer_variables (Ogryczak & Śliwiński's algorithm with integer variables)

The script creates resource allocation test instances, solves each with all three methods
for both leximin and leximax objectives, verifies consistent results, and measures
performance.
"""

import time
import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Tuple, List

# Import necessary functions from the module
from cvxpy_leximin.objective import Leximin, Leximax

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store the original solving method
original_solve_method = None


def save_original_method():
    """Save the original _solve_leximin method to restore it later"""
    global original_solve_method
    from cvxpy.problems.problem import Problem
    original_solve_method = Problem._solve_leximin


def restore_original_method():
    """Restore the original _solve_leximin method"""
    from cvxpy.problems.problem import Problem
    if original_solve_method is not None:
        Problem._solve_leximin = original_solve_method


def set_solve_method(method_name: str):
    """Set the solving method to use"""
    from cvxpy.problems.problem import Problem

    method_mapping = {
        "willson": "_solve_leximin_will",
        "ogry_relax": "_solve_leximin_ogry_relax",
        "ogry_integer": "_solve_leximin_ogry_integer_variables"
    }

    if method_name not in method_mapping:
        raise ValueError(f"Unknown method: {method_name}")

    from problem import _solve_leximin_will, _solve_leximin_ogry_relax, _solve_leximin_ogry_integer_variables

    method_funcs = {
        "willson": _solve_leximin_will,
        "ogry_relax": _solve_leximin_ogry_relax,
        "ogry_integer": _solve_leximin_ogry_integer_variables
    }

    Problem._solve_leximin = method_funcs[method_name]


def create_resource_allocation_problem(n_resources: int, n_agents: int,
                                       objective_type: str = "leximin") -> Tuple[cp.Problem, List[cp.Expression]]:
    """Create a resource allocation problem with n_resources and n_agents.

    Parameters:
        n_resources: Number of resources to allocate
        n_agents: Number of agents among which to allocate resources
        objective_type: 'leximin' or 'leximax'

    Returns:
        problem: The created CVXPY Problem
        utilities: List of agent utility expressions
    """
    # Create a random valuation matrix where each agent values each resource
    instance_seed = 42 + n_resources + n_agents
    np.random.seed(instance_seed)  # For reproducibility with different problem sizes
    valuations = np.random.randint(1, 10, size=(n_agents, n_resources))

    logger.info(
        f"Created problem instance: {objective_type}, {n_resources} resources, {n_agents} agents, seed {instance_seed}")
    logger.debug(f"Valuations matrix:\n{valuations}")

    # Allocation variables - each a[i][j] is the fraction of resource j given to agent i
    allocations = [cp.Variable(n_resources) for _ in range(n_agents)]

    # Feasibility constraints
    constraints = []

    # Each allocation variable must be between 0 and 1
    for agent_alloc in allocations:
        constraints.extend([0 <= x for x in agent_alloc])
        constraints.extend([x <= 1 for x in agent_alloc])

    # The sum of allocations for each resource must be at most 1
    for j in range(n_resources):
        constraints.append(sum(alloc[j] for alloc in allocations) <= 1)

    # Calculate utilities for each agent based on their valuations
    utilities = [cp.sum(cp.multiply(valuations[i], allocations[i])) for i in range(n_agents)]

    # Create the problem with the appropriate objective
    objective = Leximin(utilities) if objective_type.lower() == "leximin" else Leximax(utilities)
    problem = cp.Problem(objective, constraints)

    return problem, utilities


def benchmark_methods(sizes, methods, objective_types=None, repeats=3):
    """Benchmark the different methods on various problem instances.

    Parameters:
        sizes: List of resource counts to test
        methods: List of methods to test
        objective_types: List of objective types to test
        repeats: Number of times to repeat each test for robust timing

    Returns:
        results: DataFrame with benchmark results
    """
    # Save the original method to restore later
    if objective_types is None:
        objective_types = ["leximin"]
    save_original_method()

    results = []

    for objective_type in objective_types:
        for size in sizes:
            logger.info(f"Testing {objective_type} with {size} resources...")

            # For resource allocation, use a fixed ratio of resources to agents
            n_agents = max(2, size // 3)

            # Create the problem once
            problem, _ = create_resource_allocation_problem(size, n_agents, objective_type=objective_type)

            # Store the solutions for each method
            solutions = {}

            # Test each method
            for method_name in methods:
                logger.info(f"  Using {method_name} method...")

                # Set the method to use
                set_solve_method(method_name)

                # Time the method
                times = []
                success = False
                solution = None

                for i in range(repeats):
                    start_time = time.time()
                    try:
                        solution = problem.solve()
                        success = True
                        elapsed = time.time() - start_time
                        times.append(elapsed)
                        logger.info(f"    Run {i + 1}/{repeats}: {elapsed:.3f} seconds")
                    except Exception as e:
                        logger.error(f"    Error with {method_name} on {objective_type} size {size}: {e}")
                        elapsed = float('inf')
                        times.append(elapsed)

                # Store the solution and timing results
                solutions[method_name] = solution
                avg_time = np.mean(times) if times else float('inf')

                # Add the result
                results.append({
                    "objective_type": objective_type,
                    "resources": size,
                    "agents": n_agents,
                    "method": method_name,
                    "time": avg_time,
                    "success": success
                })

            # Verify that all methods produce the same result
            if all(success for _, success in [(m, solutions[m] is not None) for m in methods]):
                # Convert all solutions to rounded values for comparison
                try:
                    rounded_solutions = {
                        method: round(float(solution), 6) if not isinstance(solution, (list, np.ndarray))
                        else [round(float(val), 6) for val in solution]
                        for method, solution in solutions.items()
                    }

                    # Convert lists to tuples for hashability when using set()
                    hashable_solutions = {
                        method: tuple(sol) if isinstance(sol, list) else sol
                        for method, sol in rounded_solutions.items()
                    }

                    # Check if solutions are different
                    unique_solutions = set(map(str, hashable_solutions.values()))
                    if len(unique_solutions) > 1:
                        logger.warning(f"Different solutions for {objective_type} with {size} resources:")
                        for method_name, solution in rounded_solutions.items():
                            logger.warning(f"  {method_name}: {solution}")
                    else:
                        logger.info(f"  All methods produced consistent results")

                except (ValueError, TypeError) as e:
                    logger.error(f"  Error comparing solutions: {e}")

    # Restore the original method
    restore_original_method()

    # Convert to DataFrame
    return pd.DataFrame(results)


def plot_results(results):
    """Plot the benchmark results."""
    objective_types = sorted(results["objective_type"].unique())
    fig, axes = plt.subplots(1, len(objective_types), figsize=(12, 6), sharey=True)

    if len(objective_types) == 1:
        axes = [axes]

    for i, obj_type in enumerate(objective_types):
        obj_results = results[results["objective_type"] == obj_type]

        for method in sorted(obj_results["method"].unique()):
            method_results = obj_results[obj_results["method"] == method]
            method_results = method_results.sort_values("resources")
            axes[i].plot(method_results["resources"], method_results["time"], 'o-', label=method)

        axes[i].set_title(f"{obj_type.capitalize()}")
        axes[i].set_xlabel("Number of Resources")
        axes[i].set_ylabel("Time (seconds)" if i == 0 else "")
        axes[i].set_yscale("log")
        axes[i].grid(True, which="both", linestyle="--", linewidth=0.5)
        axes[i].legend()

    plt.suptitle("Performance Comparison of Leximin/Leximax Methods")
    plt.tight_layout()
    plt.savefig("leximin_leximax_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("Plot saved to leximin_leximax_comparison.png")
    plt.close()


def main():
    """Main function to run the benchmark."""
    # Define the methods to test
    methods = ["willson", "ogry_relax", "ogry_integer"]

    # Define the objective types to test
    objective_types = ["leximin"]

    # Define the sizes (number of resources) to test
    sizes = [10, 20, 30]

    # Run the benchmark
    logger.info("Starting benchmark...")
    results = benchmark_methods(sizes, methods, objective_types, repeats=3)

    # Print the results
    logger.info("\nBenchmark Results:")
    for obj_type in results["objective_type"].unique():
        logger.info(f"\n{obj_type.upper()}:")
        obj_results = results[results["objective_type"] == obj_type]
        pivot = obj_results.pivot(index="resources", columns="method", values="time")
        logger.info(f"\n{pivot}")

    # Plot the results
    plot_results(results)

    # Save the results to CSV
    csv_filename = "leximin_leximax_benchmark.csv"
    results.to_csv(csv_filename, index=False)
    logger.info(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    main()