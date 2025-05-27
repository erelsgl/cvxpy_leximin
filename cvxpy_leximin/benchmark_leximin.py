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

import cvxpy
import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Tuple, List

from cvxpy import Variable, Problem
from scipy.spatial.distance import cityblock

# Import necessary functions from the module
from cvxpy_leximin.objective import Leximin, Leximax

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def create_clients_facility_locations(size, p):
    # Fixed client coordinates on 2D grid 
    np.random.seed(42)
    possible_coords = [(x, y) for x in range(0, 101, 5) for y in range(0, 101, 5)]
    
    # Choose random indices instead of trying to choose from the coordinates directly
    indices = np.random.choice(len(possible_coords), size=size, replace=False)
    selected_coords = [possible_coords[i] for i in indices]
    coords = np.array(sorted(selected_coords))

    # Compute Manhattan distances
    d = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            d[i, j] = cityblock(coords[i], coords[j])

    # Decision variables
    x = Variable(size, boolean=True)  # facility open
    x_prime = Variable((size, size), boolean=True)  # assignment

    # Constraints
    constraints = []

    # Each client must be assigned to one facility
    for j in range(size):
        constraints.append(cvxpy.sum(x_prime[:, j]) == 1)

    # Only assign to open facilities
    for i in range(size):
        for j in range(size):
            constraints.append(x_prime[i, j] <= x[i])

    # Limit number of facilities
    constraints.append(cvxpy.sum(x) == p)

    # Client outcomes: distance to assigned facility
    outcomes = []
    for j in range(size):
        outcomes.append(cvxpy.sum([d[i, j] * x_prime[i, j] for i in range(size)]))

    # Create problem
    objective = Leximax(outcomes)
    problem = Problem(objective, constraints)

    # Estimate possible outcome values
    possible_values = sorted(set(d.flatten()), reverse=True)

    return problem, possible_values, outcomes


def benchmark_methods(sizes, methods, objective_types=None, facilities=None, repeats=1):
    """
    Run benchmark tests on a set of methods for solving client-facility location problems,
    measuring performance and consistency under various configurations.

    Tests multiple methods on client-facility location problems with varying
    sizes, facility counts, and objective types. Measures execution time
    and validates the consistency of computed solutions for each method. The
    results include details about the performance and success of each method.

    Parameters:
    sizes : list[int]
        A list of integers representing problem sizes (number of clients or resources).
    methods : list[str]
        A list of method names to be benchmarked.
    objective_types : list[str], optional
        A list of objective type names to be tested. Defaults to ["leximax"].
    facilities : list[int], optional
        A list of integers representing the number of facilities available for allocation.
        Defaults to None.
    repeats : int, optional
        The number of times each method is repeated for measuring performance. Defaults to 3.

    Returns:
    pandas.DataFrame
        A DataFrame containing the benchmarking results with columns for objective type,
        problem size, number of facilities, method name, average execution time, and
        success status.

    Raises:
    KeyError
        If a required key is missing from the results or during validation.
    TypeError
        If an invalid type is encountered during solution comparison.
    ValueError
        If an error occurs during result rounding or comparison of solutions.
    """
    # Save the original method to restore later
    if objective_types is None:
        objective_types = ["leximin"]

    if facilities is None:
        facilities = [2]
    results = []

    for objective_type in objective_types:
        for size in sizes:
            for facility in facilities:
                logger.info(f"Testing {objective_type} with {size} resources with {facility} facilities...")
    
                problem, outcome_levels, utilities = create_clients_facility_locations(size, facility)
    
                # Store the solutions for each method
                solutions = {}
    
                # Test each method
                for method_name in methods:
                    logger.info(f"  Using {method_name} method...")
    
                    # Time the method
                    times = []
                    success = False
                    solution = None
    
                    for i in range(repeats):
                        start_time = time.time()
                        try:
                            if method_name == "ordered_values":
                                solution = problem.solve(method=method_name, outcome_levels=outcome_levels)
                            else:
                                solution = problem.solve(method=method_name)
                            success = True
                            elapsed = time.time() - start_time
                            times.append(elapsed)
                            logger.info(f"    Run {i + 1}/{repeats}: {elapsed:.3f} seconds")
                        except Exception as e:
                            logger.error(f"    Error with {method_name} on {objective_type} size {size} facility {facility}: {e}")
                            elapsed = float('inf')
                            times.append(elapsed)
    
                    # Store the solution and timing results
                    solutions[method_name] = solution
                    avg_time = np.mean(times) if times else float('inf')
    
                    # Add the result
                    results.append({
                        "objective_type": objective_type,
                        "n": size,
                        "p": facility,
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
                            logger.warning(f"  {size} resources, {facility} facilities, {objective_type} objective")

                            for method_name, solution in rounded_solutions.items():
                                logger.warning(f"  {method_name}: {solution}")
                                logger.warning(f"Sum of the solutions: {sum(solution)}")
                        else:
                            logger.info(f"  All methods produced consistent results")
    
                    except (ValueError, TypeError) as e:
                        logger.error(f"  Error comparing solutions: {e}")

    # Convert to DataFrame
    return pd.DataFrame(results)


def plot_results(results):
    """Plot the benchmark results with separate subplots for different facility counts."""
    objective_types = sorted(results["objective_type"].unique())
    facilities = sorted(results["p"].unique())

    # Define method name mapping
    method_name_mapping = {
        "ogry_integer": "Ordered Outcomes 1",
        "ogry_relax": "Ordered Outcomes 2",
        "ordered_values": "Ordered Values",
        "willson": "Willson"
    }

    fig, axes = plt.subplots(len(objective_types), len(facilities),
                             figsize=(4 * len(facilities), 4 * len(objective_types)),
                             squeeze=False)

    for i, obj_type in enumerate(objective_types):
        for j, facility_count in enumerate(facilities):
            ax = axes[i, j]

            # Filter results for this objective type and facility count
            filtered_results = results[
                (results["objective_type"] == obj_type) &
                (results["p"] == facility_count)
                ]

            # Plot each method
            for method in sorted(filtered_results["method"].unique()):
                method_results = filtered_results[filtered_results["method"] == method]
                method_results = method_results.sort_values("n")

                # Only plot if we have successful results
                successful_results = method_results[method_results["success"] == True]
                if not successful_results.empty:
                    # Use mapped name if available, otherwise use original name
                    display_name = method_name_mapping.get(method, method)
                    ax.plot(successful_results["n"], successful_results["time"],
                            'o-', label=display_name, linewidth=2, markersize=6)

            ax.set_title(f"{obj_type.capitalize()}\n{facility_count} facilities")
            ax.set_xlabel("Number of clients")
            ax.set_ylabel("Time (seconds)")
            ax.set_yscale("log")
            ax.set_ylim(bottom=0, top=1e5/5)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.legend()

    plt.suptitle("Performance Comparison of Leximin/Leximax Methods", fontsize=16)
    plt.tight_layout()
    plt.savefig("facilities_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("Plot saved to facilities_comparison.png")
    plt.close()


def plot_results_heatmap(results):
    """Create heatmap plots showing performance across n and p dimensions."""
    objective_types = sorted(results["objective_type"].unique())
    methods = sorted(results["method"].unique())
    # Define method name mapping
    method_name_mapping = {
        "ogry_integer": "Ordered Outcomes 1",
        "ogry_relax": "Ordered Outcomes 2",
        "ordered_values": "Ordered Values",
        "willson": "Willson"
    }
    fig, axes = plt.subplots(len(objective_types), len(methods),
                             figsize=(4 * len(methods), 4 * len(objective_types)),
                             squeeze=False)

    for i, obj_type in enumerate(objective_types):
        for j, method in enumerate(methods):
            ax = axes[i, j]

            # Filter results for this objective type and method
            filtered_results = results[
                (results["objective_type"] == obj_type) &
                (results["method"] == method) &
                (results["success"] == True)
                ]

            if not filtered_results.empty:
                # Create pivot table for heatmap
                pivot_data = filtered_results.pivot(index="p", columns="n", values="time")

                # Create heatmap
                im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')

                # Set ticks and labels
                ax.set_xticks(range(len(pivot_data.columns)))
                ax.set_xticklabels(pivot_data.columns)
                ax.set_yticks(range(len(pivot_data.index)))
                ax.set_yticklabels(pivot_data.index)

                # Add colorbar
                plt.colorbar(im, ax=ax, label="Time (seconds)")

                # Add text annotations
                for row in range(len(pivot_data.index)):
                    for col in range(len(pivot_data.columns)):
                        value = pivot_data.iloc[row, col]
                        if not np.isnan(value):
                            ax.text(col, row, f'{value:.3f}',
                                    ha='center', va='center',
                                    color='white' if value > pivot_data.values.max() / 2 else 'black')

            ax.set_title(f"{obj_type.capitalize()} - {method_name_mapping[method]}")
            ax.set_xlabel("Number of clients (n)")
            ax.set_ylabel("Number of facilities (p)")

    plt.suptitle("Performance Heatmaps: Time vs Problem Size", fontsize=16)
    plt.tight_layout()
    plt.savefig("facilities_heatmap.png", dpi=300, bbox_inches="tight")
    logger.info("Heatmap saved to facilities_heatmap.png")
    plt.close()


def load_and_plot_results(csv_filename="facilities_benchmark.csv"):
    """Load results from CSV file and generate plots."""
    try:
        results = pd.read_csv(csv_filename)
        logger.info(f"Loaded results from {csv_filename}")
        logger.info(f"Data shape: {results.shape}")
        logger.info(f"Columns: {list(results.columns)}")

        # Generate both types of plots
        # plot_results(results)
        plot_results_heatmap(results)

        # Print summary statistics
        logger.info("\nSummary by method:")
        summary = results.groupby(['objective_type', 'method']).agg({
            'time': ['mean', 'std', 'min', 'max'],
            'success': 'sum'
        }).round(4)
        logger.info(f"\n{summary}")

        return results

    except FileNotFoundError:
        logger.error(f"File {csv_filename} not found. Please run the benchmark first.")
        return None
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return None


def main():
    """Main function to run the benchmark."""
    # Define the methods to test
    methods = ["willson", "ogry_relax", "ogry_integer", "ordered_values"]

    # Define the objective types to test
    objective_types = ["leximin"]

    # Define the sizes to test
    sizes = [10, 25, 40, 55, 70]
    facilities = [2, 3, 5]

    # Run the benchmark
    logger.info("Starting benchmark...")
    results = benchmark_methods(sizes, methods, objective_types, facilities=facilities, repeats=1)

    # Print the results
    logger.info("\nBenchmark Results:")
    for obj_type in results["objective_type"].unique():
        logger.info(f"\n{obj_type.upper()}:")
        obj_results = results[results["objective_type"] == obj_type]
        pivot = obj_results.pivot(index=["n", "p"], columns="method", values="time")
        logger.info(f"\n{pivot}")

    # Plot the results
    plot_results(results)

    # Save the results to CSV
    csv_filename = "facilities_benchmark.csv"
    results.to_csv(csv_filename, index=False)
    logger.info(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    # main()
    load_and_plot_results("facilities_benchmark_1.csv")