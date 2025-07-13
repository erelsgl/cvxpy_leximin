"""
Benchmark script for comparing different leximin/leximax solving methods using experiments_csv.

This script compares the runtime and solution quality of different methods:
1. _solve_leximin_saturation (Willson's saturation-based algorithm)
2. _solve_leximin_ordered_outcomes (Ogryczak & Śliwiński's algorithm without integer variables)
3. _solve_leximin_ordered_outcomes_big_M (Ogryczak & Śliwiński's algorithm with integer variables)

The script uses experiments_csv to track experiment progress and handle restarts gracefully.

REQUIREMENTS: cvxpy_leximin, pandas, matplotlib, experiments_csv

Programmer: Moshe Ofer
Since: 2025-06
"""

import time
import argparse
import logging
from typing import Tuple, List, Dict, Any

import cvxpy
from cvxpy import Variable, Problem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import experiments_csv

from scipy.spatial.distance import cityblock
from cvxpy_leximin.objective import Leximin, Leximax

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_resource_allocation_problem(n_resources: int, n_agents: int,
                                       objective_type: str = "leximin") -> Tuple[cvxpy.Problem, List[float], List[cvxpy.Expression]]:
    """Create a resource allocation problem with n_resources and n_agents."""
    # Create a random valuation matrix where each agent values each resource
    instance_seed = 42 + n_resources + n_agents
    np.random.seed(instance_seed)  # For reproducibility with different problem sizes
    valuations = np.random.randint(1, 10, size=(n_agents, n_resources))

    logger.debug(f"Created problem instance: {objective_type}, {n_resources} resources, {n_agents} agents, seed {instance_seed}")

    # Allocation variables - each a[i][j] is the fraction of resource j given to agent i
    allocations = [cvxpy.Variable(n_resources) for _ in range(n_agents)]

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
    utilities = [cvxpy.sum(cvxpy.multiply(valuations[i], allocations[i])) for i in range(n_agents)]

    # Create the problem with the appropriate objective
    objective = Leximin(utilities) if objective_type.lower() == "leximin" else Leximax(utilities)
    problem = cvxpy.Problem(objective, constraints)

    # For resource allocation, we don't have predefined outcome levels
    outcome_levels = []

    return problem, outcome_levels, utilities


def create_clients_facility_locations(size: int, p: int, objective_type: str = "leximax") -> Tuple[cvxpy.Problem, List[float], List[cvxpy.Expression]]:
    """Create a facility location problem with size clients and p facilities."""
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
    objective = Leximin(outcomes) if objective_type.lower() == "leximin" else Leximax(outcomes)
    problem = Problem(objective, constraints)

    # Estimate possible outcome values (sorted descending for leximax)
    possible_values = sorted(set(d.flatten()), reverse=True)

    return problem, possible_values, outcomes


def single_experiment(problem_type: str, objective_type: str, method: str, repeats: int = 1, **problem_params) -> Dict[str, Any]:
    """
    Run a single experiment instance and return the results.

    This function will be called by experiments_csv for each parameter combination.
    """
    logger.info(f"Running experiment: {problem_type}, {objective_type}, {method}, params: {problem_params}")

    # Select the problem creation method
    if problem_type == "resource_allocation":
        problem_creation_method = create_resource_allocation_problem
    elif problem_type == "facility_location":
        problem_creation_method = create_clients_facility_locations
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    # Create the problem
    try:
        problem, outcome_levels, utilities = problem_creation_method(
            objective_type=objective_type, **problem_params
        )
    except Exception as e:
        logger.error(f"Error creating problem: {e}")
        return {
            "time": float('inf'),
            "success": False,
            "error": str(e),
            "solution_value": None
        }

    # Run the method multiple times for timing
    times = []
    success = False
    solution_value = None
    error_message = None

    for i in range(repeats):
        start_time = time.time()
        try:
            if method == "ordered_values":
                if not outcome_levels:
                    logger.warning(f"Skipping {method} - no outcome levels available")
                    times.append(float('inf'))
                    continue
                solution_value = problem.solve(method=method, outcome_levels=outcome_levels)
            else:
                solution_value = problem.solve(method=method)
            success = True
            elapsed = time.time() - start_time
            times.append(elapsed)
            logger.debug(f"Run {i + 1}/{repeats}: {elapsed:.3f} seconds")
        except Exception as e:
            logger.error(f"Error with {method}: {e}")
            error_message = str(e)
            elapsed = float('inf')
            times.append(elapsed)

    # Calculate average time
    avg_time = np.mean(times) if times else float('inf')

    return {
        "time": avg_time,
        "success": success,
        "solution_value": solution_value,
        "error": error_message,
        "min_time": min(times) if times else float('inf'),
        "max_time": max(times) if times else float('inf')
    }


def plot_results_from_csv(results_file: str, problem_type: str):
    """Plot the benchmark results from a CSV file."""
    # Read results
    results = pd.read_csv(results_file)

    # Filter successful results only
    results = results[results["success"] == True]

    if results.empty:
        logger.warning("No successful results to plot")
        return

    objective_types = sorted(results["objective_type"].unique())

    # Determine color column based on problem type
    if problem_type == "facility_location":
        x_column = 'size'
        color_column = 'p'
    else:  # resource_allocation
        x_column = 'n_resources'
        color_column = 'n_agents'

    # Get unique values for the color column
    if color_column in results.columns:
        color_values = sorted(results[color_column].unique())
    else:
        color_values = [None]

    # Define method name mapping
    method_name_mapping = {
        "ogry_integer": "Ordered Outcomes 1",
        "ogry_relax": "Ordered Outcomes 2",
        "ordered_values": "Ordered Values",
        "willson": "Willson"
    }

    fig, axes = plt.subplots(len(objective_types), len(color_values),
                             figsize=(4 * len(color_values), 4 * len(objective_types)),
                             squeeze=False)

    for i, obj_type in enumerate(objective_types):
        for j, color_val in enumerate(color_values):
            ax = axes[i, j]

            # Filter results for this objective type and color value
            if color_val is not None:
                filtered_results = results[
                    (results["objective_type"] == obj_type) &
                    (results[color_column] == color_val)
                ]
                title_suffix = f"\n{color_column}={color_val}"
            else:
                filtered_results = results[results["objective_type"] == obj_type]
                title_suffix = ""

            # Plot each method
            for method in sorted(filtered_results["method"].unique()):
                method_results = filtered_results[filtered_results["method"] == method]
                method_results = method_results.sort_values(x_column)

                if not method_results.empty:
                    # Use mapped name if available, otherwise use original name
                    display_name = method_name_mapping.get(method, method)
                    ax.plot(method_results[x_column], method_results["time"],
                            'o-', label=display_name, linewidth=2, markersize=6)

            ax.set_title(f"{obj_type.capitalize()}{title_suffix}")
            ax.set_xlabel(x_column.replace('_', ' ').title())
            ax.set_ylabel("Time (seconds)")
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.001, top=1e3)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.legend()

    plt.suptitle("Performance Comparison of Leximin/Leximax Methods", fontsize=16)
    plt.tight_layout()
    filename = f"{problem_type}_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {filename}")
    plt.close()


def main():
    """Main function to run the benchmark using experiments_csv."""
    parser = argparse.ArgumentParser(description="Benchmark leximin/leximax methods using experiments_csv")
    parser.add_argument("--problem_type", choices=["resource_allocation", "facility_location"],
                        default="facility_location",
                        help="Type of problem to create")
    parser.add_argument("--methods", nargs="+",
                        default=["willson", "ogry_relax", "ogry_integer", "ordered_values"],
                        help="Methods to benchmark")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repetitions for each method")
    parser.add_argument("--results_dir", default="results/",
                        help="Directory for results")
    parser.add_argument("--results_file", default="benchmark_results.csv",
                        help="CSV file name for results")
    parser.add_argument("--clear_previous", action="store_true",
                        help="Clear previous results and start fresh")
    parser.add_argument("--plot_only", action="store_true",
                        help="Only plot results from existing CSV file")

    args = parser.parse_args()

    # Set up experiments_csv
    ex = experiments_csv.Experiment(
        results_folder=args.results_dir,
        results_filename=args.results_file,
        backup_folder=f"{args.results_dir}backups"
    )

    # Set logging level
    ex.logger.setLevel(logging.INFO)

    if args.clear_previous:
        ex.clear_previous_results()
        logger.info("Cleared previous results")

    # If plot_only is specified, just plot and exit
    if args.plot_only:
        results_path = f"{args.results_dir}{args.results_file}"
        plot_results_from_csv(results_path, args.problem_type)
        return

    # Define parameter ranges based on problem type
    if args.problem_type == "facility_location":
        param_ranges = {
            "problem_type": [args.problem_type],
            "objective_type": ["leximin"],
            "method": args.methods,
            "repeats": [args.repeats],
            "size": [5, 8, 12],
            "p": [2, 3]
        }
    else:  # resource_allocation
        param_ranges = {
            "problem_type": [args.problem_type],
            "objective_type": ["leximax"],
            "method": args.methods,
            "repeats": [args.repeats],
            "n_resources": [5, 10, 15],
            "n_agents": [3, 4, 5]
        }

    logger.info(f"Starting benchmark for {args.problem_type} with {len(args.methods)} methods")
    logger.info(f"Parameter ranges: {param_ranges}")

    # Run the experiments
    ex.run(single_experiment, param_ranges)

    logger.info("Benchmark completed!")

    # Plot the results
    results_path = f"{args.results_dir}{args.results_file}"
    plot_results_from_csv(results_path, args.problem_type)

    # Print summary
    results = pd.read_csv(results_path)
    logger.info(f"\nTotal experiments: {len(results)}")
    logger.info(f"Successful experiments: {len(results[results['success'] == True])}")
    logger.info(f"Failed experiments: {len(results[results['success'] == False])}")

    # Print performance summary
    successful_results = results[results['success'] == True]
    if not successful_results.empty:
        logger.info("\nPerformance Summary:")
        summary = successful_results.groupby(['objective_type', 'method'])['time'].agg(['mean', 'std', 'min', 'max'])
        logger.info(f"\n{summary}")


if __name__ == "__main__":
    main()