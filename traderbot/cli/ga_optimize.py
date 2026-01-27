"""CLI for GA hyperparameter optimization."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from traderbot.config import get_config
from traderbot.opt.ga_opt import GAConfig, GeneticOptimizer, SearchSpace


def main() -> int:
    """Run GA optimization from command line."""
    parser = argparse.ArgumentParser(
        description="Run genetic algorithm hyperparameter optimization"
    )
    
    # Data parameters
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--universe", nargs="+", required=True, help="Universe tickers")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of walk-forward splits")
    
    # GA parameters
    parser.add_argument("--population-size", type=int, default=20, help="Population size")
    parser.add_argument("--n-generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--budget", type=int, help="Total budget (population * generations)")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.7, help="Crossover rate")
    parser.add_argument("--elite-frac", type=float, default=0.2, help="Elite fraction")
    parser.add_argument("--parallel", action="store_true", default=True, help="Parallel evaluation")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs")
    
    # Objective
    parser.add_argument(
        "--objective",
        default="total_reward",
        choices=["total_reward", "sharpe", "sortino"],
        help="Optimization objective",
    )
    
    # Output
    parser.add_argument("--output-dir", default="runs/ga_opt", help="Output directory")
    
    args = parser.parse_args()
    
    # Handle budget shorthand
    if args.budget:
        # Distribute budget across population and generations
        # budget = pop_size * n_gens
        args.population_size = min(20, args.budget // 5)
        args.n_generations = args.budget // args.population_size
    
    # Build base command
    universe_str = " ".join(args.universe)
    base_cmd = (
        f"python -m traderbot.cli.walkforward "
        f"--start-date {args.start_date} "
        f"--end-date {args.end_date} "
        f"--universe {universe_str} "
        f"--n-splits {args.n_splits} "
        f"--output-dir runs/ga_temp"
    )
    
    # Setup GA
    ga_config = GAConfig(
        population_size=args.population_size,
        n_generations=args.n_generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elite_frac=args.elite_frac,
        parallel=args.parallel,
        n_jobs=args.n_jobs,
    )
    
    search_space = SearchSpace()
    
    optimizer = GeneticOptimizer(
        search_space=search_space,
        ga_config=ga_config,
        base_cmd=base_cmd,
        objective=args.objective,
    )
    
    # Run optimization
    print(f"Starting GA optimization with {args.population_size} individuals, "
          f"{args.n_generations} generations")
    print(f"Objective: {args.objective}")
    print(f"Base command: {base_cmd}")
    print()
    
    best_config = optimizer.optimize()
    
    # Save results
    output_dir = Path(args.output_dir)
    optimizer.save_results(output_dir)
    
    print()
    print(f"Optimization complete!")
    print(f"Best {args.objective}: {optimizer.best_fitness:.4f}")
    print(f"Best config saved to: {output_dir / 'best_config.json'}")
    print(f"Replay with: {output_dir / 'replay.sh'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

