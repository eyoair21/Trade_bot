"""Genetic algorithm optimization for hyperparameters and reward weights.

Searches for optimal configuration by maximizing OOS reward or Sharpe ratio.
"""

import json
import multiprocessing as mp
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GAConfig:
    """GA optimization configuration.
    
    Args:
        population_size: Number of individuals per generation
        n_generations: Number of generations to evolve
        mutation_rate: Probability of mutation per gene
        crossover_rate: Probability of crossover
        elite_frac: Fraction of top individuals to preserve
        parallel: Whether to evaluate in parallel
        n_jobs: Number of parallel jobs (-1 = all CPUs)
    """
    population_size: int = 20
    n_generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_frac: float = 0.2
    parallel: bool = True
    n_jobs: int = -1


@dataclass
class SearchSpace:
    """Hyperparameter search space definition.
    
    Each parameter is (min, max, type) where type is 'float', 'int', or 'choice'.
    """
    # Reward weights
    lambda_dd: tuple[float, float, str] = (0.0, 1.0, "float")
    tau_turnover: tuple[float, float, str] = (1e-5, 1e-2, "float")
    kappa_breach: tuple[float, float, str] = (0.0, 2.0, "float")
    
    # Model hyperparams
    lookback: tuple[int, int, str] = (16, 64, "int")
    patch_size: tuple[int, int, str] = (4, 16, "int")
    learning_rate: tuple[float, float, str] = (1e-5, 1e-2, "float")
    
    # Policy params
    target_vol: tuple[float, float, str] = (0.10, 0.25, "float")
    max_leverage: tuple[float, float, str] = (0.5, 1.5, "float")
    
    # Universe
    top_n: tuple[int, int, str] = (20, 200, "int")


class GeneticOptimizer:
    """Genetic algorithm optimizer for TraderBot hyperparameters."""
    
    def __init__(
        self,
        search_space: SearchSpace,
        ga_config: GAConfig,
        base_cmd: str,
        objective: str = "total_reward",
    ):
        """Initialize optimizer.
        
        Args:
            search_space: Hyperparameter search space
            ga_config: GA configuration
            base_cmd: Base command to run backtest (will append params)
            objective: Objective metric ('total_reward', 'sharpe', 'sortino')
        """
        self.search_space = search_space
        self.ga_config = ga_config
        self.base_cmd = base_cmd
        self.objective = objective
        
        self.population: list[dict[str, Any]] = []
        self.fitness_scores: list[float] = []
        self.best_individual: dict[str, Any] | None = None
        self.best_fitness: float = -np.inf
        self.history: list[dict[str, Any]] = []
    
    def optimize(self) -> dict[str, Any]:
        """Run GA optimization.
        
        Returns:
            Best configuration found
        """
        # Initialize population
        self._initialize_population()
        
        # Evolve for N generations
        for gen in range(self.ga_config.n_generations):
            print(f"Generation {gen + 1}/{self.ga_config.n_generations}")
            
            # Evaluate fitness
            self._evaluate_population()
            
            # Track best
            gen_best_idx = np.argmax(self.fitness_scores)
            gen_best_fitness = self.fitness_scores[gen_best_idx]
            
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_individual = self.population[gen_best_idx].copy()
                print(f"  New best {self.objective}: {self.best_fitness:.4f}")
            
            # Log generation stats
            self.history.append({
                "generation": gen,
                "best_fitness": gen_best_fitness,
                "avg_fitness": np.mean(self.fitness_scores),
                "std_fitness": np.std(self.fitness_scores),
            })
            
            # Create next generation
            self._evolve_population()
        
        return self.best_individual  # type: ignore
    
    def _initialize_population(self) -> None:
        """Create initial random population."""
        self.population = []
        for _ in range(self.ga_config.population_size):
            individual = self._random_individual()
            self.population.append(individual)
    
    def _random_individual(self) -> dict[str, Any]:
        """Generate random individual from search space."""
        individual = {}
        for param, (low, high, ptype) in asdict(self.search_space).items():
            if ptype == "float":
                # Log scale for small ranges
                if high / low > 100:
                    individual[param] = np.exp(
                        random.uniform(np.log(low), np.log(high))
                    )
                else:
                    individual[param] = random.uniform(low, high)
            elif ptype == "int":
                individual[param] = random.randint(low, high)
            else:
                raise ValueError(f"Unknown param type: {ptype}")
        return individual
    
    def _evaluate_population(self) -> None:
        """Evaluate fitness for all individuals."""
        if self.ga_config.parallel and self.ga_config.n_jobs != 1:
            # Parallel evaluation
            n_jobs = (
                mp.cpu_count()
                if self.ga_config.n_jobs == -1
                else self.ga_config.n_jobs
            )
            with mp.Pool(processes=n_jobs) as pool:
                self.fitness_scores = pool.map(
                    self._evaluate_individual,
                    self.population
                )
        else:
            # Sequential evaluation
            self.fitness_scores = [
                self._evaluate_individual(ind) for ind in self.population
            ]
    
    def _evaluate_individual(self, individual: dict[str, Any]) -> float:
        """Evaluate single individual by running backtest.
        
        Args:
            individual: Parameter configuration
        
        Returns:
            Fitness score (higher is better)
        """
        # Build command with params
        cmd_parts = [self.base_cmd]
        for param, value in individual.items():
            cmd_parts.append(f"--{param.replace('_', '-')} {value}")
        cmd = " ".join(cmd_parts)
        
        try:
            # Run backtest
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout
            )
            
            if result.returncode != 0:
                return -np.inf
            
            # Parse results.json
            results_path = Path("runs") / "latest" / "results.json"
            if not results_path.exists():
                return -np.inf
            
            with open(results_path) as f:
                results = json.load(f)
            
            # Extract objective
            fitness = results.get(self.objective, -np.inf)
            return float(fitness)
        
        except Exception as e:
            print(f"  Eval failed: {e}")
            return -np.inf
    
    def _evolve_population(self) -> None:
        """Create next generation via selection, crossover, mutation."""
        # Elite preservation
        n_elite = max(1, int(self.ga_config.elite_frac * self.ga_config.population_size))
        elite_indices = np.argsort(self.fitness_scores)[-n_elite:]
        next_pop = [self.population[i].copy() for i in elite_indices]
        
        # Generate offspring
        while len(next_pop) < self.ga_config.population_size:
            # Tournament selection
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            # Crossover
            if random.random() < self.ga_config.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            child = self._mutate(child)
            
            next_pop.append(child)
        
        self.population = next_pop
    
    def _tournament_select(self, k: int = 3) -> dict[str, Any]:
        """Tournament selection."""
        indices = random.sample(range(len(self.population)), k)
        best_idx = max(indices, key=lambda i: self.fitness_scores[i])
        return self.population[best_idx].copy()
    
    def _crossover(
        self,
        parent1: dict[str, Any],
        parent2: dict[str, Any],
    ) -> dict[str, Any]:
        """Uniform crossover."""
        child = {}
        for param in parent1.keys():
            child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
        return child
    
    def _mutate(self, individual: dict[str, Any]) -> dict[str, Any]:
        """Gaussian mutation for continuous, ±1 for discrete."""
        mutated = individual.copy()
        for param, (low, high, ptype) in asdict(self.search_space).items():
            if random.random() < self.ga_config.mutation_rate:
                if ptype == "float":
                    # Gaussian perturbation
                    sigma = (high - low) * 0.1
                    mutated[param] = np.clip(
                        mutated[param] + random.gauss(0, sigma),
                        low,
                        high,
                    )
                elif ptype == "int":
                    mutated[param] = np.clip(
                        mutated[param] + random.choice([-1, 1]),
                        low,
                        high,
                    )
        return mutated
    
    def save_results(self, output_dir: Path) -> None:
        """Save best config and history.
        
        Args:
            output_dir: Directory to save results (e.g., runs/<ts>/ga/)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Best config
        best_path = output_dir / "best_config.json"
        with open(best_path, "w") as f:
            json.dump(
                {
                    "config": self.best_individual,
                    "fitness": self.best_fitness,
                    "objective": self.objective,
                },
                f,
                indent=2,
            )
        
        # History
        history_path = output_dir / "ga_history.json"
        with open(history_path, "w") as f:
            json.dump({"history": self.history}, f, indent=2)
        
        # Generate replay script
        self._generate_replay_script(output_dir)
    
    def _generate_replay_script(self, output_dir: Path) -> None:
        """Generate shell/PowerShell script to replay best config."""
        if self.best_individual is None:
            return
        
        # Shell script
        sh_path = output_dir / "replay.sh"
        with open(sh_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Replay best GA config\n\n")
            cmd_parts = [self.base_cmd]
            for param, value in self.best_individual.items():
                cmd_parts.append(f"--{param.replace('_', '-')} {value}")
            f.write(" \\\n  ".join(cmd_parts) + "\n")
        sh_path.chmod(0o755)
        
        # PowerShell script
        ps_path = output_dir / "replay.ps1"
        with open(ps_path, "w") as f:
            f.write("# Replay best GA config\n\n")
            cmd_parts = [self.base_cmd]
            for param, value in self.best_individual.items():
                cmd_parts.append(f"--{param.replace('_', '-')} {value}")
            f.write(" `\n  ".join(cmd_parts) + "\n")

