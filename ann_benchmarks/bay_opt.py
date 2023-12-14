from bayes_opt import BayesianOptimization
import argparse
import numpy as np
from ann_benchmarks.plotting.utils import compute_metrics_all_runs
from ann_benchmarks.results import build_result_filepath, move_result_to_bay_opt_dir, load_a_result
from ann_benchmarks.definitions import Definition
from numbers import Number
from ann_benchmarks.datasets import DATASETS, get_dataset


def create_parameter_bounds_from(param_positions_bounds_dict: dict[int, tuple]) -> dict[str, tuple]:
    pbounds = {}
    for _, (k,v) in enumerate(param_positions_bounds_dict.items()):
        pbounds[str(k)] = v
    return pbounds

def set_params(definition: Definition, new_params: dict):
    for _, (k,v) in enumerate(new_params.items()):
        definition.arguments[int(k)] = v

def obtain_recall_from(filepath: str, dataset_name: str) -> float:
    if len(list(load_a_result(filepath))) > 0:
            res = load_a_result(filepath)
            dataset, _ = get_dataset(dataset_name)
            run_results = compute_metrics_all_runs(dataset, res)
            for result in run_results:
                return result["k-nn"]  # 'k-nn': The key for Recall value

def run_using_bayesian_optimizer(definition: Definition, args: argparse.Namespace, param_positions_bounds_dict: dict[int, tuple]):
    # pbounds = {'1': (10, 1000), '2': (1, 105),...} 
    # TODO: Remove print statements after testing.
    pbounds = create_parameter_bounds_from(param_positions_bounds_dict)
    print(f"Parameter Bounds are: {pbounds}")
    random_state=1
    init_points=1
    n_iter=1
    def black_box_function(**kwargs):
        """Function with unknown internals we wish to maximize."""
        new_params = kwargs
        print(f"New Parameters: {new_params}")
        # Check the no. of parameters
        assert len(definition.arguments) >= len(new_params), "NO. OF OPTIMIZED PARAMETERS IS MORE THAN REQUIRED."

        # Set parameters in definition as the newly obtained parameters
        set_params(definition, new_params)

        # RUN ANN-Benchmarks for this definition
        from ann_benchmarks.main import create_workers_and_execute  # Import here to avoid cyclical imports error
        create_workers_and_execute([definition], args)

        # Compute the Recall from the result (written in a file) of this newly run experiment
        filepath = build_result_filepath(args.dataset, args.count, definition, definition.query_argument_groups, args.batch)
        print(f"Looking for file: {filepath}")
        recall = obtain_recall_from(filepath, args.dataset)
        print(f"Recall: {recall}")

        # Move the result file to another directory
        if args.move_bayesian_optimizer_result_files:
            move_result_to_bay_opt_dir(filepath)

        return recall # Return recall value to be maximized by Bayesian Optimizer

    optimizer = BayesianOptimization(
        f=black_box_function, # Function to be evaluated
        pbounds=pbounds, # Bounded region of parameter space
        random_state=random_state,
    )
    
    optimizer.maximize( # Choose the parameters which maximize the function value
        init_points=init_points,
        n_iter=n_iter,
    )

    print(f"Optimizer max: {optimizer.max}")
    # Set parameters providing maximum Recall in definition
    # set_params(definition, optimizer.max['params'])
    # print(f"New definition: {definition}")
    
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    


