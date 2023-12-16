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
        # Set new parameter based on original parameter type
        if isinstance(definition.arguments[int(k)], int):
            definition.arguments[int(k)] = int(v)
        elif isinstance(definition.arguments[int(k)], float):
            definition.arguments[int(k)] = float(v)
        else:
            definition.arguments[int(k)] = v
        print(f"New Value at Location : {definition.arguments[int(k)]} at {int(k)}")

def obtain_recall_from(filepath: str, dataset_name: str) -> float:
    print(f"Looking for file: {filepath}")
    if len(list(load_a_result(filepath))) > 0:
            res = load_a_result(filepath)
            dataset, _ = get_dataset(dataset_name)
            run_results = compute_metrics_all_runs(dataset, res)
            for result in run_results:
                return result["k-nn"]  # 'k-nn': The key for Recall value
    else:
        return 0    # To handle error cases e.g. FileNotFound Error

def obtain_recalls_and_results(definition: Definition, args: argparse.Namespace) -> (list, list):
    recall_values = []
    result_file_paths = []
    if definition.query_argument_groups:
        for query_arguments in definition.query_argument_groups:
            filepath = build_result_filepath(args.dataset, args.count, definition, query_arguments, args.batch)
            recall_values.append(obtain_recall_from(filepath, args.dataset))
            result_file_paths.append(filepath)
    else:
        filepath = build_result_filepath(args.dataset, args.count, definition, definition.query_argument_groups, args.batch)
        recall_values.append(obtain_recall_from(filepath, args.dataset))
        result_file_paths.append(filepath)

    return (recall_values, result_file_paths)   # Return recall values and file paths
    

def run_using_bayesian_optimizer(definition: Definition, args: argparse.Namespace, param_positions_bounds_dict: dict[int, tuple]):
    # pbounds = {'1': (10, 1000), '2': (1, 105),...} 
    # TODO: Remove print statements after testing.
    pbounds = create_parameter_bounds_from(param_positions_bounds_dict)
    print(f"Parameter Bounds are: {pbounds}")
    random_state=1
    init_points=2
    n_iter=5
    def black_box_function(**kwargs):
        """Function with unknown internals we wish to maximize."""
        new_params = kwargs
        # Check the no. of parameters
        assert len(definition.arguments) >= len(new_params), "NO. OF OPTIMIZED PARAMETERS IS MORE THAN REQUIRED."
        # Set newly obtained parameters in definition
        set_params(definition, new_params)
        # RUN ANN-Benchmarks for this updated definition
        from ann_benchmarks.main import create_workers_and_execute  # Import here to avoid cyclical imports error
        create_workers_and_execute([definition], args)
        # Compute the maximum Recall from the result files of this newly run experiment        
        recall_values, result_path_list = obtain_recalls_and_results(definition, args)
        print(f"Recall Values: {recall_values}")
        print(f"Max Recall Value: {max(recall_values)}")
        # Move the result files to another directory, if asked
        if args.move_bayesian_optimizer_result_files:
            for filepath in result_path_list:
                move_result_to_bay_opt_dir(filepath)

        return max(recall_values) # Return max recall value to Bayesian Optimizer

    optimizer = BayesianOptimization(
        f=black_box_function, # Function to be evaluated
        pbounds=pbounds, # Bounded region of parameter space
        random_state=random_state,
        allow_duplicate_points=True
    )
    
    optimizer.maximize( # Choose the parameters which maximize the function value
        init_points=init_points,
        n_iter=n_iter,
    )

    print(f"Optimizer max: {optimizer.max}")
    
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    


