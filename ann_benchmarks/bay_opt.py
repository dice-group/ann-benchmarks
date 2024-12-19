import argparse
from ann_benchmarks.plotting.utils import compute_metrics_all_runs
from ann_benchmarks.results import build_result_filepath, move_result_to_bay_opt_dir, load_a_result
from ann_benchmarks.definitions import Definition
from numbers import Number
from ann_benchmarks.datasets import DATASETS, get_dataset
import pandas as pd
import math
from random import randint

from bayes_opt import BayesianOptimization
from copy import copy

str_dict = {}

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

def new_value_in_original_type(old_value, k, v):
    if isinstance(old_value, int):
        new_value = round(v)
    elif isinstance(old_value, float):
        new_value = float(v)
    elif isinstance(old_value, str):
        new_value = str(str_dict[k][int(round(v))]) # Refer `str_dict` to obtain the string representing the returned no.
    elif isinstance(old_value, dict):
        key = k[k.rindex('key_')+4:k.rindex('_')]
        new_dict = old_value
        if '_list_' in key:   # To handle the rare case where a dictionary holds a list 
            key = key[:key.rindex('_list_')]   # Obtain the original key. ALERT: May misbehave if the original dict key has '_list_' in it. 
        new_dict_value = new_value_in_original_type(old_value[key], k, v) # ALERT: Recursion here!
        new_dict[key] = new_dict_value
        new_value = new_dict
    elif isinstance(old_value, list):
        loc = int(k[k.rindex('list_')+5:k.rindex('_')])
        new_list = old_value
        new_list_value = new_value_in_original_type(old_value[loc], k, v) # ALERT: Recursion here!
        new_list[loc] = new_list_value
        new_value = new_list
    else:
        new_value = v
    return new_value

def set_params(definition: Definition, new_params: dict):
    # Replace the values obtained from the optimizer in their original positions 
    print(f"In set_params")
    print(f"Original arguments: {definition.arguments}")
    print(f"Original query arguments: {definition.query_argument_groups}")
    arguments_list = None
    query_arguments_list = None
    if len(definition.arguments) > 0:
        arguments_list = copy(definition.arguments)
    if len(definition.query_argument_groups) > 0:
        query_arguments_list = copy(definition.query_argument_groups[0])
        
    for _, (k,v) in enumerate(new_params.items()):
        print(f"k: {k}")
        # Get the original position of the parameter (encoded in key)
        pos = int(k[k.rindex('_')+1:])
        print(f"pos: {pos}")
        if k[0:k.index('_')] == 'args':     # If key begins with 'args', value belongs in arguments list
            print(f"In ARGS")
            old_value = definition.arguments[pos]
            print(f"old_value: {old_value}")
            new_value = new_value_in_original_type(old_value, k, v)     # Set new parameter based on original parameter type
            arguments_list[pos] = new_value
            print(f"New Value(Type) at Location : {new_value}({type(new_value)}) at {pos}")
        elif k[0:k.index('_')] == 'query':      # If key begins with 'query', value belongs in query arguments list
            print(f"In QUERY args")
            old_value = definition.query_argument_groups[0][pos]
            print(f"old_value: {old_value}")
            new_value = new_value_in_original_type(old_value, k, v)     # Set new parameter based on original parameter type
            query_arguments_list[pos] = new_value
            print(f"New Value(Type) at Location : {new_value}({type(new_value)}) at {pos}")
    if arguments_list is not None:
        definition.arguments = arguments_list
    if query_arguments_list is not None:
        definition.query_argument_groups = [query_arguments_list]
    print(f"New arguments: {definition.arguments}")
    print(f"New query arguments: {definition.query_argument_groups}")

def obtain_min_max_from_series(pds:pd.Series, golden_key:str) -> tuple:
    if isinstance(pds[0], Number):
        # Calculate min, max
        min_value = pds.min()
        max_value = pds.max()
    elif isinstance(pds[0], str):
        uniq_str_array = pd.unique(pds)
        str_dict[golden_key] = uniq_str_array  # Needs to be stored globally for retrieval
        min_value = 0
        max_value = len(uniq_str_array)-1
    return (min_value, max_value)

def obtain_param_positions_bounds_dict(dataframes_dict: dict[str, pd.DataFrame]) -> dict[str, tuple]:
    print(f"In obtain_param_positions_bounds_dict")
    param_positions_bounds_dict = {}
    # Args dataframe
    for _, (key_string,df) in enumerate(dataframes_dict.items()):
        print(f"key_string: {key_string}")
        print(f"Dataframe Shape: {df.shape}")
        for i in range(df.shape[1]):
            print(f"Column {i}")
            print(f'type(df[{i}][0]): {type(df[i][0])}')
            if isinstance(df[i][0], Number):
                # Skip, if NaN
                if math.isnan(df[i][0]):
                    print(f"NaN Value!")
                    continue
                golden_key = str(key_string+'_pos_'+str(i))
                print(f"golden_key: {golden_key}")
                param_positions_bounds_dict[golden_key] = obtain_min_max_from_series(df[i], golden_key)
                print(f"min, max: {param_positions_bounds_dict[golden_key]}")
            elif isinstance(df[i][0], str):
                golden_key = str(key_string+'_pos_'+str(i))
                print(f"golden_key: {golden_key}")
                param_positions_bounds_dict[golden_key] = obtain_min_max_from_series(df[i], golden_key)
                print(f"min, max: {param_positions_bounds_dict[golden_key]}")
            elif isinstance(df[i][0], dict):
                print("In Dict")
                dict_df = pd.json_normalize(df[i].to_list())
                for key in dict_df.keys():
                    print(f"Key: {key}")
                    print(f'type(dict_df[{key}][0]): {type(dict_df[key][0])}')
                    if not isinstance(dict_df[key][0], list):   
                        golden_key = str(key_string+'_pos_key_'+key+'_'+str(i))
                        print(f"golden_key: {golden_key}")
                        param_positions_bounds_dict[golden_key] = obtain_min_max_from_series(dict_df[key], golden_key)
                        print(f"min, max: {param_positions_bounds_dict[golden_key]}")
                    else:   # To handle the rare case where a dictionary holds a list
                        list_df = pd.DataFrame(dict_df[key].to_list())
                        for col in range(list_df.shape[1]):
                            print(f"Column {col}")
                            print(f'type(list_df[{col}][0]): {type(list_df[col][0])}')
                            if isinstance(list_df[col][0], Number) or isinstance(list_df[col][0], str):
                                golden_key = str(key_string+'_pos_key_'+key+'_list_'+str(col)+'_'+str(i))  
                                print(f"golden_key: {golden_key}")
                                param_positions_bounds_dict[golden_key] = obtain_min_max_from_series(list_df[col], golden_key)
                                print(f"min, max: {param_positions_bounds_dict[golden_key]}")
            elif isinstance(df[i][0], list):
                print("In List")
                list_df = pd.DataFrame(df[i].to_list())
                for col in range(list_df.shape[1]):
                    print(f"Column {col}")
                    print(f'type(list_df[{col}][0]): {type(list_df[col][0])}')
                    if isinstance(list_df[col][0], Number) or isinstance(list_df[col][0], str):
                        golden_key = str(key_string+'_pos_list_'+str(col)+'_'+str(i))
                        print(f"golden_key: {golden_key}")
                        param_positions_bounds_dict[golden_key] = obtain_min_max_from_series(list_df[col], golden_key)
                        print(f"min, max: {param_positions_bounds_dict[golden_key]}")
    return param_positions_bounds_dict

def obtain_dataframes_from_definitions(definitions: list[Definition]) -> (pd.DataFrame, pd.DataFrame):
    # Create lists of arguments and query arguments
    args_list = []
    query_args_list = []
    for definition in definitions:
        args_list.append(definition.arguments)
        for query_args in definition.query_argument_groups:
            query_args_list.append(query_args)
    # Create pandas dataframe from list of lists (containing arguments and query arguments)
    args_df = pd.DataFrame(args_list)
    query_args_df = pd.DataFrame(query_args_list)
    return args_df, query_args_df

def run_using_bayesian_optimizer(definition: Definition, args: argparse.Namespace, param_positions_bounds_dict: dict[str, tuple], random_seed):
    # param_positions_bounds_dict = {'args_pos_1': (10, 1000), 'query_args_pos_0': (10, 60), 'query_args_pos_1': (70, 120)}
    # TODO: Remove print statements after testing.
    # pbounds = create_parameter_bounds_from(param_positions_bounds_dict)
    # print(f"Parameter Bounds are: {pbounds}")
    print(f"Parameter Bounds are: {param_positions_bounds_dict}")
    init_points=2
    n_iter=5
    def black_box_function(**kwargs):
        """Function with unknown internals we wish to maximize."""
        new_params = kwargs
        # Check the no. of parameters
        # assert len(definition.arguments) >= len(new_params), "NO. OF OPTIMIZED PARAMETERS IS MORE THAN REQUIRED."
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
        # pbounds=pbounds, # Bounded region of parameter space
        pbounds=param_positions_bounds_dict, # Bounded region of parameter space
        random_state=random_seed,
        allow_duplicate_points=True
    )
    optimizer.maximize( # Choose the parameters which maximize the function value
        init_points=init_points,
        n_iter=n_iter,
    )
    print(f"Optimizer max: {optimizer.max}")
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

def execute_using_bayesian_optimizer(bay_opt_definitions: list[Definition], args: argparse.Namespace):
    # Arrange parameters in dataframes 
    args_df, query_args_df = obtain_dataframes_from_definitions(bay_opt_definitions)
    print(f'args_df: {args_df}')
    print(f'query_args_df: {query_args_df}')
    # Arrange dataframes in dictionaries as per required format
    param_positions_bounds_dict = obtain_param_positions_bounds_dict({'args': args_df, 'query_args': query_args_df})
    print(f'param_positions_bounds_dict: {param_positions_bounds_dict}')
    print(f'str_dict: {str_dict}')
    no_of_runs = 5  # default no. of args.runs
    print(f'no_of_runs: {no_of_runs}')
    for iteration in range(no_of_runs):
        print(f'Iteration : {iteration}')
        random_seed_value = randint(0, 2**32 - 1)   # Seed must be between 0 and 2**32 - 1 to avoid exception
        print(f'Random Seed Value : {random_seed_value}')
        run_using_bayesian_optimizer(bay_opt_definitions[0], args, param_positions_bounds_dict, random_seed_value)
    


