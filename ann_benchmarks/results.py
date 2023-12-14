import json
import os
import re
import traceback
from typing import Any, Optional, Set, Tuple, Iterator
import h5py
import shutil
from pathlib import PurePath

from ann_benchmarks.definitions import Definition


def build_result_filepath(dataset_name: Optional[str] = None, 
                          count: Optional[int] = None, 
                          definition: Optional[Definition] = None, 
                          query_arguments: Optional[Any] = None, 
                          batch_mode: bool = False) -> str:
    """
    Constructs the filepath for storing the results.

    Args:
        dataset_name (str, optional): The name of the dataset.
        count (int, optional): The count of records.
        definition (Definition, optional): The definition of the algorithm.
        query_arguments (Any, optional): Additional arguments for the query.
        batch_mode (bool, optional): If True, the batch mode is activated.

    Returns:
        str: The constructed filepath.
    """
    d = ["results"]
    if dataset_name:
        d.append(dataset_name)
    if count:
        d.append(str(count))
    if definition:
        d.append(definition.algorithm + ("-batch" if batch_mode else ""))
        data = definition.arguments + query_arguments
        d.append(re.sub(r"\W+", "_", json.dumps(data, sort_keys=True)).strip("_") + ".hdf5")
    return os.path.join(*d)


def store_results(dataset_name: str, count: int, definition: Definition, query_arguments:Any, attrs, results, batch):
    """
    Stores results for an algorithm (and hyperparameters) running against a dataset in a HDF5 file.

    Args:
        dataset_name (str): The name of the dataset.
        count (int): The count of records.
        definition (Definition): The definition of the algorithm.
        query_arguments (Any): Additional arguments for the query.
        attrs (dict): Attributes to be stored in the file.
        results (list): Results to be stored.
        batch (bool): If True, the batch mode is activated.
    """
    filename = build_result_filepath(dataset_name, count, definition, query_arguments, batch)
    directory, _ = os.path.split(filename)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with h5py.File(filename, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        times = f.create_dataset("times", (len(results),), "f")
        neighbors = f.create_dataset("neighbors", (len(results), count), "i")
        distances = f.create_dataset("distances", (len(results), count), "f")
        
        for i, (time, ds) in enumerate(results):
            times[i] = time
            neighbors[i] = [n for n, d in ds] + [-1] * (count - len(ds))
            distances[i] = [d for n, d in ds] + [float("inf")] * (count - len(ds))


def load_all_results(dataset: Optional[str] = None, 
                 count: Optional[int] = None, 
                 batch_mode: bool = False) -> Iterator[Tuple[dict, h5py.File]]:
    """
    Loads all the results from the HDF5 files in the specified path.

    Args:
        dataset (str, optional): The name of the dataset.
        count (int, optional): The count of records.
        batch_mode (bool, optional): If True, the batch mode is activated.

    Yields:
        tuple: A tuple containing properties as a dictionary and an h5py file object.
    """
    for root, _, files in os.walk(build_result_filepath(dataset, count)):
        for filename in files:
            if os.path.splitext(filename)[-1] != ".hdf5":
                continue
            try:
                with h5py.File(os.path.join(root, filename), "r+") as f:
                    properties = dict(f.attrs)
                    if batch_mode != properties["batch_mode"]:
                        continue
                    yield properties, f
            except Exception:
                print(f"Was unable to read {filename}")
                traceback.print_exc()


def get_unique_algorithms() -> Set[str]:
    """
    Retrieves unique algorithm names from the results.

    Returns:
        set: A set of unique algorithm names.
    """
    algorithms = set()
    for batch_mode in [False, True]:
        for properties, _ in load_all_results(batch_mode=batch_mode):
            algorithms.add(properties["algo"])
    return algorithms


def load_a_result(result_file_path: str) -> Iterator[Tuple[dict, h5py.File]]:
    """
    Loads the result from the HDF5 file in the specified path.

    Args:
        result_file_path (str): The result file path.

    Yields:
        tuple: A tuple containing properties as a dictionary and an h5py file object.
    """
    assert os.path.splitext(result_file_path)[-1] == ".hdf5", "Provided file does not have a .hdf5 extension!"
    try:
        with h5py.File(result_file_path, "r+") as f:
            properties = dict(f.attrs)
            yield properties, f
    except Exception:
        print(f"Was unable to read {result_file_path}")
        traceback.print_exc()
    

def move_result_to_bay_opt_dir(file_path: str):
    """
    Move result file to `bay_opt` directory. Will overwrite, if needed.

    Args:
        file_path (str): File path to be moved (relative to `results` directory).
    """
    pure_path_obj = PurePath(file_path)
    path_components = list(pure_path_obj.parts)
    path_components.insert(1, 'bay_opt') # Destination file path to be 'results/bay_opt/...'
    destination_file_path = PurePath('').joinpath(*path_components)
    print(f"Moving result to: {destination_file_path}")
    directory, _ = os.path.split(destination_file_path)
    try:
    # Create directories if not present
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except Exception:
        print(f"Could not create directory {directory}")
        traceback.print_exc()
        return
    try:
        shutil.move(file_path, destination_file_path)
    except Exception:
        print(f"Could not move file to {destination_file_path}")
        traceback.print_exc()
        return
    # Remove now empty source file directories
    directory, _ = os.path.split(file_path)
    try:
        os.removedirs(directory)
    except Exception:
        print(f"Could not remove directory {directory}")
        traceback.print_exc()
        return
