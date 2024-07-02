# paad/src/prepare_dataset/__init__.py
from .prepare_dataset import get_data_paths_and_labels_from_machine

VERSION = "0.1.0"

__all__ = [ "version", "get_data_paths_and_labels_from_machine", "get_data_paths_and_labels_from_edge_dir"]

print("Initializeing prepare_dataset...")