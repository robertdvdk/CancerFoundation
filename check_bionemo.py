import importlib.metadata
import inspect

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset

print("bionemo-scdl version:", importlib.metadata.version("bionemo-scdl"))
print(
    "get_row_padded signature:",
    inspect.signature(SingleCellMemMapDataset.get_row_padded),
)
