from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset

mm = SingleCellMemMapDataset("DATA/medium/processed_dataqq2/train/mem.map")


def row_bounds_ok(i):
    start = mm.row_index[i]
    end = mm.row_index[i + 1]
    cols = mm.col_index[start:end]
    ncols = mm._feature_index.number_vars_at_row(i)
    bad = cols[(cols < 0) | (cols >= ncols)]
    return bad, cols, ncols


for i in range(mm.number_of_rows()):
    bad, cols, ncols = row_bounds_ok(i)
    if bad.size:
        print(
            f"ROW {i}: ncols={ncols}, max(col)={cols.max()}, min(col)={cols.min()}, first_bad={bad[0]}"
        )
        break
else:
    print("All rows OK")
