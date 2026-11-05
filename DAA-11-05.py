import argparse
import sys
import time
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
from numba import njit, prange


@njit
def slope_distance_np(row1, row2):
    if np.sum(row1) == 0 and np.sum(row2) == 0:
        return 0.0

    n = len(row1)
    distance = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff_A = row1[i] - row1[j]
            diff_B = row2[i] - row2[j]
            distance += 2 * abs(diff_A - diff_B)
    return distance


def arc_values(x, angle):
    """Heights of the reference arc of the given angle (degrees) at the abscissas x in [0, 1]."""
    t = np.tan(np.deg2rad(angle) / 2)
    return 2*x*(1-t+t*x) / (1+t+np.sqrt((1-t)**2+4*t*(1-x)*(1+t*x)))


@njit(parallel=True)
def build_distance_matrix(data_matrix):

    N = data_matrix.shape[0]

    dist_matrix = np.empty((N, N), dtype=np.float64)
    for i in prange(N):
        for j in range(N):
            dist_matrix[i, j] = 0.0

    for i in prange(N):
        for j in range(i + 1, N):
            dist_ij = slope_distance_np(data_matrix[i], data_matrix[j])
            dist_matrix[i, j] = dist_ij
            dist_matrix[j, i] = dist_ij

    return dist_matrix


def main():
    def compute_union_degrees(dist_matrix1, dist_matrix2, names1, names2, pos_cor1, pos_cor2):
        def adjacency_from_distances(distances, thresholds):
            """Apply a scalar limit, per-OTU mutual limits, or a legacy pair-limit matrix."""
            thresholds = np.asarray(thresholds, dtype=np.float64)
            if thresholds.ndim == 1:
                adjacency = (distances < thresholds[:, None]) & (distances < thresholds[None, :])
            else:
                adjacency = distances < thresholds
            np.fill_diagonal(adjacency, False)
            return adjacency

        object_names = list(dict.fromkeys(list(names1) + list(names2)))
        positions = {name: index for index, name in enumerate(object_names)}
        indices1 = [positions[name] for name in names1]
        indices2 = [positions[name] for name in names2]

        adjacency1 = np.zeros((len(object_names), len(object_names)), dtype=bool)
        adjacency2 = np.zeros_like(adjacency1)
        adjacency1[np.ix_(indices1, indices1)] = adjacency_from_distances(dist_matrix1, pos_cor1)
        adjacency2[np.ix_(indices2, indices2)] = adjacency_from_distances(dist_matrix2, pos_cor2)
        np.fill_diagonal(adjacency1, False)
        np.fill_diagonal(adjacency2, False)

        changed = adjacency1 ^ adjacency2
        degrees_arr = changed.sum(axis=1)
        return np.asarray(object_names, dtype=object), degrees_arr

    def prepare_values(values, column_normalize=False):
        """Optional column normalization, followed by row min-max."""
        def min_max_scale(matrix):

            row_min = matrix.min(axis=1, keepdims=True)
            row_max = matrix.max(axis=1, keepdims=True)
            diff = row_max - row_min

            diff[diff == 0] = 1

            scaled = (matrix - row_min) / diff
            return scaled

        values = np.asarray(values, dtype=np.float64)
        if column_normalize:
            sums = values.sum(axis=0, keepdims=True)
            values = values / np.where(sums == 0, 1.0, sums)
        return np.ascontiguousarray(min_max_scale(values))

    def angle_threshold(n, angle):
        def standard_arc(n, angle):
            """Standard line and its reference arc; angle is in degrees."""
            x = np.linspace(0.0, 1.0, n)
            return x, arc_values(x, angle)

        x, y = standard_arc(n, angle)
        return slope_distance_np(x, y)

    def group_threshold(scaled, angle):
        """
        The group's adaptive threshold.
        Standard OTU s: every object's min-max values sorted increasingly, summed rank by rank into a virtual sum object,
        and the sum min-max scaled. Its deviated OTU is the arc of the given angle taken at the standard OTU's values,
        f(s_k), and the threshold is the slope distance between s and f(s).
        If the standard OTU is the straight line, the threshold is angle_threshold(n, angle).
        """
        total = np.sort(scaled, axis=1).sum(axis=0)
        span = total.max() - total.min()
        if span == 0:
            raise ValueError("Every object is constant, so there is no standard OTU")
        standard = (total - total.min()) / span
        deviated = arc_values(standard, angle)
        curve = pd.DataFrame({'rank': np.arange(1, standard.size + 1), 'line': np.linspace(0.0, 1.0, standard.size),
                              'standard_otu': standard, 'deviated_otu': deviated})
        return slope_distance_np(standard, deviated), curve

    start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Compare networks using each group's adaptive threshold")
    parser.add_argument('-in1', required=True, help='First input CSV file path')
    parser.add_argument('-in2', required=True, help='Second input CSV file path')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    parser.add_argument('-pos1', type=float, default=None,
                        help="Positive correlation threshold of file 1 (default: its adaptive threshold)")
    parser.add_argument('-pos2', type=float, default=None,
                        help="Positive correlation threshold of file 2 (default: its adaptive threshold)")
    parser.add_argument('-a', '--angle', type=float, default=90.0, help='angle used for threshold calculation')
    parser.add_argument('-l', '--lam', type=float, default=1.0, help='Multiplier for the threshold of both groups')
    parser.add_argument('--column-normalize', action='store_true',
                        help='Normalize each sample column to relative abundance before row min-max')
    args = parser.parse_args()

    input_file1 = args.in1
    input_file2 = args.in2
    output_file = args.output

    # try:
    #     df1 = pd.read_csv(input_file1, header=None)
    #     df2 = pd.read_csv(input_file2, header=None)
    # except Exception as e:
    #     print(f"Error reading inputs: {e}")
    #     sys.exit(1)
    # df1 = pd.read_csv(input_file1, header=None)
    # df2 = pd.read_csv(input_file2, header=None)
    df1 = pd.read_csv(input_file1, header=None, dtype={0: str})
    df2 = pd.read_csv(input_file2, header=None, dtype={0: str})

    df1_p = df1.copy()
    df2_p = df2.copy()

    col_num1 = len(df1_p.columns)
    col_num2 = len(df2_p.columns)

    # Each file's threshold now comes from its own standard OTU after scaling (see group_threshold above);
    # -pos1 / -pos2 set it directly.
    # Earlier single thresholds per file:
    # pos_cor1 = col_num1 - 2
    # pos_cor1 = (0.132494583836701*((col_num1-1)**2) + 0.395799799362735*(col_num1-1) - 2.74115847415123*(1/(col_num1-1)))
    # pos_cor1 = angle_threshold(col_num1 - 1, args.angle)

    filter_set = [1.0]

    # object_names = df1_p.iloc[:, 0].values
    names1 = df1_p.iloc[:, 0].values
    names2 = df2_p.iloc[:, 0].values
    numeric_data1 = df1_p.iloc[:, 1:].to_numpy(dtype=np.float64)
    numeric_data2 = df2_p.iloc[:, 1:].to_numpy(dtype=np.float64)

    # Each file's adaptive threshold (or -pos1 / -pos2), the same for every object of that file
    scaled1 = prepare_values(numeric_data1, args.column_normalize)
    scaled2 = prepare_values(numeric_data2, args.column_normalize)
    adaptive1, standard1 = group_threshold(scaled1, args.angle)
    adaptive2, standard2 = group_threshold(scaled2, args.angle)
    pos_cor1 = args.lam * (adaptive1 if args.pos1 is None else args.pos1)
    pos_cor2 = args.lam * (adaptive2 if args.pos2 is None else args.pos2)
    for label, thr, adaptive, given, scaled in (("file 1", pos_cor1, adaptive1, args.pos1, scaled1),
                                                ("file 2", pos_cor2, adaptive2, args.pos2, scaled2)):
        print(f"Threshold {label}: {thr:.6f} ({'custom' if given is not None else 'adaptive'}, lambda {args.lam:g}; "
              f"adaptive {adaptive:.6f}; straight line with the {args.angle:g} degree arc "
              f"{angle_threshold(scaled.shape[1], args.angle):.6f})")
    output_path = Path(output_file)
    standard_file = output_path.with_name(output_path.stem + "_standard_otu.csv")
    pd.concat([standard1.assign(file=1, threshold=pos_cor1), standard2.assign(file=2, threshold=pos_cor2)])[
        ["file", "rank", "line", "standard_otu", "deviated_otu", "threshold"]].to_csv(standard_file, index=False)
    print(f"Standard OTUs and thresholds saved to {standard_file}")

    results_df = pd.DataFrame()

    for idx, fval in enumerate(filter_set):
        print("Processing direct comparison ...")
        dist_matrix1 = build_distance_matrix(scaled1)
        dist_matrix2 = build_distance_matrix(scaled2)

        object_names, degrees_arr = compute_union_degrees(
            dist_matrix1, dist_matrix2, names1, names2, pos_cor1, pos_cor2
        )
        edges = np.sum(degrees_arr) // 2

        header_row = [("Direct Comparison", "")]
        data_rows = [("Edges", edges)]
        data_edges = list(zip(object_names, degrees_arr))
        data_edges.sort(key=lambda x: x[1], reverse=True)
        data_rows.extend(data_edges)

        temp_df = pd.DataFrame(header_row + data_rows, columns=["Object", f"Score_{idx + 1}"])
        temp_df["Blank"] = ""

        start_col = idx * 3
        temp_df.columns = [
            f"Column_{start_col}",
            f"Column_{start_col + 1}",
            f"Column_{start_col + 2}"
        ]
        results_df = pd.concat([results_df, temp_df], axis=1)

    # try:
    #     results_df.to_csv(output_file, index=False, header=False)
    #     print(f"Results successfully saved to {output_file}")
    # except Exception as e:
    #     print(f"Error writing output file: {e}")
    #     sys.exit(1)
    results_df.to_csv(output_file, index=False, header=False)
    print(f"Results successfully saved to {output_file}")

    print(f"total time needed: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
