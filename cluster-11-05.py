

import math
from collections import defaultdict
import re
import pandas as pd
import numpy as np
import sys
import os
import colorsys
from openpyxl.styles import PatternFill
import argparse
import multiprocessing
from numba import njit
import time

@njit(fastmath=True, nogil=True)
def slope_distance_numba(row1, row2):
    """Numba-optimized slope distance calculation"""
    v = row1 - row2
    v_sorted = np.sort(v)
    n = len(v_sorted)
    distance = 0
    for k in range(n):
        distance += (2 * k - n + 1) * v_sorted[k]
    return 2.0 * distance

def arc_values(x, angle):
    """Heights of the reference arc of the given angle (degrees) at the abscissas x in [0, 1]."""
    t = np.tan(np.deg2rad(angle) / 2)
    return 2*x*(1-t+t*x) / (1+t+np.sqrt((1-t)**2+4*t*(1-x)*(1+t*x)))

def process_row(args):

    # Separate function for processing rows to enable multiprocessing

    df_slice, start, end = args
    results = []
    n=df_slice.shape[0]
    for i in range(start, end):
        row_i = df_slice[i]
        for j in range(i + 1, n):
            dist = slope_distance_numba(row_i, df_slice[j])
            results.append((i, j, dist))
    return results

class UnionFind:
    """Optimized UnionFind with path compression"""
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x):
        """Path compression find method"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union by rank"""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

def main():
    def check_blank(data, file_name):
        """Check for blank rows"""
        if data.iloc[:, 0].isnull().any():
            raise ValueError(f"Blank row in '{file_name}'")

    def check_duplicate(data, file_name):
        """Check for duplicate entries"""
        if data.iloc[:, 0].duplicated().any():
            raise ValueError(f"Duplicate object in '{file_name}'")

    def prepare_values(values, column_normalize=False):
        """Optional column normalization, followed by row min-max."""
        values = np.asarray(values, dtype=np.float64)
        if column_normalize:
            sums = values.sum(axis=0, keepdims=True)
            values = values / np.where(sums == 0, 1.0, sums)
        low = values.min(axis=1, keepdims=True)
        span = values.max(axis=1, keepdims=True) - low
        return np.ascontiguousarray((values - low) / np.where(span == 0, 1.0, span))

    def angle_threshold(n, angle):
        def standard_arc(n, angle):
            """Standard line and its reference arc; angle is in degrees."""
            x = np.linspace(0.0, 1.0, n)
            return x, arc_values(x, angle)

        x, y = standard_arc(n, angle)
        return slope_distance_numba(x, y)

    def group_threshold(values, angle):
        """The group's adaptive threshold.
        Standard OTU s: every OTU's min-max values sorted increasingly, summed rank by rank into a virtual sum OTU,
        and the sum min-max scaled. Its deviated OTU is the arc of the given angle taken at the standard OTU's values,
        f(s_k), and the threshold is the slope distance between s and f(s).
        If the standard OTU is the straight line, the threshold is angle_threshold(n, angle)."""
        total = np.sort(values, axis=1).sum(axis=0)
        span = total.max() - total.min()
        if span == 0:
            raise ValueError("Every OTU is constant, so there is no standard OTU")
        standard = (total - total.min()) / span
        deviated = arc_values(standard, angle)
        curve = pd.DataFrame({'Rank': np.arange(1, standard.size + 1), 'Line': np.linspace(0.0, 1.0, standard.size),
                              'Standard OTU': standard, 'Deviated OTU': deviated})
        return slope_distance_numba(standard, deviated), curve

    def parallel_slope_distance(df, thresholds):
        """Parallel processing of slope distance calculations; an edge needs the distance below the threshold"""
        n = df.shape[0]

        dist_matrix = np.zeros((n, n), dtype=np.float64)
        uf = UnionFind(n)
        if n < 2:
            return dist_matrix, uf

        # Chunk up the rows among CPU cores
        cpu_core_count = min(os.cpu_count() or 1, n)
        chunk_size = (n // cpu_core_count) + 1
        args_list = []
        start = 0
        for _ in range(cpu_core_count):
            end = min(start + chunk_size, n)
            if start < n:
                args_list.append((df, start, end))
            start = end

        with multiprocessing.Pool(len(args_list)) as pool:
            all_results = pool.map(process_row, args_list)

        # Combine results, fill in dist_matrix and do union
        for results in all_results:
            for (i, j, dist) in results:
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
                if dist < min(thresholds[i], thresholds[j]):
                    uf.union(i, j)

        return dist_matrix, uf

    def save_colored_network(dist_matrix, groups, thresholds, output_file, names, threshold_info, standard_otu):
        def generate_unique_colors(num_clusters):
            """Generate a large number of unique colors using HSV color space"""
            colors = []
            for i in range(num_clusters):
                hue = (0.618033988749895 * i) % 1.0
                saturation = 0.5 + (0.5 * (i % 2))
                value = 0.7 + (0.3 * ((i // 2) % 2))

                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                hex_color = ''.join([f'{int(x * 255):02X}' for x in rgb])

                colors.append(hex_color)

            return colors

        def extract_numeric_suffix(name: str) -> int:
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='ignore')
            else:
                name = str(name)
            match = re.search(r'(\d+)$', name)
            if match:
                return int(match.group(1))
            return 0

        # Generate colors and sort the groups
        colors = generate_unique_colors(len(groups))
        group_list = list(groups.items())
        group_list.sort(key=lambda x: min(extract_numeric_suffix(m) for m in x[1]))
        new_order = []
        for _, members in group_list:
            members.sort(key=lambda x: extract_numeric_suffix(x)) #sort inside
            new_order.extend(members)
        dist_matrix = pd.DataFrame(dist_matrix, index=names, columns=names)
        df_dist = dist_matrix.loc[new_order, new_order]
        thr = pd.Series(thresholds, index=names)

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Convert dist_matrix to DataFrame for easier Excel writing
            df_dist.to_excel(writer, sheet_name='Distance Matrix', index=True)
            threshold_info.to_excel(writer, sheet_name='Threshold', index=True)
            standard_otu.to_excel(writer, sheet_name='Standard OTU', index=False)
            worksheet = writer.book['Distance Matrix']

            # Fill colors
            color_index = 0
            for group_key, members in groups.items():
                if len(members) >= 2:
                    fill = PatternFill(
                        start_color=colors[color_index],
                        end_color=colors[color_index],
                        fill_type='solid'
                    )
                    color_index += 1

                    for member in members:
                        row_idx = df_dist.index.get_loc(member) + 2
                        # +2 for 1-based Excel index plus header row
                        name_cell = worksheet.cell(row=row_idx, column=1)
                        name_cell.fill = fill

                        # Color the threshold connections
                        for other in df_dist.columns:
                            if other != member:
                                dist_val = df_dist.loc[member, other]
                                if dist_val < min(thr[member], thr[other]):
                                    col_idx = df_dist.columns.get_loc(other) + 2
                                    cell = worksheet.cell(row=row_idx, column=col_idx)
                                    cell.fill = fill

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Cluster OTUs using the group's adaptive threshold")
    parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    parser.add_argument('-o', '--output', required=True, help='Output Excel file')
    parser.add_argument('-t', '--threshold', type=float,
                        help="Custom threshold (optional; default: the group's adaptive threshold)")
    parser.add_argument('-a', '--angle', type=float, default=90.0, help='angle used for threshold calculation')
    parser.add_argument('-l', '--lam', type=float, default=1.0, help='Multiplier for the threshold')
    parser.add_argument('--column-normalize', action='store_true',
                        help='Normalize each sample column to relative abundance before row min-max')
    args = parser.parse_args()


    df = pd.read_csv(args.input, header=None, dtype={0: str})

    names = df.iloc[:, 0].values
    # Check data prerequisites silently
    check_blank(df, args.input)
    check_duplicate(df, args.input)

    df_values = prepare_values(df.iloc[:, 1:].to_numpy(dtype=np.float64), args.column_normalize)

    # The group's adaptive threshold, or a custom one; the same for every OTU
    adaptive, standard_otu = group_threshold(df_values, args.angle)
    line_threshold = angle_threshold(df_values.shape[1], args.angle)
    kind = 'custom' if args.threshold is not None else 'adaptive'
    threshold = args.lam * (args.threshold if args.threshold is not None else adaptive)
    thresholds = np.full(df_values.shape[0], threshold)
    print(f"Using {kind} threshold: {threshold:.6f} (lambda {args.lam:g}; adaptive {adaptive:.6f}; "
          f"straight line with the {args.angle:g} degree arc {line_threshold:.6f})")
    threshold_info = pd.Series({'Threshold used': threshold, 'Kind': kind, 'Lambda': args.lam,
                                'Adaptive threshold': adaptive, 'Angle': args.angle,
                                'Straight line threshold': line_threshold, 'Samples': df_values.shape[1],
                                'OTUs': df_values.shape[0],
                                'Non-constant OTUs': int((df_values.max(axis=1) > df_values.min(axis=1)).sum())},
                               name='Value')

    # calculate distance matrix and do uf in one run
    dist_matrix, uf = parallel_slope_distance(df_values, thresholds)

    n = df_values.shape[0]
    groups_dict = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        groups_dict[root].append(names[i])

    # Save the colored network
    save_colored_network(dist_matrix, groups_dict, thresholds=thresholds, output_file=args.output, names=names,
                         threshold_info=threshold_info, standard_otu=standard_otu)

    # Print cluster information
    print("\nClustering Results:")
    for i, (key, group) in enumerate(groups_dict.items(), 1):
        if len(group) >= 2:
            print(f"Cluster {i}: {len(group)} members")



if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"run time: {end - start} s (perf_counter）")
