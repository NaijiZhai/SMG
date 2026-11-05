# SMG: Slope-Matrix-Graph

Python scripts for the Slope-Matrix-Graph (SMG) algorithm for microbiome data. Work in progress: options and defaults may change.

The original SMG method is described in: Zhang M, Li X, Oladeinde A, Rothrock M Jr, Pokoo-Aikins A, Zock G. A Novel Slope-Matrix-Graph Algorithm to Analyze Compositional Microbiome Data. *Microorganisms*. 2024;12(9):1866. https://doi.org/10.3390/microorganisms12091866

## Scripts

| Script | What it does | Output |
|---|---|---|
| `cluster-11-05.py` | OTU network of one group and its connected components | Excel file: distance matrix colored by component, plus the sheets `Threshold` and `Standard OTU` |
| `DAA-11-05.py` | Compares the OTU networks of two groups: for each OTU, the number of edges present in one network but not the other | CSV file, plus `<output>_standard_otu.csv` |
| `BDA-09-03.py` | Sample–sample SMG distance matrix | Excel file |

## Input

CSV files without a header row. The first column holds the OTU or sample ID; the other columns hold the values (relative abundances). `cluster-11-05.py` and `DAA-11-05.py` take one row per OTU (DAA takes one file per group); `BDA-09-03.py` takes one row per sample.

## Threshold

`cluster-11-05.py` and `DAA-11-05.py` give each group one threshold:

1. Each OTU is min-max scaled and its values are sorted.
2. The sorted values are summed rank by rank, and the sum is min-max scaled: this is the group's standard OTU.
3. The 90° arc taken at the standard OTU's values is the deviated OTU.
4. The threshold is the SMG distance between the standard and the deviated OTU. Two OTUs are connected when their distance is below it.

`-a` changes the angle, `-l` multiplies the threshold, and `-t` (cluster) or `-pos1`/`-pos2` (DAA) set it directly. `--column-normalize` rescales each sample to sum to 1 before the min-max scaling.

## Usage

```bash
python cluster-11-05.py -i group.csv -o clusters.xlsx
python DAA-11-05.py -in1 group1.csv -in2 group2.csv -o changed_edges.csv
python BDA-09-03.py -i samples.csv -o distances.xlsx
```

## Requirements

Python 3 with numpy, pandas, numba and openpyxl.
