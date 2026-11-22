# Axelrod Culture Model Replication

Python implementation of Robert Axelrod's cultural dissemination model, used to reproduce Table 2 and Figure 2 from his 1997 paper.

Agents occupy an `n × n` grid. Each agent has `F` cultural features, each with `q` possible values. Neighboring agents may interact depending on how similar they are. A run ends when no further cultural changes are possible.

## Setup

Python 3.11 and the packages in `requirements.txt` are required. In PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running the project

```powershell
.\.venv\Scripts\python.exe -m tests.test_model
.\.venv\Scripts\python.exe -m experiments.table2
.\.venv\Scripts\python.exe -m experiments.table2_summary
.\.venv\Scripts\python.exe -m experiments.table2_dynamics_summary
.\.venv\Scripts\python.exe -m experiments.table2_plots
.\.venv\Scripts\python.exe -m experiments.figure2
.\.venv\Scripts\python.exe -m experiments.figure2_summary
.\.venv\Scripts\python.exe -m experiments.figure2_plots
```

Running `experiments.table2` again overwrites `data/table2/table2_runs.csv`. The Figure 2 script skips runs already saved in its CSV file.

## Results

Table 2: 90 runs on a 10 × 10 grid, with `F` and `q` set to 5, 10 or 15. For `F=5, q=15`, the mean was 21.7 stable regions (20.0 in Axelrod's table).

Figure 2: 380 runs with `F=5`, `q=15` and grid sizes from 2 × 2 to 100 × 100. The highest mean was 21.3 regions at `n=12`; at `n=100`, it was 3.2.

CSV files are in `data/`, and plots are in `plots/`. The model is in `model.py`, with tests in `tests/`. Fixed random seeds are used. Values from Axelrod's Figure 2 are approximate readings of the published graph.

## Reference

Axelrod, R. (1997). The Dissemination of Culture: A Model with Local Convergence and Global Polarization. *Journal of Conflict Resolution, 41*(2), 203–226. https://doi.org/10.1177/0022002797041002001

Authors:
Mateusz Pandura, Marcin Kochalski

Polish version: [README_PL.md](README_PL.md).
