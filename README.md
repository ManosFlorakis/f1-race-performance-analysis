# Analysis and Prediction of Performance in F1 Races

Statistical and machine-learning analysis accompanying the study by Emmanouil Florakis, Konstadinos Politis, and Apostolos Bozikas, University of Piraeus.

The analysis combines descriptive statistics, circuit-level time-series modelling, multinomial logistic regression, principal component analysis, Random Forest, support vector machines, K-Means, and hierarchical clustering. The main dataset covers 2011–2024; the circuit-specific time series extend back to 2004.

## Repository structure

```text
f1-race-performance-analysis/
├── README.md
├── requirements.txt
├── f1_analysis.ipynb
├── f1_analysis.py
└── data/
    ├── working_f1_data_V5.xlsx
    ├── circuits_overtakes.xlsx
    ├── Nationalities.xlsx
    └── original CSV files
```

`f1_analysis.ipynb` contains all analysis cells with saved outputs and execution counts cleared. Running the cells generates the numerical results and figures, including the interactive violin plots. `f1_analysis.py` provides the corresponding code as a script. Generated Excel files are written to `outputs/`, which is created at runtime.

## Environment

The author reported Python **3.13.16** and the exact package versions listed in `requirements.txt` from the Google Colab environment used for the revised analysis.

For local execution, clone the repository and install the dependencies:

```bash
git clone https://github.com/ManosFlorakis/f1-race-performance-analysis.git
cd f1-race-performance-analysis
python -m pip install -r requirements.txt
```

Open `f1_analysis.ipynb` in a Jupyter environment, or run:

```bash
python f1_analysis.py
```

Jupyter is optional notebook tooling and is not included in the analysis dependency list. The supplied package versions record the author's environment; they are not a claim that a fresh installation and complete execution have been verified.

## Google Colab

Use the repository root as the working directory. In a new Colab session, run:

```python
!git clone https://github.com/ManosFlorakis/f1-race-performance-analysis.git
%cd /content/f1-race-performance-analysis
!python -m pip install -r requirements.txt
```

Then run the analysis script:

```python
%run f1_analysis.py
```

Alternatively, open the notebook, run the setup commands above once, and execute its analysis cells. Restart the runtime after dependency installation if Colab has already imported packages that were replaced, then return to the repository directory before executing the notebook.

## Data

The public race-data source is the [Formula 1 World Championship dataset on Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020). Use the supplied snapshot to reproduce the study; downloading a newer snapshot can change the observations and results.

The code reads these CSV files from `data/`:

```text
drivers.csv
constructors.csv
results.csv
races.csv
lap_times.csv
pit_stops.csv
qualifying.csv
driver_standings.csv
constructor_standings.csv
seasons.csv
circuits.csv
status.csv
```

`constructor_results.csv` and `sprint_results.csv` are retained with the source snapshot but are not read by the present code.

Three Excel files are also required:

- `working_f1_data_V5.xlsx`: the original filtered analytical checkpoint, containing cumulative points through the current race.
- `circuits_overtakes.xlsx`: circuit overtaking information used for grouping circuits.
- `Nationalities.xlsx`: the nationality mapping used in preprocessing.

The auxiliary Excel files and working checkpoint are supplied separately from the original CSV source. Preserve their filenames and capitalization.

## Execution from the checkpoint

To reproduce the revised modelling analyses without repeating the earlier descriptive analysis:

1. Execute the notebook's library-import, path-setup, and random-seed cell.
2. Execute the code cells from **Chapter 5: Regression** onward in order.

This route reads the original checkpoint from `data/`, applies the correction to cumulative points, and continues through regression, circuit-level time-series analysis, PCA, classification, and clustering. The time-series section also requires `results.csv`, `races.csv`, and `circuits.csv`.

For a full run, execute every code cell sequentially. Preprocessing writes a reconstructed checkpoint into `outputs/`, and subsequent sections use that reconstructed file. A fresh session starts with the supplied checkpoint in `data/` as the default.

## Correction of cumulative season points

At the start of Chapter 5, the current race's points are subtracted from the original cumulative total:

```python
working_df['cumulative_season_driver_points'] = (
    working_df['cumulative_season_driver_points']
    - working_df['driver_points']
)
```

The corrected value contains points earned only in earlier races of the same season. Because the original cumulative total was constructed before filtering observations, subtraction retains historical points from races missing from the filtered checkpoint.

Apply this correction once after loading the original checkpoint. Reload the checkpoint before repeating the correction. Earlier descriptive and correlation sections use the original cumulative values; Chapter 5 and the subsequent modelling sections use the corrected values.

The finishing-count variables `top_3_positions`, `positions_4_10`, and `outside_10_positions` count results from previous seasons only, using `year < current year`.

## Seeds and evaluation

- NumPy and Python random seeds: `1234`.
- Regression split: random 75% training / 25% test, with `random_state=1234`.
- Random Forest and SVM split: random 80% training / 20% test, with `random_state=1234`.
- Classifier cross-validation: five folds on the training subset.
- K-Means: three clusters, with `random_state=1234`.

Use the same input snapshot, row order, and package versions when comparing results. Cluster labels are arbitrary identifiers and do not directly correspond to the three race-outcome categories.

## Output files

The notebook displays summaries and figures inline; the script prints summaries and displays plots. Generated intermediate files include:

```text
outputs/final_f1_data_V16.xlsx
outputs/working_f1_data_V5.xlsx
outputs/circuits_overtakesV5.xlsx
```

The files supplied in `data/` are used as inputs and are not overwritten by these exports.
