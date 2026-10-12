# -*- coding: utf-8 -*-
"""Analysis and Prediction of Performance in F1 Races.

Python export of f1_analysis.ipynb. See README.md for data and execution instructions.
"""

#Neccesary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import plotly.express as px
import scipy.stats as stats
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, auc
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import random
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy.stats import anderson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Repository input and generated-output locations
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA_DIR = REPO_DIR / 'data'
OUTPUT_DIR = REPO_DIR / 'outputs'

if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        'The data directory was not found. Run the notebook from the repository root.'
    )

OUTPUT_DIR.mkdir(exist_ok=True)
WORKING_CHECKPOINT = DATA_DIR / 'working_f1_data_V5.xlsx'

np.random.seed(1234)
random.seed(1234)

# Seaborn style for better visualizations
sns.set(style="whitegrid")

# Loading Data from the CSV files
drivers = pd.read_csv(DATA_DIR / "drivers.csv")
constructors = pd.read_csv(DATA_DIR / "constructors.csv")
results = pd.read_csv(DATA_DIR / "results.csv")
races = pd.read_csv(DATA_DIR / "races.csv")
lap_times = pd.read_csv(DATA_DIR / "lap_times.csv")
pit_stops = pd.read_csv(DATA_DIR / "pit_stops.csv")
qualifying = pd.read_csv(DATA_DIR / "qualifying.csv")
driver_standings = pd.read_csv(DATA_DIR / "driver_standings.csv")
constructor_standings = pd.read_csv(DATA_DIR / "constructor_standings.csv")
seasons = pd.read_csv(DATA_DIR / "seasons.csv")
circuits = pd.read_csv(DATA_DIR / "circuits.csv")
status = pd.read_csv(DATA_DIR / "status.csv")

"""FIGURES FOR THEORY PART CHAPTER 2

Wins per Driver
"""

# Filter results for position 1
wins_per_driver = results[results["positionOrder"] == 1].groupby("driverId").size()

# Merge with driver names
wins_per_driver = wins_per_driver.reset_index()
wins_per_driver.columns = ["driverId", "wins"]
wins_per_driver = wins_per_driver.merge(drivers[["driverId", "surname"]], on="driverId")

# Sort and display top 10 drivers
top_drivers = wins_per_driver.sort_values(by="wins", ascending=False).head(10)

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x="wins", y="surname", data=top_drivers, palette="viridis")
plt.title("Top 10 Drivers by Wins")
plt.xlabel("Number of Wins")
plt.ylabel("Driver")
plt.show()

"""Points per Team"""

# Total points per team
points_per_team = results.groupby("constructorId")["points"].sum().reset_index()
points_per_team = points_per_team.merge(constructors[["constructorId", "name"]], on="constructorId")

# Sort and display top 10 teams
top_teams = points_per_team.sort_values(by="points", ascending=False).head(10)

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x="points", y="name", data=top_teams, palette="coolwarm")
plt.title("Top 10 Teams by Points")
plt.xlabel("Total Points")
plt.ylabel("Team")
plt.show()

"""Number of Races per Year"""

# Number of races per year
races_per_year = races.groupby("year").size().reset_index(name="race_count")

# Visualization
plt.figure(figsize=(12, 6))
sns.lineplot(x="year", y="race_count", data=races_per_year, marker="o")
plt.title("Number of Races per Year")
plt.xlabel("Year")
plt.ylabel("Number of Races")
plt.show()

"""Fastest Lap Timeseries"""

# Convert lap times to numeric
lap_times["milliseconds"] = pd.to_numeric(lap_times["milliseconds"], errors="coerce")

# Calculate average lap time per circuit and year
avg_lap_time_per_circuit = lap_times.groupby(["raceId"])["milliseconds"].mean().reset_index()
avg_lap_time_per_circuit = avg_lap_time_per_circuit.merge(races[["raceId", "year", "circuitId"]], on="raceId")
avg_lap_time_per_circuit = avg_lap_time_per_circuit.merge(circuits[["circuitId", "name"]], on="circuitId")

# Filter only for Albert Park Grand Prix Circuit
filtered_circuits = ["Albert Park Grand Prix Circuit"]
avg_lap_time_filtered = avg_lap_time_per_circuit[avg_lap_time_per_circuit["name"].isin(filtered_circuits)]

# Calculate average lap time per circuit and year
avg_lap_time_filtered = avg_lap_time_filtered.groupby(["year", "name"])["milliseconds"].mean().reset_index()

# Visualization
plt.figure(figsize=(12, 6))
sns.lineplot(x="year", y="milliseconds", data=avg_lap_time_filtered, marker="o")
plt.title("Average Lap Time: Albert Park Grand Prix Circuit", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Lap Time (milliseconds)", fontsize=12)
plt.tight_layout()
plt.show()

"""Pit Stops Analysis"""

# Calculate the average number of pit stops per race per year
pit_stops_per_year = pit_stops.merge(races[["raceId", "year"]], on="raceId")
pit_stops_per_race = (pit_stops_per_year.groupby("year").size()  # Total pit stops per year
                     / races.groupby("year").size()              # Total races per year
                     ).reset_index(name="average_pit_stops_per_race")

# Only for finished periods
pit_stops_per_race = pit_stops_per_race[pit_stops_per_race["year"] <= 2023]

# Visualization
plt.figure(figsize=(12, 6))
sns.lineplot(x="year", y="average_pit_stops_per_race", data=pit_stops_per_race, marker="o")
plt.title("Average Pit Stops per Race per Year")
plt.xlabel("Year")
plt.ylabel("Average Pit Stops per Race")
plt.grid(True)
plt.show()

"""Creating a unified dataset for our analysis's purposes"""

# Grouping data by race and driver, calculating total pit stops and total duration
Cumulative_pit_stops = pit_stops.groupby(['raceId', 'driverId']).agg(
    total_pit_stops=('stop', 'count'),        # Total pit stops for the driver in the race
    total_duration_ms=('milliseconds', 'sum') # Total duration in milliseconds
).reset_index()

# Updated selection of important columns
drivers_selected = drivers[['driverId', 'forename', 'surname', 'nationality', 'dob']]
constructors_selected = constructors[['constructorId', 'name', 'nationality']]
races_selected = races[['raceId', 'date', 'circuitId', 'year', 'round']]
qualifying_selected = qualifying[['raceId', 'driverId', 'position', 'q1', 'q2', 'q3']]
results_selected = results[['raceId', 'driverId', 'constructorId', 'grid', 'points', 'fastestLapTime', 'statusId', 'laps', 'positionOrder']]
circuits_selected = circuits[['circuitId', 'name']]
pit_stops_selected = Cumulative_pit_stops[['raceId', 'driverId', 'total_pit_stops','total_duration_ms']]

# Renaming columns for clarity
drivers_selected = drivers_selected.rename(columns={
    'forename': 'driver_forename',
    'surname': 'driver_surname',
    'nationality': 'driver_nationality'
})
constructors_selected = constructors_selected.rename(columns={
    'name': 'constructor_name',
    'nationality': 'constructor_nationality'
})
races_selected = races_selected.rename(columns={
    'date': 'race_date'
})
qualifying_selected = qualifying_selected.rename(columns={
    'position': 'qualifying_position'
})
results_selected = results_selected.rename(columns={
    'grid': 'grid_position',
    'points': 'driver_points',
    'fastestLapTime': 'fastest_lap_time',
    'statusId': 'race_status'
})
circuits_selected = circuits_selected.rename(columns={
    'name': 'circuit_name'})

# Merging DataFrames
df_merged = results_selected.merge(qualifying_selected, on=['raceId', 'driverId'], how='left') \
                             .merge(drivers_selected, on='driverId', how='left') \
                             .merge(constructors_selected, on='constructorId', how='left') \
                             .merge(races_selected, on='raceId', how='left') \
                             .merge(circuits_selected, on='circuitId', how='left') \
                             .merge(pit_stops_selected, on=['raceId', 'driverId'], how='left')

# Selecting the final columns
final_columns = ['driverId', 'driver_forename', 'driver_surname', 'driver_nationality', 'dob',
                 'constructorId', 'constructor_name', 'constructor_nationality',
                 'raceId', 'race_date','circuitId','circuit_name','year', 'round', 'qualifying_position', 'q1', 'q2', 'q3',
                 'grid_position','driver_points', 'fastest_lap_time', 'race_status','laps', 'positionOrder',
                 'total_pit_stops','total_duration_ms']

# Creating the final DataFrame
final_df = df_merged[final_columns]

"""Making one column for the drivers"""

# Concatenate 'driver_forename' and 'driver_surname' with a space in between
final_df['driver_name'] = final_df['driver_forename'].astype(str) + ' ' + final_df['driver_surname'].astype(str)

# Drop the original columns
final_df = final_df.drop(columns=['driver_forename', 'driver_surname'])

"""Sort by Race date"""

# Sorting the final DataFrame by 'race_date' in ascending order
final_df = final_df.sort_values(by='race_date').reset_index(drop=True)

"""Additional Columns"""

# Convert 'dob' to datetime format if not already done
final_df['dob'] = pd.to_datetime(final_df['dob'], errors='coerce')

# Calculate Age for each race
final_df['age'] = final_df['year'] - final_df['dob'].dt.year

# Calculate Years of Experience
# Find the first year each driver participated in a race
first_race_year = final_df.groupby('driverId')['year'].transform('min')
final_df['years_experience'] = final_df['year'] - first_race_year

# Calculate cumulative race starts for each driver within the season up to each race
final_df['race_starts'] = final_df.groupby('driverId').cumcount() + 1

# Calculate cumulative points for each driver within the season up to each race
final_df['cumulative_season_driver_points'] = final_df.groupby(['year', 'driverId'])['driver_points'].cumsum()

"""Adding the Constructor with the most years of co-working with each driver to each date."""

# First the necessary fields, excluding the year of the current race to only count previous collaborations
driver_constructor_years = final_df[['driverId', 'constructor_name', 'year']].drop_duplicates()
driver_constructor_years['years_with_constructor'] = driver_constructor_years.groupby(['driverId', 'constructor_name'])['year'].cumcount()

# Merging with the final DataFrame to add the cumulative collaboration information
final_df = final_df.merge(driver_constructor_years[['driverId', 'constructor_name', 'year', 'years_with_constructor']], on=['driverId', 'constructor_name', 'year'], how='left')

# Finding the constructor with the most cumulative collaboration for each driver up to the previous year
# Ignoring the current year when calculating the dominant constructor
def get_top_constructor(row):
    driver_data = final_df[(final_df['driverId'] == row['driverId']) & (final_df['year'] < row['year'])]
    if not driver_data.empty:
        return driver_data.groupby('constructor_name')['years_with_constructor'].max().idxmax()
    return row['constructor_name']

# Applying the function to each row of the DataFrame
final_df['top_constructor'] = final_df.apply(get_top_constructor, axis=1)

"""Adding Top3, Top10 (4-10) and OutsideTop10"""

# Create a new function to calculate the positions
def calculate_positions(row):
    # Filter the data for the driver up to the current year
    driver_data = final_df[(final_df['driverId'] == row['driverId']) & (final_df['year'] < row['year'])]

    # Calculate the positions in the Top 3 category (positions 1, 2, 3)
    top_3_count = driver_data[driver_data['positionOrder'].isin([1, 2, 3])].shape[0]

    # Calculate the positions in the 4th to 10th category (positions 4-10)
    top_10_count = driver_data[(driver_data['positionOrder'] >= 4) & (driver_data['positionOrder'] <= 10)].shape[0]

    # Calculate the positions outside the Top 10 (positions > 10)
    outside_10_count = driver_data[driver_data['positionOrder'] > 10].shape[0]

    return pd.Series([top_3_count, top_10_count, outside_10_count], index=['top_3_positions', 'positions_4_10', 'outside_10_positions'])

# Apply the function to each row of the DataFrame
final_df[['top_3_positions', 'positions_4_10', 'outside_10_positions']] = final_df.apply(calculate_positions, axis=1)

"""Creating column for further analysis"""

# Define a function to classify the race outcome
def classify_race_outcome(row):
    if row['positionOrder'] in [1, 2, 3]:
        return 'Podium finish'
    elif 4 <= row['positionOrder'] <= 10:
        return 'Points without podium'
    else:
        return 'No points'

# Apply the function to create the new 'race_outcome' column
final_df['race_outcome'] = final_df.apply(classify_race_outcome, axis=1)

"""Making our working df from 2011, to have exact metrics for qualifying and pitstops"""

# Filter final_df to include only rows where 'year' is 2011 or later
working_df = final_df[final_df['year'] >= 2011].copy()

"""Dropping rows with NaN that didn't contribute to the analysis"""

# Replace '\N' with NaN only in the 'fastest_lap_time' column
working_df['fastest_lap_time'].replace('\\N', np.nan, inplace=True)

# Drop rows with NaN in any of the specified columns
working_df.dropna(subset=['qualifying_position','fastest_lap_time', 'total_pit_stops', 'total_duration_ms'], inplace=True)

# Verify that there are no more NaN values in the specified columns
print(working_df[['qualifying_position','fastest_lap_time', 'total_pit_stops', 'total_duration_ms']].isnull().sum())

"""Dealing with NaNs from qualifying sessions"""

# Replace '\N' with NaN in q1, q2, and q3 columns to standardize missing values
working_df['q1'].replace('\\N', np.nan, inplace=True)
working_df['q2'].replace('\\N', np.nan, inplace=True)
working_df['q3'].replace('\\N', np.nan, inplace=True)

# Add binary indicators for participation in Q1, Q2, and Q3 (to not be misinterpreted as an actual time)
working_df['valid_time_set_in_q1'] = working_df['q1'].notna().astype(int)
working_df['qualified_for_q2'] = working_df['q2'].notna().astype(int)
working_df['qualified_for_q3'] = working_df['q3'].notna().astype(int)

# Replace NaN values in Q2 and Q3 with '0:00.000' as the time for non-participation
working_df['q1'].fillna('0:00.000', inplace=True)
working_df['q2'].fillna('0:00.000', inplace=True)
working_df['q3'].fillna('0:00.000', inplace=True)

"""Rearranging and keeping the needed columns"""

# Dropping specified columns
working_df = working_df.drop(['driverId', 'dob', 'constructorId', 'raceId', 'race_date', 'circuitId'], axis=1)

# Rearranging the remaining columns in a specific order
# Define the new column order as a list of column names
new_column_order = [
    'driver_name', 'driver_nationality', 'age', 'years_experience', 'race_starts',
    'constructor_name', 'constructor_nationality','years_with_constructor', 'top_constructor',
    'year', 'round', 'circuit_name', 'valid_time_set_in_q1','q1','qualified_for_q2','q2', 'qualified_for_q3','q3', 'qualifying_position',
    'grid_position','positionOrder','race_status','driver_points', 'cumulative_season_driver_points','fastest_lap_time', 'laps',
    'total_pit_stops', 'total_duration_ms','top_3_positions', 'positions_4_10', 'outside_10_positions', 'race_outcome',
]

# Reordering the DataFrame columns
working_df = working_df[new_column_order]

# Convert Q1, Q2, Q3, and Fastest_lap_time to seconds
def time_to_seconds(time_str):
    try:
        mins, secs = map(float, time_str.split(':'))
        return mins * 60 + secs
    except:
        return None

for col in ['q1', 'q2', 'q3', 'fastest_lap_time']:
    working_df[col] = working_df[col].apply(time_to_seconds)

# Grouping race_status variable

# Define groupings of numbers for categories
group_mapping = {
    'FINISHED': [1],
    'DSQ': [2, 96],
    'RETIRED': [31],
    'WITHDREW': [54],
    'DRIVER RELATED': [139],
    'LAPPED': [11,12,13,14,15,16,17,18,19,45,88,111],
    'RACE INCIDENT': [3,4,20,29,130,137],
    'CAR RELATED': [5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 47, 48, 51, 60, 61, 63, 65, 75, 76, 79, 84, 91, 95, 101, 103, 131, 132, 135, 136, 140, 141]
}

# Reverse the group_mapping to map each number to its category
rename_mapping = {num: category for category, nums in group_mapping.items() for num in nums}

# Map the 'race_status' column
working_df['race_status'] = working_df['race_status'].map(rename_mapping)

"""Checking the whole working_df for NaNs"""

# Check for NaN values across all columns
nan_counts = working_df.isna().sum()
print("NaN Counts:")
print(nan_counts[nan_counts > 0])  # Display only columns with NaN values

# Check for "\N" values across all columns
backslash_n_counts = (working_df == r'\N').sum()
print("\\N Counts:")
print(backslash_n_counts[backslash_n_counts > 0])  # Display only columns with \N values

print(working_df)

# Export merged_df to an Excel file in the current directory
final_df.to_excel(OUTPUT_DIR / "final_f1_data_V16.xlsx", index=False, engine="openpyxl")

# Export merged_df to an Excel file in the current directory
working_df.to_excel(OUTPUT_DIR / "working_f1_data_V5.xlsx", index=False, engine="openpyxl")
WORKING_CHECKPOINT = OUTPUT_DIR / 'working_f1_data_V5.xlsx'

"""Descriptive Statistics"""

# Read the created working excel
working_df = pd.read_excel(WORKING_CHECKPOINT)

"""Metrics"""

# Calculate descriptive statistics for numerical columns
descriptive_stats = working_df.describe(percentiles=[0.25, 0.5, 0.75]).T

# Keep only the columns that we are interested in
descriptive_stats = descriptive_stats[['min', '25%', '50%', 'mean', '75%', 'max']]

# Rename columns for clearer display
descriptive_stats.columns = ['Min', '1st Q (25%)', 'Median (50%)', 'Mean', '3rd Q (75%)', 'Max']

# Display the results
print(descriptive_stats)

"""Timeseries Plots"""

# Calculate the mean per year excluding zero values
def custom_mean(series):
    # Filter out zero values (0)
    valid_values = series[series > 0]
    return valid_values.mean()

# Grouping data by year
yearly_data = working_df.groupby('year').agg({
     'driver_points': 'mean',
    'cumulative_season_driver_points': 'mean',
    'fastest_lap_time': 'mean',
    'laps': 'mean',
    'total_pit_stops': 'mean',
    'total_duration_ms': 'mean',
    'top_3_positions': 'mean',
    'positions_4_10': 'mean',
    'outside_10_positions': 'mean'
}).reset_index()

# Defining variables for plotting
plot_vars = {
    'driver_points': 'Avg. Driver Points',
    'cumulative_season_driver_points': 'Avg. Cumulative Points',
    'fastest_lap_time': 'Avg. Fastest Lap (s)',
    'laps': 'Avg. Laps',
    'total_pit_stops': 'Avg. Total Pit Stops per Race',
    'total_duration_ms': 'Avg. Total Pit Stop Duration per Race (ms)',
    'top_3_positions': 'Avg. Times in Podium',
    'positions_4_10': 'Avg. Times in Positions 4-10',
    'outside_10_positions': 'Avg. Times Without Points'
}

# Creating a grid of plots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, (var, label) in enumerate(plot_vars.items()):
    ax = axes[i]
    sns.lineplot(data=yearly_data, x='year', y=var, ax=ax, marker='o')
    ax.set_title(label, fontsize=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(True)

fig.tight_layout()
plt.show()

# Grouping data by year
yearly_data = working_df.groupby('year').agg({
    'qualifying_position': 'mean',
    'grid_position': custom_mean,
    'positionOrder': 'mean',
    'valid_time_set_in_q1': 'sum',
    'qualified_for_q2': 'sum',
    'qualified_for_q3': 'sum',
    'q1': custom_mean,
    'q2': custom_mean,
    'q3': custom_mean,
}).reset_index()

# Defining variables for plotting
plot_vars = {
    'qualifying_position': 'Avg. Qualifying Position',
    'grid_position': 'Avg. Grid Position',
    'positionOrder': 'Avg. Position Order',
    'valid_time_set_in_q1': 'Total Times with Valid Time in Q1',
    'qualified_for_q2': 'Total Times Qualified for Q2',
    'qualified_for_q3': 'Total Times Qualified for Q3',
    'q1': 'Avg. Fastest Lap in Q1',
    'q2': 'Avg. Fastest Lap in Q2',
    'q3': 'Avg. Fastest Lap in Q3'
}

# Creating a grid of plots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, (var, label) in enumerate(plot_vars.items()):
    ax = axes[i]
    sns.lineplot(data=yearly_data, x='year', y=var, ax=ax, marker='o')
    ax.set_title(label, fontsize=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(True)

fig.tight_layout()
plt.show()

# Grouping data by year
yearly_data = working_df.groupby('year').agg({
    'age': 'mean',
    'years_experience': 'mean',
    'years_with_constructor': 'mean',
    'race_starts': 'mean'
}).reset_index()

# Defining variables for plotting
plot_vars = {
    'age': 'Avg. Age',
    'years_experience': 'Avg. Years of Experience',
    'years_with_constructor': 'Avg. Years with Current Constructor',
    'race_starts': 'Avg. Race Starts'
}

# Creating a grid of plots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, (var, label) in enumerate(plot_vars.items()):
    ax = axes[i]
    sns.lineplot(data=yearly_data, x='year', y=var, ax=ax, marker='o')
    ax.set_title(label, fontsize=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(True)

# Removing empty subplots (we have fewer than 9 plots)
for j in range(len(plot_vars), len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.show()

# Bar Plot for driver_name

plt.figure(figsize=(12, 22))
data = working_df['driver_name'].value_counts()
sns.barplot(x=data.values, y=data.index, palette='viridis')
plt.title('Driver Appearances', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Driver Names', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Variables to create Barplots
barplot_vars = ['driver_nationality', 'constructor_name', 'constructor_nationality','top_constructor','circuit_name']

# Custom Titles for Each Plot
custom_titles = [
    "Nationality Appearances of Drivers",
    "Constructors Appearances",
    "Nationality Appearances of Constructors",
    "Constructor Experience of each Driver per Race (Longest Period of Working Together)",
    "Circuit Appearances"
]

# Bar Plots with Custom Titles
for var, title in zip(barplot_vars, custom_titles):
    plt.figure(figsize=(12, 8))  # Adjust the figure size to fit all categories
    data = working_df[var].value_counts()  # Count all categories
    sns.barplot(x=data.values, y=data.index, palette='viridis')
    plt.title(title, fontsize=16)  # Use the custom title here
    plt.xlabel('Count', fontsize=14)
    plt.ylabel(var, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Define the custom order for the race outcome categories
custom_order = ["Podium finish", "Points without podium", "No points"]

# Count the occurrences of each race outcome category
outcome_counts = working_df['race_outcome'].value_counts()

# Reorder the data according to the custom order
outcome_counts = outcome_counts.reindex(custom_order)

# Plotting the vertical bar plot with reordered categories
plt.figure(figsize=(8, 6))
sns.barplot(x=outcome_counts.index, y=outcome_counts.values, palette='viridis')

# Adding labels and a title
for i, count in enumerate(outcome_counts.values):
    plt.text(i, count + 0.5, str(count), ha='center', fontsize=12)

plt.title("Cumulative Race Outcomes", fontsize=16)
plt.xlabel("Race Outcome", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Driver Appearances per Year Bar Plot
plt.figure(figsize=(8, 6))
data = working_df['year'].value_counts()  # Count all categories
sns.barplot(x=data.index, y=data.values)
plt.title("Driver Appearances per Year", fontsize=16)
plt.xlabel("Years", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Driver Appearances per Round Bar Plot
plt.figure(figsize=(8, 6))
data = working_df['round'].value_counts()  # Count all categories
sns.barplot(x=data.index, y=data.values)
plt.title("Driver Appearances per Round", fontsize=16)
plt.xlabel("Rounds", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Count occurrences of each race_status
status_counts = working_df['race_status'].value_counts()

# Plot the distribution of race_status
plt.figure(figsize=(13, 6))
sns.barplot(x=status_counts.index, y=status_counts.values)
plt.title("Race Status", fontsize=16)
plt.xlabel("Race Status", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Adding counts on the bars
for i, count in enumerate(status_counts.values):
    plt.text(i, count + 5, str(count), ha='center', fontsize=12)
plt.show()

"""Scatter Plots vs Position Order"""

# Identify numeric columns to plot
numeric_columns = working_df.select_dtypes(include=['float64', 'int64']).columns

# Determine the grid size
num_vars = len(numeric_columns)
cols = 4  # Number of columns in the grid
rows = math.ceil(num_vars / cols)

# Create subplots with a shared figure
fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 2), constrained_layout=True)
axes = axes.flatten()  # Flatten for easy indexing

# Loop through numeric columns and create scatterplots
for i, column in enumerate(numeric_columns):
    sns.scatterplot(
        data=working_df,
        x=column,
        y='positionOrder',  # PositionOrder as the dependent variable
        alpha=0.2,
        size=300,
        legend=False,
        ax=axes[i]
    )
    axes[i].set_xlabel(column, fontsize=8)
    axes[i].set_ylabel('Position Order', fontsize=8)

# Turn off unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.show()

# Identify numeric columns to plot
numeric_columns = working_df.select_dtypes(include=['float64', 'int64']).columns

# Exclude specific columns from the numeric columns
columns_to_exclude = ['years_with_constructor', 'year', 'round', 'valid_time_set_in_q1',
                      'q1', 'qualified_for_q2', 'q2', 'qualified_for_q3', 'q3',
                      'positionOrder', 'total_duration_ms']
numeric_columns = [col for col in numeric_columns if col not in columns_to_exclude]

# Determine the grid size
num_vars = len(numeric_columns)
cols = 5  # Number of columns in the grid
rows = math.ceil(num_vars / cols)

# Create subplots with a shared figure
fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 2), constrained_layout=True)
axes = axes.flatten()  # Flatten for easy indexing

# Loop through numeric columns and create scatterplots
for i, column in enumerate(numeric_columns):
    sns.scatterplot(
        data=working_df,
        x=column,
        y='positionOrder',  # PositionOrder as the dependent variable
        alpha=0.2,
        size=300,
        legend=False,
        ax=axes[i]
    )
    axes[i].set_xlabel(column, fontsize=8)
    axes[i].set_ylabel('Position Order', fontsize=8)

# Turn off unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.show()

"""Time Series Plots groupped by Race Outcome"""

# Grouping data by year
yearly_data = working_df.groupby('year').agg({
     'driver_points': 'mean',
    'cumulative_season_driver_points': 'mean',
    'fastest_lap_time': 'mean',
    'laps': 'mean',
    'total_pit_stops': 'mean',
    'total_duration_ms': 'mean',
    'top_3_positions': 'mean',
    'positions_4_10': 'mean',
    'outside_10_positions': 'mean'
}).reset_index()

# Defining variables for plotting
plot_vars = {
    'driver_points': 'Avg. Driver Points',
    'cumulative_season_driver_points': 'Avg. Cumulative Points',
    'fastest_lap_time': 'Avg. Fastest Lap (s)',
    'laps': 'Avg. Laps',
    'total_pit_stops': 'Avg. Total Pit Stops per Race',
    'total_duration_ms': 'Avg. Total Pit Stop Duration per Race (ms)',
    'top_3_positions': 'Avg. Times in Podium',
    'positions_4_10': 'Avg. Times in Positions 4-10',
    'outside_10_positions': 'Avg. Times Without Points'
}

# Creating time series with separation by race_outcome
fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # Create a grid of plots
axes = axes.flatten()

for i, (var, label) in enumerate(plot_vars.items()):
    ax = axes[i]
    sns.lineplot(
        data=working_df,
        x='year',
        y=var,
        hue='race_outcome',  # Separation by the categorical variable
        ax=ax,
        marker='o',
        errorbar='sd'  # Displaying dispersion (standard deviation)
    )
    ax.set_title(label, fontsize=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(True)

fig.tight_layout()
plt.show()

# Grouping data by year
yearly_data = working_df.groupby('year').agg({
    'qualifying_position': 'mean',
    'grid_position': custom_mean,
    'positionOrder': 'mean',
    'valid_time_set_in_q1': 'sum',
    'qualified_for_q2': 'sum',
    'qualified_for_q3': 'sum',
    'q1': custom_mean,
    'q2': custom_mean,
    'q3': custom_mean,
}).reset_index()

# Defining variables for plotting
plot_vars = {
    'qualifying_position': 'Avg. Qualifying Position',
    'grid_position': 'Avg. Grid Position',
    'positionOrder': 'Avg. Position Order',
    'valid_time_set_in_q1': 'Total Times with Valid Time in Q1',
    'qualified_for_q2': 'Total Times Qualified for Q2',
    'qualified_for_q3': 'Total Times Qualified for Q3',
    'q1': 'Avg. Fastest Lap in Q1',
    'q2': 'Avg. Fastest Lap in Q2',
    'q3': 'Avg. Fastest Lap in Q3'
}

# Creating time series with separation by race_outcome
fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # Create a grid of plots
axes = axes.flatten()

for i, (var, label) in enumerate(plot_vars.items()):
    ax = axes[i]
    sns.lineplot(
        data=working_df,
        x='year',
        y=var,
        hue='race_outcome',  # Separation by the categorical variable
        ax=ax,
        marker='o',
        errorbar='sd'  # Displaying dispersion (standard deviation)
    )
    ax.set_title(label, fontsize=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(True)

fig.tight_layout()
plt.show()

# Grouping data by year
yearly_data = working_df.groupby('year').agg({
    'age': 'mean',
    'years_experience': 'mean',
    'years_with_constructor': 'mean',
    'race_starts': 'mean'
}).reset_index()

# Defining variables for plotting
plot_vars = {
    'age': 'Avg. Age',
    'years_experience': 'Avg. Years of Experience',
    'years_with_constructor': 'Avg. Years with Current Constructor',
    'race_starts': 'Avg. Race Starts'
}

# Creating time series with separation by race_outcome
fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # Create a grid of plots
axes = axes.flatten()

for i, (var, label) in enumerate(plot_vars.items()):
    ax = axes[i]
    sns.lineplot(
        data=working_df,
        x='year',
        y=var,
        hue='race_outcome',  # Separation by the categorical variable
        ax=ax,
        marker='o',
        errorbar='sd'  # Displaying dispersion (standard deviation)
    )
    ax.set_title(label, fontsize=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(True)

# Removing unused plots
for j in range(len(plot_vars), len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.show()

"""Violin Plots"""

# Selecting numerical columns from the working_df
numeric_columns = working_df.select_dtypes(include=['float64', 'int64']).columns

# Creating a violin plot for each numerical column
for column in numeric_columns:
    fig = px.violin(
        working_df,
        x='race_outcome',         # Categories
        y=column,                 # The current numerical column
        color='race_outcome',     # Coloring by category
        box=True,                 # Adding boxplot inside the violin plot
        hover_data=working_df.columns,  # Display data on hover
        width=1000,               # Width of the plot
        category_orders={         # Setting the category order in the plot
            'race_outcome': ['Podium finish', 'Points without podium', 'No points']
        }
    )

    # Adjusting title and axis labels
    fig.update_layout(
        title=f'{column} by Race Outcome',
        xaxis_title='Race Outcome',
        yaxis_title=column,
        violinmode='group',       # Sorting by group
        font=dict(size=12)        # Font size
    )

    # Display the plot
    fig.show()

# Pivot the dataset for measurements by category
grouped_data = working_df.groupby(['constructor_nationality', 'race_outcome']).size().reset_index(name='counts')

# Plot with Plotly (Grouped Bar Chart)
fig = px.bar(
    grouped_data,
    x='constructor_nationality',        # X-axis: Constructor Nationality
    y='counts',                         # Y-axis: Count of occurrences
    color='race_outcome',               # Color differentiation by Race Outcome
    barmode='group',                    # Grouped Bar Plot
    title='Comparison of Constructor Nationalities by Race Outcome', # Title of the chart
    labels={'constructor_nationality': 'Constructor Nationality', 'counts': 'Count', 'race_outcome': 'Race Outcome'}, # Axis labels
    width=900,
    height=600
)

# Layout customization
fig.update_layout(
    xaxis=dict(tickangle=45),           # Rotate X-axis labels by 45 degrees
    font=dict(size=12),                  # Set font size for readability
    legend=dict(title='Race Outcome')    # Add a title to the legend
)

# Show the plot
fig.show()

# Pivot the dataset to count occurrences by driver nationality and race outcome
grouped_data = working_df.groupby(['driver_nationality', 'race_outcome']).size().reset_index(name='counts')

# Create a grouped bar chart using Plotly
fig = px.bar(
    grouped_data,
    x='driver_nationality',        # X-axis: Driver Nationality
    y='counts',                     # Y-axis: Count of occurrences
    color='race_outcome',           # Color the bars by race outcome
    barmode='group',                # Group the bars by nationality and outcome
    title='Comparison of Driver Nationality by Race Outcome',  # Chart title
    labels={'driver_nationality': 'Driver Nationality', 'counts': 'Count', 'race_outcome': 'Race Outcome'},  # Axis labels
    width=900,                      # Set the width of the plot
    height=600                      # Set the height of the plot
)

# Customize the layout of the plot
fig.update_layout(
    xaxis=dict(tickangle=45),         # Rotate the X-axis labels by 45 degrees for better readability
    font=dict(size=12),                # Set the font size for labels and title
    legend=dict(title='Race Outcome')  # Set the title for the legend
)

# Display the plot
fig.show()

# Group the data by constructor and race outcome and count the occurrences
grouped_data = working_df.groupby(['constructor_name', 'race_outcome']).size().reset_index(name='counts')

# Create a grouped bar chart using Plotly
fig = px.bar(
    grouped_data,
    x='constructor_name',        # X-axis: Constructor Names
    y='counts',                  # Y-axis: Count of occurrences
    color='race_outcome',        # Color the bars by race outcome
    barmode='group',             # Group the bars by constructor and outcome
    title='Comparison of Constructor by Race Outcome',  # Chart title
    labels={'constructor_name': 'Constructor', 'counts': 'Count', 'race_outcome': 'Race Outcome'},  # Axis labels
    width=900,                   # Set the width of the plot
    height=600                   # Set the height of the plot
)

# Customize the layout of the plot
fig.update_layout(
    xaxis=dict(tickangle=45),         # Rotate the X-axis labels by 45 degrees for better readability
    font=dict(size=12),                # Set the font size for labels and title
    legend=dict(title='Race Outcome')  # Set the title for the legend
)

# Display the plot
fig.show()

# Pivot the dataset to get counts by top constructor and race outcome
grouped_data = working_df.groupby(['top_constructor', 'race_outcome']).size().reset_index(name='counts')

# Create a grouped bar chart using Plotly
fig = px.bar(
    grouped_data,
    x='top_constructor',               # X-axis: Constructor Names
    y='counts',                         # Y-axis: Count of occurrences
    color='race_outcome',               # Color the bars by race outcome
    barmode='group',                    # Group the bars by constructor and outcome
    title='Comparison of Constructor with most Experience of each Driver per Race by Race Outcome',  # Chart title
    labels={'top_constructor': 'Constructor', 'counts': 'Count', 'race_outcome': 'Race Outcome'},  # Axis labels
    width=900,                           # Set the width of the plot
    height=600                           # Set the height of the plot
)

# Customize the layout of the plot
fig.update_layout(
    xaxis=dict(tickangle=45),           # Rotate the X-axis labels by 45 degrees for better readability
    font=dict(size=12),                  # Set the font size for labels and title
    legend=dict(title='Race Outcome')    # Set the title for the legend
)

# Display the plot
fig.show()

# Pivot the dataset to get counts by circuit and race outcome
grouped_data = working_df.groupby(['circuit_name', 'race_outcome']).size().reset_index(name='counts')

# Create a grouped bar chart using Plotly
fig = px.bar(
    grouped_data,
    x='circuit_name',               # X-axis: Circuit names
    y='counts',                     # Y-axis: Count of occurrences
    color='race_outcome',           # Color the bars by race outcome
    barmode='group',                # Group the bars by circuit and outcome
    title='Comparison of Circuit by Race Outcome',  # Chart title
    labels={'circuit_name': 'Circuit', 'counts': 'Count', 'race_outcome': 'Race Outcome'},  # Axis labels
    width=900,                      # Set the width of the plot
    height=600                      # Set the height of the plot
)

# Customize the layout of the plot
fig.update_layout(
    xaxis=dict(tickangle=45),       # Rotate the X-axis labels by 45 degrees for better readability
    font=dict(size=12),              # Set the font size for labels and title
    legend=dict(title='Race Outcome')  # Set the title for the legend
)

# Display the plot
fig.show()

# Pivot the dataset to get counts by race status and race outcome
grouped_data = working_df.groupby(['race_status', 'race_outcome']).size().reset_index(name='counts')

# Create a grouped bar chart using Plotly
fig = px.bar(
    grouped_data,
    x='race_status',               # X-axis: Race status categories
    y='counts',                     # Y-axis: Count of occurrences
    color='race_outcome',           # Color the bars by race outcome
    barmode='group',                # Group the bars by race status and outcome
    title='Comparison of Race Status by Race Outcome',  # Chart title
    labels={'race_status': 'Race Status', 'counts': 'Count', 'race_outcome': 'Race Outcome'},  # Axis labels
    width=900,                      # Set the width of the plot
    height=600                      # Set the height of the plot
)

# Customize the layout of the plot
fig.update_layout(
    xaxis=dict(tickangle=45),       # Rotate the X-axis labels by 45 degrees for better readability
    font=dict(size=12),              # Set the font size for labels and title
    legend=dict(title='Race Outcome')  # Set the title for the legend
)

# Display the plot
fig.show()

# Pivot the dataset to get counts by driver name and race outcome
grouped_data = working_df.groupby(['driver_name', 'race_outcome']).size().reset_index(name='counts')

# Create a grouped bar chart using Plotly
fig = px.bar(
    grouped_data,
    x='driver_name',               # X-axis: Driver names
    y='counts',                     # Y-axis: Count of occurrences
    color='race_outcome',           # Color the bars by race outcome
    barmode='group',                # Group the bars by race outcome
    title='Comparison of Driver by Race Outcome',  # Chart title
    labels={'driver_name': 'Driver', 'counts': 'Count', 'race_outcome': 'Race Outcome'},  # Axis labels
    width=1300,                     # Set the width of the plot
    height=600                      # Set the height of the plot
)

# Customize the layout of the plot
fig.update_layout(
    xaxis=dict(tickangle=45),       # Rotate the X-axis labels by 45 degrees for better readability
    font=dict(size=10),              # Set the font size for labels and title
    legend=dict(title='Race Outcome')  # Set the title for the legend
)

# Display the plot
fig.show()

# List of specific drivers to include
selected_drivers = ['Lewis Hamilton', 'Sebastian Vettel', 'Max Verstappen', 'Sergio Perez', 'Carlos Sainz', 'Fernando Alonso']

# Filter the dataset to include only the selected drivers
filtered_data = grouped_data[grouped_data['driver_name'].isin(selected_drivers)]

# Create a grouped bar chart using Plotly for the filtered dataset
fig = px.bar(
    filtered_data,
    x='driver_name',               # X-axis: Driver names
    y='counts',                     # Y-axis: Count of occurrences
    color='race_outcome',           # Color the bars by race outcome
    barmode='group',                # Group the bars by race outcome
    title='Comparison of Selected Drivers by Race Outcome',  # Chart title
    labels={'driver_name': 'Driver', 'counts': 'Count', 'race_outcome': 'Race Outcome'},  # Axis labels
    width=800,                     # Set the width of the plot
    height=600                      # Set the height of the plot
)

# Customize the layout of the plot
fig.update_layout(
    font=dict(size=12),              # Set the font size for labels and title
    legend=dict(title='Race Outcome')  # Set the title for the legend
)

# Display the plot
fig.show()

"""Fastest Laps by Circuits"""

import matplotlib.pyplot as plt
import seaborn as sns

# Group by 'circuit_name' and 'year' to calculate the mean fastest lap time
circuit_data = working_df.groupby(['circuit_name', 'year']).agg({
    'fastest_lap_time': 'mean'
}).reset_index()

# Get a list of unique circuits
circuits = circuit_data['circuit_name'].unique()

# Define figure size based on the number of circuits
fig, axes = plt.subplots(len(circuits), 1, figsize=(12, 4 * len(circuits)), sharex=True)

# If there's only one circuit, make axes a list for consistency
if len(circuits) == 1:
    axes = [axes]

# Plot fastest lap times for each circuit
for ax, circuit in zip(axes, circuits):
    data = circuit_data[circuit_data['circuit_name'] == circuit]
    sns.lineplot(data=data, x='year', y='fastest_lap_time', ax=ax, marker='o')
    ax.set_title(f"Avg. Fastest Lap Times - {circuit}", fontsize=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Avg. Fastest Lap (s)', fontsize=10)
    ax.grid(True)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# List of selected circuits
selected_circuits = [
    "Albert Park Grand Prix Circuit",
    "Autodromo Nazionale di Monza",
    "Bahrain International Circuit",
    "Circuit de Barcelona-Catalunya",
    "Circuit de Monaco",
    "Circuit de Spa-Francorchamps",
    "Circuit of the Americas",
    "Hungaroring",
    "Marina Bay Street Circuit",
    "Silverstone Circuit"
]

# Filter data for selected circuits and years up to 2023
filtered_data = working_df[
    (working_df['circuit_name'].isin(selected_circuits)) &
    (working_df['year'] <= 2023)
]

# Group by 'circuit_name' and 'year' to calculate the mean fastest lap time
circuit_data = filtered_data.groupby(['circuit_name', 'year']).agg({
    'fastest_lap_time': 'mean'
}).reset_index()

# Get the number of selected circuits
num_circuits = len(selected_circuits)

# Set up a grid with 2 columns
fig, axes = plt.subplots((num_circuits + 1) // 2, 2, figsize=(14, 3 * (num_circuits // 2)), sharex=True)

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot fastest lap times for each selected circuit
for ax, circuit in zip(axes, selected_circuits):
    data = circuit_data[circuit_data['circuit_name'] == circuit]
    sns.lineplot(data=data, x='year', y='fastest_lap_time', ax=ax, marker='o')
    ax.set_title(f"{circuit}", fontsize=12)
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Avg. Fastest Lap (s)', fontsize=9)
    ax.grid(True)

# Remove any empty subplots (if the number of circuits is odd)
for i in range(len(selected_circuits), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

"""Chapter 4: Hypothesis Testing and Correlation"""

# Read the created working excel
working_df = pd.read_excel(WORKING_CHECKPOINT)

"""Kolmogorov-Smirnov Normality Test"""

# List of numeric variables
numeric_variables = working_df.select_dtypes(include=['float64', 'int64']).columns

# Prepare the plot layout for 6x4 grid of histograms and QQ plots (in separate axes)
fig, axes = plt.subplots(6, 4, figsize=(18, 24))
fig_qq, axes_qq = plt.subplots(6, 4, figsize=(18, 24))

# Flatten axes for easy indexing
axes = axes.flatten()
axes_qq = axes_qq.flatten()

# Initialize a list to store KS statistics and p-values
ks_stats = []
ks_pvalues = []

# Loop through each numeric variable to generate plots and calculate KS test
for idx, var in enumerate(numeric_variables):
    data = working_df[var]

    # Kolmogorov-Smirnov test (for normality)
    stat_ks, p_value_ks = stats.kstest(data, 'norm')

    # Store the KS statistic and p-value for later use
    ks_stats.append(stat_ks)
    ks_pvalues.append(p_value_ks)

    # Histogram
    sns.histplot(data, kde=True, color='skyblue', ax=axes[idx])

    # Q-Q Plot
    stats.probplot(data, dist="norm", plot=axes_qq[idx])
    axes_qq[idx].set_title(f'Q-Q Plot of {var}')

    # Tighten layout for histograms and QQ plots
    plt.tight_layout()

# Create a board to display the KS statistics and p-values with more decimals
fig_ks, ax_ks = plt.subplots(figsize=(12, 6))
ax_ks.axis('tight')
ax_ks.axis('off')

# Create a DataFrame to display KS statistic and p-value
ks_values_df = pd.DataFrame({
    'Variable': numeric_variables,
    'KS Statistic': [f'{stat:.3f}' for stat in ks_stats],
    'KS p-value': [f'{p_val:.3f}' for p_val in ks_pvalues]
})

# Display the KS statistic and p-value table
ax_ks.table(cellText=ks_values_df.values, colLabels=ks_values_df.columns, loc='center', cellLoc='center')

plt.show()

#Checking Qualy times without zeros
variables_to_check = ['q1', 'q2', 'q3']

# Create figure layouts for histograms and Q-Q plots
fig, axes = plt.subplots(1,len(variables_to_check), figsize=(16, 4))
fig_qq, axes_qq = plt.subplots(1,len(variables_to_check), figsize=(16,4))

# Ensure axes are iterable even if there is one variable
if len(variables_to_check) == 1:
    axes = [axes]
    axes_qq = [axes_qq]
else:
    axes = axes.flatten()
    axes_qq = axes_qq.flatten()

# Initialize lists for KS statistics and p-values
ks_stats = []
ks_pvalues = []

# Loop through each variable separately
for idx, var in enumerate(variables_to_check):
    data = working_df[var]

    # Remove only the zero values for this specific variable
    data_no_zeros = data[data != 0]

    # Kolmogorov-Smirnov test (for normality)
    stat_ks, p_value_ks = stats.kstest(data_no_zeros, 'norm')

    # Store KS statistic and p-value
    ks_stats.append(stat_ks)
    ks_pvalues.append(p_value_ks)

    # Histogram
    sns.histplot(data_no_zeros, kde=True, color='skyblue', ax=axes[idx])
    axes[idx].set_title(f'Histogram of {var} (Zeros Removed)')

    # Q-Q Plot
    stats.probplot(data_no_zeros, dist="norm", plot=axes_qq[idx])
    axes_qq[idx].set_title(f'Q-Q Plot of {var} (Zeros Removed)')

# Adjust layout for better visualization
plt.tight_layout()

# Display KS statistics in a table
fig_ks, ax_ks = plt.subplots(figsize=(8, 2))
ax_ks.axis('tight')
ax_ks.axis('off')

# Create DataFrame to display KS statistic and p-value
ks_values_df = pd.DataFrame({
    'Variable': variables_to_check,
    'KS Statistic': [f'{stat:.3f}' for stat in ks_stats],
    'KS p-value': [f'{p_val:.3f}' for p_val in ks_pvalues]
})

# Display KS statistic and p-value table
ax_ks.table(cellText=ks_values_df.values, colLabels=ks_values_df.columns, loc='center', cellLoc='center')

plt.show()

"""Kolmogorov-Smirnov Normality Test Groupped"""

# List of numeric variables
numeric_variables = ['age', 'years_experience', 'race_starts', 'years_with_constructor',
                     'q1', 'q2', 'q3', 'qualifying_position', 'grid_position',
                     'positionOrder', 'driver_points', 'cumulative_season_driver_points',
                     'fastest_lap_time', 'laps', 'total_pit_stops', 'total_duration_ms',
                     'top_3_positions', 'positions_4_10', 'outside_10_positions']

# Define the Race Outcome categories
race_outcomes = ['Podium finish', 'Points without podium', 'No points']

# Loop through each Race Outcome category
for outcome in race_outcomes:
    # Filter the data for the specific Race Outcome
    subgroup_data = working_df[working_df['race_outcome'] == outcome]

    # Initialize lists to store K-S statistics and p-values
    ks_stats = []
    ks_pvalues = []

    # Perform K-S test for each numeric variable
    for var in numeric_variables:
        data = subgroup_data[var]

        # Kolmogorov-Smirnov test (for normality)
        stat_ks, p_value_ks = stats.kstest(data, 'norm')

        # Append results
        ks_stats.append(stat_ks)
        ks_pvalues.append(p_value_ks)

    # Print the results in a text format
    print(f"K-S Test Results for Race Outcome: {outcome}")
    print(f"{'Variable':<30}{'KS Statistic':<15}{'KS p-value':<15}")
    print("-" * 60)
    for i, var in enumerate(numeric_variables):
        print(f"{var:<30}{ks_stats[i]:<15.3f}{ks_pvalues[i]:<15.3f}")
    print("\n")

"""Correlations"""

# Selecting only the numeric variables
numeric_variables = ['age', 'years_experience', 'race_starts', 'years_with_constructor',
                     'q1', 'q2', 'q3', 'qualifying_position', 'grid_position',
                     'driver_points', 'cumulative_season_driver_points',
                     'fastest_lap_time', 'laps', 'total_pit_stops', 'total_duration_ms',
                     'top_3_positions', 'positions_4_10', 'outside_10_positions']

# Subset the DataFrame to only include the numeric variables
numeric_df = working_df[numeric_variables]

# Calculate the correlation matrix using Spearman correlation
correlation_matrix = numeric_df.corr(method='spearman')

# Plotting the heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Spearman Correlation Heatmap of Numeric Variables")
plt.show()

"""Correlations Grouped"""

# Selecting only the numeric variables
numeric_variables = ['age', 'years_experience', 'race_starts', 'years_with_constructor',
                     'q1', 'q2', 'q3', 'qualifying_position', 'grid_position',
                     'cumulative_season_driver_points',
                     'fastest_lap_time', 'laps', 'total_pit_stops', 'total_duration_ms',
                     'top_3_positions', 'positions_4_10', 'outside_10_positions', 'driver_points']

# Add the Race Outcome variable to the DataFrame
working_df = working_df.copy()
working_df['race_outcome'] = working_df['race_outcome']

# Subset the DataFrame to only include the numeric variables
numeric_df = working_df[numeric_variables]

# Iterate over each unique category in the Race Outcome variable
race_outcome_categories = working_df['race_outcome'].unique()

for outcome in race_outcome_categories:
    # Filter the DataFrame for the current category
    subset_df = numeric_df[working_df['race_outcome'] == outcome]

    # Calculate the Spearman correlation matrix for this subset
    correlation_matrix = subset_df.corr(method='spearman')

    # Plotting the heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title(f"Spearman Correlation Heatmap for Race Outcome: {outcome}")
    plt.show()

"""Chapter 5: Regression"""

# Read the created working excel
working_df = pd.read_excel(WORKING_CHECKPOINT)

# Season points accumulated before the current race
working_df['cumulative_season_driver_points'] = (
    working_df['cumulative_season_driver_points']
    - working_df['driver_points']
)

# Load Data
data = working_df
data.head()

"""Grouping Circuits and Nationalities"""

print(data['circuit_name'].unique())

circuit_df = pd.read_excel(DATA_DIR / "circuits_overtakes.xlsx")
circuit_df.head()

circuit_df["Avg_Overtakes"] = circuit_df["Avg_Overtakes"].fillna(circuit_df["Avg_Overtakes"].mean())
circuit_df.head()

# Define bins and labels
bins = [0, 24.99, 49.99, 74.99, 100]  # 4 groups
labels = [1, 2, 3, 4]

# Create a new column for grouping
circuit_df["Overtake_Group"] = pd.cut(circuit_df["Avg_Overtakes"], bins=bins, labels=labels, include_lowest=True)

circuit_df.head()

# Save to an Excel file
circuit_df.to_excel(OUTPUT_DIR / "circuits_overtakesV5.xlsx", index=False)

print(data['driver_nationality'].unique())

nationalities_df = pd.read_excel(DATA_DIR / "Nationalities.xlsx")
nationalities_df.head()

"""Encoding"""

# List with categorical values for Encoding
categorical_columns = ['driver_name', 'constructor_name','constructor_nationality',
                       'top_constructor', 'race_status'
]

# Applying LabelEncoder in each column
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

data.head()

# Define the label mapping from your custom labels to numeric values
outcome_labels = {"Podium finish": 0, "Points without podium": 1, "No points": 2}

# Apply the mapping to the 'race_outcome' column
data['race_outcome'] = data['race_outcome'].map(outcome_labels)

# Check the result
print(data['race_outcome'])

data['circuit_name'] = data['circuit_name'].map(circuit_df.set_index('Circuit Name')['Overtake_Group'])
data['circuit_name'].head()

data['driver_nationality'] = data['driver_nationality'].map(nationalities_df.set_index('Driver Nationalities')['European'])
data['driver_nationality'].head()

"""Early Testing"""

# # Select Variables for x
x = data.iloc[:,0:31]

# Select Outcome as target
y = data.iloc[:,31]

# Create a 75% random split of data for training/testing
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1234)

# Initialize Logistic Regression model
log_reg_model = LogisticRegression(class_weight='balanced', solver='lbfgs', random_state=1234, max_iter=100)

# Fit the model
log_reg_model.fit(x_train, y_train)

# Predict on the test set
y_pred = log_reg_model.predict(x_test)

# Evaluate model performance
accuracy_train = accuracy_score(y_train, log_reg_model.predict(x_train))
accuracy_test = accuracy_score(y_test, y_pred)

# Classification Report
class_report = classification_report(y_test, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate balanced accuracy
balanced_acc = balanced_accuracy_score(y_test, log_reg_model.predict(x_test))

# Get probabilities for each class
y_proba = log_reg_model.predict_proba(x_test)
# Compute ROC-AUC (one-vs-rest)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

# Print results
print("Train Accuracy for Logistic Regression: {:.4f}".format(accuracy_train))
print("Test Accuracy for Logistic Regression: {:.4f}".format(accuracy_test))
print("\nClassification Report:")
print(class_report)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nBalanced Accuracy: {:.4f}".format(balanced_acc))
print("ROC-AUC Score: {:.4f}".format(roc_auc))

"""VIF"""

# Calculating VIF
vif_data = pd.DataFrame()
vif_data["feature"] = data.columns[:31]
vif_data["VIF"] = [variance_inflation_factor(x, i) for i in range(x.shape[1])]

print(vif_data)

# Select Variables for x
x = data.iloc[:,[5,6,7,18,22,23,26,27,28,30]]

# Calculating VIF
selected_columns = [5,6,7,18,22,23,26,27,28,30]
vif_data = pd.DataFrame()
vif_data["feature"] = data.columns[selected_columns]
vif_data["VIF"] = [variance_inflation_factor(x, i) for i in range(x.shape[1])]

print(vif_data)

"""Correlations as is"""

# Selecting the variables
variables = ['constructor_name', 'constructor_nationality','years_with_constructor', 'qualifying_position',
                     'driver_points','cumulative_season_driver_points','total_pit_stops',
                     'total_duration_ms', 'top_3_positions','outside_10_positions']

# Subset the DataFrame to only include the numeric variables
numeric_df = working_df[variables]

# Calculate the correlation matrix using Spearman correlation
correlation_matrix = numeric_df.corr(method='spearman')

# Plotting the heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Spearman Correlation Heatmap of Numeric Variables")
plt.show()

#Variables after Correlation investigating
x = data.iloc[:,[5,6,7,18,23,26,30]]

#Final Variables after Wald checks AIC and BIC compairing etc
x = data.iloc[:,[7,18,23,26]]

# Calculating VIF
selected_columns = [7,18,23,26]
vif_data = pd.DataFrame()
vif_data["feature"] = data.columns[selected_columns]
vif_data["VIF"] = [variance_inflation_factor(x, i) for i in range(x.shape[1])]

print(vif_data)

"""Multinomial Final"""

# Split data into training and testing sets (75% training, 25% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1234)

# Fit MNLogit model using statsmodels
x_train_with_const = sm.add_constant(x_train)
x_test_with_const = sm.add_constant(x_test)

mnlogit_model = sm.MNLogit(y_train, x_train_with_const)
mnlogit_result = mnlogit_model.fit(method='newton')

# Predict on test set
y_pred_proba = mnlogit_result.predict(x_test_with_const)  # Predicted probabilities
y_pred = y_pred_proba.idxmax(axis=1)  # Convert probabilities to class predictions

# Metrics Calculation
accuracy_train = accuracy_score(y_train, mnlogit_result.predict(x_train_with_const).idxmax(axis=1))
accuracy_test = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# ROC-AUC Calculation (One-vs-Rest)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

# Get AIC and BIC from the fitted model
aic = mnlogit_result.aic
bic = mnlogit_result.bic

# Print AIC and BIC scores
print("AIC: {:.4f}".format(aic))
print("BIC: {:.4f}".format(bic))
# Print model summary
print(mnlogit_result.summary())

# Print metrics
print(f"Train Accuracy: {accuracy_train:.4f}")
print(f"Test Accuracy: {accuracy_test:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(class_report)


# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ROC Curve for each class
y_pred_proba = pd.DataFrame(mnlogit_result.predict(sm.add_constant(x_test)))
# Loop through each class and compute ROC curve
plt.figure(figsize=(8, 6))
for i in range(y_pred_proba.shape[1]):  # Iterate through available classes
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba.iloc[:, i], pos_label=i)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
# Add diagonal line and labels
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.title("ROC Curve for Multinomial Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="best")
plt.grid()
plt.show()

"""Learning Curve"""

from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

# Define the model
log_reg_model = LogisticRegression()

# Generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    log_reg_model, x, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculate mean and standard deviation of train and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(6, 4))
plt.plot(train_sizes, train_scores_mean, label="Training Score", color="blue")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="blue", alpha=0.2)
plt.plot(train_sizes, test_scores_mean, label="Validation Score", color="green")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color="green", alpha=0.2)
plt.title("Learning Curve (Logistic Regression)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()

"""Residuals"""

# Extract predicted probabilities for the actual classes
y_test_numeric = y_test.astype(int).to_numpy()  # Convert to numpy array for indexing
# y_pred_proba_actual = y_pred_proba[np.arange(len(y_test_numeric)), y_test_numeric]
y_pred_proba_actual = y_pred_proba.values[np.arange(len(y_test_numeric)), y_test_numeric] # Access the underlying NumPy array

# Compute Pearson Residuals
pearson_residuals = (y_test_numeric - y_pred_proba_actual) / np.sqrt(y_pred_proba_actual * (1 - y_pred_proba_actual))

# --- Q-Q Plot (Check Normality) ---
sm.qqplot(pearson_residuals, line='s')
plt.title("Q-Q Plot of Pearson Residuals")
plt.show()

# --- Scatter Plot (Check Independence) ---
plt.figure(figsize=(5, 3))
plt.scatter(np.arange(len(pearson_residuals)), pearson_residuals, alpha=0.5, color="blue")
plt.xlabel("Sample Index")
plt.ylabel("Pearson Residuals")
plt.title("Residuals Scatter Plot")
plt.legend()
plt.show()

"""LassoCV"""

#LassoCV auto selecting features
x = data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30]]
def select_best_variables_lasso(x, y, penalty_strengths=np.logspace(-4, 1, 30), cv=5):

    # Standardize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Lasso-based logistic regression with cross-validation
    lasso = LogisticRegressionCV(
       Cs=penalty_strengths,
       cv=5,
       penalty="l1",
       solver="saga",  # Faster than liblinear
       multi_class="multinomial",
       max_iter=1000,
       n_jobs=-1  # Use all available CPU cores

    )

    # Fit model
    lasso.fit(x_scaled, y)

    # Identify non-zero coefficients (selected variables)
    selected_mask = np.any(lasso.coef_ != 0, axis=0)  # Keeps features with non-zero coefficients
    selected_features = x.columns[selected_mask]
    removed_features = x.columns[~selected_mask]  # Features removed by Lasso

    print("\nSelected Features for MNL Model:")
    print(selected_features.tolist())

    print("\nRemoved Features (Lasso Shrunk to Zero):")
    print(removed_features.tolist())

    # Return reduced dataset with selected features
    return x[selected_features], removed_features.tolist()

# Example Usage:
x_selected_lasso, removed_features_lasso = select_best_variables_lasso(x, y)

"""Chapter 6: Timeseries"""

# Load necessary datasets
results = pd.read_csv(DATA_DIR / "results.csv")  # Contains fastest lap times
races = pd.read_csv(DATA_DIR / "races.csv")  # Contains race locations & years
circuits = pd.read_csv(DATA_DIR / "circuits.csv")  # Contains circuit details

# Merge races with circuits
races = races.merge(circuits, on="circuitId")

# Merge results with races to get circuit information
df = results.merge(races, on="raceId")

#AICc Calculation
def calculate_aicc(aic, k, n):
    """Calculate AICc (Corrected AIC) for small sample sizes."""
    return aic + (2 * k * (k + 1)) / (n - k - 1)

"""ALBERT PARK"""

# Keep only Albert Park races
albert_park_races = df[df["circuitRef"] == "albert_park"]

# Convert fastest lap time to seconds
def time_to_seconds(time_str):
    try:
        m, s = map(float, time_str.split(":"))
        return m * 60 + s
    except:
        return None  # Handle missing or incorrect values

albert_park_races["fastestLapTime_seconds"] = albert_park_races["fastestLapTime"].apply(time_to_seconds)

# Compute average fastest lap time per race (year)
tsdf = albert_park_races.groupby("year")["fastestLapTime_seconds"].mean().reset_index()
tsdf = tsdf[tsdf["year"] >= 2004].copy()

# TS plot, ACF, PACF
fig, axes = plt.subplots(1, 3, figsize=(25, 8))
tsdf["fastestLapTime_seconds"].plot(title="Albert Park Avg Fastest Lap Times Over Time", ax=axes[0])
plot_acf(tsdf["fastestLapTime_seconds"], ax=axes[1])
plot_pacf(tsdf["fastestLapTime_seconds"], ax=axes[2])
plt.show()

# ADF Test
result = adfuller(tsdf["fastestLapTime_seconds"])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

if result[1] < 0.05:
    print("Time series is stationary.")
else:
    print("Time series is NOT stationary.")

# 1st Diff
tsdf["lap_time_diff"] = tsdf["fastestLapTime_seconds"].diff()

# Drop the first NaN row
tsdf.dropna(inplace=True)

# ADF Test
result = adfuller(tsdf["lap_time_diff"])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

if result[1] < 0.05:
    print("Time series is now stationary.")
else:
    print("Still not stationary.")

# 2nd Diff
tsdf["lap_time_diff2"] = tsdf["lap_time_diff"].diff()
tsdf.dropna(inplace=True)

# ADF Test Again
result = adfuller(tsdf["lap_time_diff2"])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# TS plot, ACF, PACF
fig, axes = plt.subplots(1, 3, figsize=(25, 8))
tsdf["lap_time_diff2"].plot(title="Albert Park 2nd Diff Over Time", ax=axes[0])
plot_acf(tsdf["lap_time_diff2"], ax=axes[1])
plot_pacf(tsdf["lap_time_diff2"], ax=axes[2])
plt.show()

# Define ARIMA models to compare
models = [(1,0,1), (2,0,1), (1,0,2), (2,0,2)]
results = []

for order in models:
    model = sm.tsa.ARIMA(tsdf["lap_time_diff2"], order=order)
    fit = model.fit()

    # Number of parameters (p + d + q + 1 for variance)
    k = sum(order) + 1
    n = len(tsdf["lap_time_diff2"])

    results.append({
        "Model": f"ARIMA{order}",
        "AIC": fit.aic,
        "BIC": fit.bic,
        "AICc": calculate_aicc(fit.aic, k, n)
    })

# Convert results to a DataFrame and display
Results = pd.DataFrame(results)
print(Results)

# Residual Testing
fit = sm.tsa.ARIMA(tsdf["lap_time_diff2"], order=(1, 0, 1)).fit()
residuals = fit.resid

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Q-Q plot
sm.qqplot(residuals, line="s", ax=axes[0])
axes[0].set_title("Q-Q Plot of Residuals")
# Residuals over time
axes[1].plot(residuals)
axes[1].set_title("Residuals Over Time")

# Autocorrelation plot
plot_acf(residuals, ax=axes[2])
axes[2].set_title("Residuals ACF plot")
plt.show()

# Testing
# Residual Shapiro
stat, p_shapiro = shapiro(fit.resid)  # Residuals from Auto-ARIMA
print(f"Shapiro-Wilk p-value: {p_shapiro:.4f}")

# Res Anderson-Darling
result = anderson(fit.resid, dist='norm')
print(f"Anderson-Darling Test Statistic: {result.statistic:.4f}")
print(f"Critical Values: {result.critical_values}")
print(f"Significance Levels: {result.significance_level}")

# Box-Ljung Test
ljung_p_values = acorr_ljungbox(fit.resid, lags=[10], return_df=True)['lb_pvalue']
print(f"Box-Ljung p-value: {ljung_p_values.iloc[-1]:.4f}")

# Forecasting for the next 2 years
forecast = fit.forecast(steps=2)
conf_int = fit.get_forecast(steps=2).conf_int()

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(tsdf["year"], tsdf["lap_time_diff2"], marker="o", label="Actual Data")
plt.plot(range(tsdf["year"].max() + 1, tsdf["year"].max() + 3), forecast, marker="o", color="red", label="Forecast")
plt.fill_between(range(tsdf["year"].max() + 1, tsdf["year"].max() + 3), conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.8)
plt.xlabel("Year")
plt.ylabel("Differenced Lap Time")
plt.title("ARIMA (1,0,1) Forecast for Albert Park")
plt.legend()
plt.show()

"""MONACO"""

# Keep only Monaco races
monaco_races = df[df["circuitRef"] == "monaco"]

# Convert fastest lap time to seconds
def time_to_seconds(time_str):
    try:
        m, s = map(float, time_str.split(":"))
        return m * 60 + s
    except:
        return None  # Handle missing or incorrect values

monaco_races["fastestLapTime_seconds"] = monaco_races["fastestLapTime"].apply(time_to_seconds)

# Compute average fastest lap time per race (year)
tsdf = monaco_races.groupby("year")["fastestLapTime_seconds"].mean().reset_index()
tsdf = tsdf[(tsdf["year"] >= 2004) & (tsdf["year"] <= 2023)].copy()

# TS plot, ACF, PACF
fig, axes = plt.subplots(1, 3, figsize=(25, 8))
tsdf["fastestLapTime_seconds"].plot(title="Monaco Avg Fastest Lap Times Over Time", ax=axes[0])
plot_acf(tsdf["fastestLapTime_seconds"], ax=axes[1])
plot_pacf(tsdf["fastestLapTime_seconds"], ax=axes[2])
plt.show()

# ADF Test
result = adfuller(tsdf["fastestLapTime_seconds"])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

if result[1] < 0.05:
    print("Time series is stationary.")
else:
    print("Time series is NOT stationary.")

# 1st Diff
tsdf["lap_time_diff"] = tsdf["fastestLapTime_seconds"].diff()

# Drop the first NaN row
tsdf.dropna(inplace=True)

# ADF Test
result = adfuller(tsdf["lap_time_diff"])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

if result[1] < 0.05:
    print("Time series is stationary.")
else:
    print("Time series is NOT stationary.")

# TS plot, ACF, PACF
fig, axes = plt.subplots(1, 3, figsize=(25, 8))
tsdf["lap_time_diff"].plot(title="Monaco 1st Diff Over Time", ax=axes[0])
plot_acf(tsdf["lap_time_diff"], ax=axes[1])
plot_pacf(tsdf["lap_time_diff"], ax=axes[2])
plt.show()

# Define ARIMA models to compare
models = [(1,0,0), (0,0,1), (1,0,1)]
results = []

for order in models:
    model = sm.tsa.ARIMA(tsdf["lap_time_diff"], order=order)
    fit = model.fit()

    # Number of parameters (p + d + q + 1 for variance)
    k = sum(order) + 1
    n = len(tsdf["lap_time_diff"])

    results.append({
        "Model": f"ARIMA{order}",
        "AIC": fit.aic,
        "BIC": fit.bic,
        "AICc": calculate_aicc(fit.aic, k, n)
    })

# Convert results to a DataFrame and display
Results = pd.DataFrame(results)
print(Results)

# Residual Testing
fit = sm.tsa.ARIMA(tsdf["lap_time_diff"], order=(1, 0, 0)).fit()
residuals = fit.resid

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Q-Q plot
sm.qqplot(residuals, line="s", ax=axes[0])
axes[0].set_title("Q-Q Plot of Residuals")
# Residuals over time
axes[1].plot(residuals)
axes[1].set_title("Residuals Over Time")

# Autocorrelation plot
plot_acf(residuals, ax=axes[2])
axes[2].set_title("Residuals ACF plot")
plt.show()

# Residual Testing
# Residual Shapiro
stat, p_shapiro = shapiro(fit.resid)  # Residuals from Auto-ARIMA
print(f"Shapiro-Wilk p-value: {p_shapiro:.4f}")

# Res Anderson-Darling
result = anderson(fit.resid, dist='norm')
print(f"Anderson-Darling Test Statistic: {result.statistic:.4f}")
print(f"Critical Values: {result.critical_values}")
print(f"Significance Levels: {result.significance_level}")

# Box-Ljung Test
ljung_p_values = acorr_ljungbox(fit.resid, lags=[10], return_df=True)['lb_pvalue']
print(f"Box-Ljung p-value: {ljung_p_values.iloc[-1]:.4f}")

# Forecasting for the next 2 years
forecast = fit.forecast(steps=2)
conf_int = fit.get_forecast(steps=2).conf_int()

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(tsdf["year"], tsdf["lap_time_diff"], marker="o", label="Actual Data")
plt.plot(range(tsdf["year"].max() + 1, tsdf["year"].max() + 3), forecast, marker="o", color="red", label="Forecast")
plt.fill_between(range(tsdf["year"].max() + 1, tsdf["year"].max() + 3), conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.8)
plt.xlabel("Year")
plt.ylabel("Differenced Lap Time")
plt.title("ARIMA (1,0,0) Forecast for Monaco")
plt.legend()
plt.show()

"""Chapter 7: Machine Learning

Scaling
"""

data.head()

# Scaling for PCA
# Separate the features (X) from the target (y)
X = data.drop(columns=['driver_name','positionOrder','race_outcome'])
y = data['race_outcome']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the X variables to scale them
X_scaled = scaler.fit_transform(X)

# Convert the scaled data back to a DataFrame (optional, for readability)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(X_scaled.head())

"""PCA Feature Selection"""

# Initialize the PCA model
pca = PCA()

# Fit the PCA model to the scaled data
pca.fit(X_scaled)

# Transform the data into the principal components
X_pca = pca.transform(X_scaled)

# Explained variance ratio - shows the proportion of variance captured by each principal component
print("Explained Variance Ratio by each component:")
print(pca.explained_variance_ratio_)

# Cumulative explained variance - shows how much variance is explained by the first N components
print("\nCumulative Explained Variance:")
print(pca.explained_variance_ratio_.cumsum())

# Plot the explained variance to see how many components to keep
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Explain 95% of the variance
n_components = (pca.explained_variance_ratio_.cumsum() <= 0.95).sum()

print(f"\nNumber of components explaining 95% of the variance: {n_components}")

# Reduce the dataset to the first 10 principal components
X_pca_reduced = X_pca[:, :10]

# Convert back to a DataFrame for readability
X_pca_reduced_df = pd.DataFrame(X_pca_reduced, columns=[f'PC{i+1}' for i in range(10)])

#Top 3 Contributing Features for Each Principal Component

# Get the PCA components (loadings) from the PCA object
loadings = pca.components_

# Create a DataFrame of loadings with original feature names as columns
loadings_df = pd.DataFrame(loadings, columns=X.columns)

# Get the absolute value of the loadings to determine feature importance
loadings_abs = loadings_df.abs()

# For each principal component, get the top contributing features
top_features = {}
for i in range(loadings_abs.shape[0]):
    top_feature_indices = loadings_abs.iloc[i].nlargest(3).index  # Get top 3 features for each component
    top_features[f'PC{i+1}'] = top_feature_indices.tolist()

# Display the most significant features for each principal component with pprint
print("Top 3 Contributing Features for Each Principal Component:")
pprint(top_features)

"""Splitting the Data"""

# Split data into features (X) and target (y)
X = X_pca_reduced_df
y = data['race_outcome']

# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

"""Classification with Random Forest"""

# Create the model
model = RandomForestClassifier(class_weight='balanced', random_state=1234)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix
print("Confusion Matrix:")

# Plot the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'],cbar=False)

# Labels, title, and display
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Cross-Validation for generalization check
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f'Mean CV Accuracy: {cv_scores.mean():.2f}')

"""Classification with SVM"""

# Create the model
svm_model = SVC(kernel='rbf', C=10, gamma = 0.01, random_state=1234)  # RBF kernel works well for non-linear data

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate performance
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm:.2f}')

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)

# Confusion Matrix
print("Confusion Matrix:")

# Plot the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'],cbar=False)

# Labels, title, and display
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Perform cross-validation
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)  # 5-fold cross-validation

# Print cross-validation results
print(f'Mean CV Accuracy: {cv_scores.mean():.2f}')

"""Clustering with K-Means"""

# Elbow Method to determine the optimal number of clusters
distortions = []
K_range = range(1, 11)  # You can change this range based on the expected number of clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=1234)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(K_range, distortions, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.show()

# Number of clusters (k)
optimal_k = 3

# Perform K-Means with the chosen k
kmeans = KMeans(n_clusters=optimal_k, random_state=1234)
kmeans.fit(X)

# Get the cluster labels
cluster_labels = kmeans.labels_

# Add the cluster labels to your data for further analysis (optional)
X['Cluster'] = cluster_labels

# Print the cluster centers (optional)
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Print the first few rows of data with cluster labels
print(X.head())

# Silhouette Score
score = silhouette_score(X, kmeans.labels_)
print(f"Silhouette Score: {score}")

# Calculate Adjusted Rand Index
true_labels = data['race_outcome']
ari = adjusted_rand_score(true_labels, kmeans.labels_)
print(f"Adjusted Rand Index: {ari}")

# Calinski-Harabasz
ch_score = calinski_harabasz_score(X, kmeans.labels_)
print(f"Calinski-Harabasz Index: {ch_score}")

# Visualizing the Clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with the PCA results and cluster labels
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = cluster_labels

# Plotting the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, alpha=0.7, edgecolor='k')
plt.title('Clusters Visualized using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

"""Clustering with Hierarchical Clustering"""

X = X.iloc[:, :-1]

X

# Fit the AgglomerativeClustering model (Hierarchical Clustering)
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X)

# Create the Dendrogram (Hierarchical Tree) to visualize clusters
linked = linkage(X, 'ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Evaluate the clustering performance

# Silhouette Score
sil_score_hierarchical = silhouette_score(X, hierarchical_labels)

# Calculate Adjusted Rand Index
true_labels = data['race_outcome']
ari_hierarchical = adjusted_rand_score(true_labels, hierarchical_labels)

# Calinski-Harabasz
ch_score_hierarchical = calinski_harabasz_score(X, hierarchical_labels)

# Display the results
print(f"Silhouette Score for Hierarchical Clustering: {sil_score_hierarchical}")
print(f"Adjusted Rand Index for Hierarchical Clustering: {ari_hierarchical}")
print(f"Calinski-Harabasz Index for Hierarchical Clustering: {ch_score_hierarchical}")

# Visualizing the Clusters using PCA
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X)

# Create a DataFrame for the 2D PCA components and the cluster labels
df_pca = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
df_pca['Cluster'] = hierarchical_labels

# Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, alpha=0.7, edgecolor='k')
plt.title('Hierarchical Clustering Results with PCA Visualization', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Cluster', loc='best', fontsize=12)
plt.show()

# Sample labels from KMeans and Agglomerative clustering
kmeans_labels = kmeans.labels_
agglo_labels = hierarchical_labels

# Create a DataFrame for comparison
df_clusters = pd.DataFrame({
    "KMeans": kmeans_labels,
    "Agglomerative": agglo_labels
})

# Check where KMeans and Agglomerative assigned the same cluster
matching_labels = (df_clusters["KMeans"] == df_clusters["Agglomerative"]).sum()

# Calculate the percentage of matching labels
accuracy = matching_labels / len(df_clusters) * 100
print(f"Percentage of matching labels: {accuracy:.2f}%")

### Bar Chart
plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df_clusters.melt(var_name="Method", value_name="Cluster"),
                   x="Cluster", hue="Method", palette="Set1")

# Add count labels on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title("Cluster Distribution: KMeans vs Agglomerative")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.legend(title="Method")
plt.show()

### Confusion Matrix
conf_matrix = pd.crosstab(df_clusters["KMeans"], df_clusters["Agglomerative"])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
plt.title("Confusion Matrix: KMeans vs Agglomerative")
plt.xlabel("Agglomerative Clusters")
plt.ylabel("KMeans Clusters")
plt.show()