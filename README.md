# F1 2023 Season Analysis Project

A comprehensive data analysis project examining various aspects of the 2023 Formula 1 season using machine learning techniques and statistical analysis.

## Project Overview

This project analyzes Formula 1 data from the 2023 season to uncover insights about:
- Tire degradation patterns across different tracks
- Team performance progression throughout the season to see if any underlying trends led to specific performances during 2024.
- Weather impact on lap times and performance
- Driver-specific performance analysis (focusing on Sergio Pérez) to gain insight into Red Bull's 2024 performance. 
- Pit stop efficiency and strategies
- Lap time clustering and track characteristics

## Project Structure

```
f1_2023/
├── library_data_2023/           # Data processing modules
│   ├── __init__.py
│   ├── events.py               # Events data processing
│   ├── fastestlaps.py          # Fastest laps data processing
│   ├── laps.py                 # Main laps data processing
│   ├── results.py              # Race results data processing
│   ├── sprintlaps.py           # Sprint race laps processing
│   ├── sprintresults.py        # Sprint race results processing
│   ├── telemetry.py            # Telemetry data processing
│   ├── testing_modules.py      # Module testing utilities
│   └── weather.py              # Weather data processing
├── f1_analysis/                # Analysis modules
│   ├── README.md
│   ├── analysisA.py            # Tire degradation analysis (K-Means)
│   ├── analysisB.py            # McLaren progression analysis (Linear Regression)
│   ├── analysisC.py            # Weather impact analysis (Random Forest/Gradient Boosting)
│   ├── analysisD.py            # Pérez performance analysis (Multiple ML models)
│   ├── analysisE.py            # Lap time clustering (K-Means + Gradient Boosting)
│   ├── analysisF.py            # Pit stop analysis (Classification/Regression)
│   └── analysisG.py            # Team improvement analysis (Linear/Random Forest)
├── *.png                       # Generated analysis plots
├── .gitignore
└── README.md
```

## Data Processing Modules

### Core Data Loaders (`library_data_2023/`)

Each module provides a standardized interface for loading and cleaning F1 data:

- **[`events.py`](library_data_2023/events.py)**: [`open_events_data()`](library_data_2023/events.py) - Race weekend events and sessions
- **[`laps.py`](library_data_2023/laps.py)**: [`open_laps_data()`](library_data_2023/laps.py) - Individual lap data with timing and tire information
- **[`results.py`](library_data_2023/results.py)**: [`open_results_data()`](library_data_2023/results.py) - Final race results and standings
- **[`telemetry.py`](library_data_2023/telemetry.py)**: [`open_telemetry_data()`](library_data_2023/telemetry.py) - Detailed car telemetry (speed, throttle, brake, etc.)
- **[`weather.py`](library_data_2023/weather.py)**: [`open_weather_data()`](library_data_2023/weather.py) - Weather conditions during sessions
- **[`fastestlaps.py`](library_data_2023/fastestlaps.py)**: [`open_fastestlaps_data()`](library_data_2023/fastestlaps.py) - Fastest lap records
- **[`sprintlaps.py`](library_data_2023/sprintlaps.py)**: [`open_sprintlaps_data()`](library_data_2023/sprintlaps.py) - Sprint race lap data
- **[`sprintresults.py`](library_data_2023/sprintresults.py)**: [`open_sprintresults_data()`](library_data_2023/sprintresults.py) - Sprint race results

All modules include:
- Data type conversions (int, float, bool, string, timedelta, datetime)
- Label encoding for categorical variables
- Null value handling
- Standardized column naming

## Analysis Modules

### A. Tire Degradation Analysis ([`analysisA.py`](f1_analysis/analysisA.py))
**Objective**: Identify tracks with highest tire degradation using K-Means clustering

**Key Functions**:
- [`highest_tyre_deg()`](f1_analysis/analysisA.py) - Main analysis function

**Machine Learning**: K-Means clustering with WCSS elbow method
**Features**: Tire life vs. lap time regression slopes, median lap times
**Output**: Track degradation rankings, cluster visualizations

### B. McLaren Progression Analysis ([`analysisB.py`](f1_analysis/analysisB.py))
**Objective**: Track McLaren's performance improvement throughout 2023 season

**Key Functions**:
- [`mclaren_linreg()`](f1_analysis/analysisB.py) - Performance trend analysis
- [`mclaren_linreg_plots()`](f1_analysis/analysisB.py) - Visualization generation

**Machine Learning**: Linear Regression
**Features**: Race positions, grid positions, points per race
**Output**: Trend analysis with R² scores, progression plots

### C. Weather Impact Analysis ([`analysisC.py`](f1_analysis/analysisC.py))
**Objective**: Determine weather effects on lap time performance

**Key Functions**:
- [`preprocess_weather()`](f1_analysis/analysisC.py) - Data preparation and merging
- [`feature_forging()`](f1_analysis/analysisC.py) - Advanced feature engineering
- [`weather_gb()`](f1_analysis/analysisC.py) - Model training and evaluation

**Machine Learning**: Random Forest Classifier, Gradient Boosting Classifier
**Features**: Temperature, humidity, wind, tire compounds, track characteristics
**Output**: Feature importance rankings, weather effect visualizations

### D. Pérez Performance Analysis ([`analysisD.py`](f1_analysis/analysisD.py))
**Objective**: Comprehensive analysis of Sergio Pérez's performance patterns

**Key Functions**:
- [`straights_speed()`](f1_analysis/analysisD.py) - Straight-line speed analysis
- [`corners()`](f1_analysis/analysisD.py) - Cornering performance analysis

**Machine Learning**: Linear Regression, Ridge Regression, Random Forest (Regression & Classification)
**Features**: DRS usage, telemetry data, tire information, weather conditions
**Output**: Race-by-race performance evolution, feature importance trends

### E. Lap Time Clustering ([`analysisE.py`](f1_analysis/analysisE.py))
**Objective**: Cluster tracks by lap time characteristics and weather patterns

**Key Functions**:
- [`var_lap_times()`](f1_analysis/analysisE.py) - Data loading
- [`analyze_lap_times()`](f1_analysis/analysisE.py) - Clustering and prediction

**Machine Learning**: K-Means clustering, Gradient Boosting Regressor
**Features**: Track temperature, air temperature, humidity
**Output**: Track clustering, lap time predictions

### F. Pit Stop Analysis ([`analysisF.py`](f1_analysis/analysisF.py))
**Objective**: Analyze pit stop efficiency and team performance

**Key Functions**:
- [`pit_stops()`](f1_analysis/analysisF.py) - Data preparation
- [`analyze_pit_stops()`](f1_analysis/analysisF.py) - Statistical analysis

**Machine Learning**: Logistic Regression, Random Forest (Classification & Regression)
**Features**: Pit stop duration calculations
**Output**: Team rankings, pit stop time distributions

### G. Team Improvement Analysis ([`analysisG.py`](f1_analysis/analysisG.py))
**Objective**: Track performance improvements across all teams

**Key Functions**:
- [`team_improvement_data()`](f1_analysis/analysisG.py) - Data aggregation
- [`analyze_team_improvement()`](f1_analysis/analysisG.py) - Trend analysis

**Machine Learning**: Linear Regression, Random Forest Regressor
**Features**: Average lap times by team and round
**Output**: Team-specific improvement trends, RMSE comparisons

## Installation and Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd f1_2023
```

2. **Install required dependencies**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. **Data Setup**:
   - Place CSV data files in `library_data_2023/data_2023/` directory
   - Required files: `Laps.csv`, `Events.csv`, `Results.csv`, `Telemetry.csv`, `Weather.csv`, etc.

## Usage

### Running Individual Analyses

Each analysis module can be run independently:

```bash
# Tire degradation analysis
python f1_analysis/analysisA.py

# McLaren progression analysis  
python f1_analysis/analysisB.py

# Weather impact analysis
python f1_analysis/analysisC.py

# Pérez performance analysis
python f1_analysis/analysisD.py

# Lap time clustering
python f1_analysis/analysisE.py

# Pit stop analysis
python f1_analysis/analysisF.py

# Team improvement analysis
python f1_analysis/analysisG.py
```

### Using Data Modules

```python
from library_data_2023 import laps, events, weather

# Load data
laps_df = laps.open_laps_data()
events_df = events.open_events_data()
weather_df = weather.open_weather_data()

# Data is pre-processed and ready for analysis
print(laps_df.head())
```

## Key Features

### Data Processing
- **Automated data type conversion**: Ensures consistent data types across all datasets
- **Label encoding**: Categorical variables are properly encoded for ML models
- **Time series handling**: Proper datetime and timedelta conversions
- **Missing value management**: Systematic handling of null values

### Machine Learning Models
- **K-Means Clustering**: Track and performance grouping
- **Linear/Ridge Regression**: Trend analysis and performance prediction
- **Random Forest**: Both classification and regression tasks
- **Gradient Boosting**: Advanced predictive modeling
- **Cross-validation**: Model validation and hyperparameter tuning

### Visualizations
- **Scatter plots with clustering**: Color-coded performance groups
- **Time series analysis**: Progression tracking over races
- **Feature importance charts**: ML model interpretability
- **Box plots**: Performance distribution analysis
- **Heatmaps**: Correlation and confusion matrices

## Output Files

Generated visualizations are saved as high-resolution PNG files:
- `f1_tire_degradation_kmeans.png` - Tire degradation clustering
- `mclaren_progression.png` - McLaren season progression
- `weather_feature_importance.png` - Weather impact analysis
- `perez_*.png` - Various Pérez performance analyses
- `*_clusters.png` - Clustering visualizations
- And many more...

## Data Sources

This project expects Formula 1 data in CSV format with the following structure:
- Race results and lap times
- Telemetry data (speed, throttle, brake, DRS)
- Weather conditions
- Tire compound and degradation information
- Session timing data

## Contributing

When adding new analysis modules:
1. Follow the existing naming convention (`analysisX.py`)
2. Include comprehensive docstrings for all functions
3. Use the standardized data loading modules from `library_data_2023/`
4. Save visualizations with descriptive filenames
5. Include error handling and data validation
