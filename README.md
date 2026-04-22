# Market Regime Detector

Market Regime Detector is a Streamlit-based financial analytics dashboard for detecting and analyzing market regimes from time-series data. The project combines an academic machine learning pipeline with a professional quantitative dashboard for regime monitoring, feature analysis, volatility review, returns analysis, and strategy diagnostics.

The goal is not to predict exact future prices. Instead, the app identifies different market states using unsupervised learning and then helps interpret those states through financial analytics.

## Core Idea

Financial markets often move through different regimes such as trending, sideways, high-volatility, or low-volatility periods. These regimes are not directly labeled in raw market data, so the project uses unsupervised clustering to group similar market behavior.

The app converts raw price data into financial features such as returns, rolling mean, rolling volatility, and momentum. KMeans clustering is then used to assign each valid observation to a market regime. The resulting regimes are analyzed through charts, transition probabilities, performance metrics, and strategy diagnostics.

## Key Features

- CSV upload with reusable session-state data flow
- Date, price, and feature column selection
- Dataset preview, missing-value summary, and shape statistics
- Exploratory data analysis with summary statistics, distributions, correlations, and PCA visualization
- Data cleaning with missing-value handling and IQR-based outlier filtering
- Financial feature engineering using returns, rolling mean, rolling volatility, and momentum
- Feature selection using correlation filtering and variance thresholding
- Time-based train/test splitting for financial time-series data
- KMeans clustering for market regime detection
- Validation with TimeSeriesSplit
- Model evaluation using silhouette score, inertia, and Davies-Bouldin score
- Hyperparameter tuning for the number of clusters
- Regime monitor with price overlays, state timeline, transition matrix, and Markov-style regime projection
- Quantitative tools for correlation, volatility, and returns analysis
- Strategy diagnostics with cumulative returns, drawdown, Sharpe, Sortino, CAGR, and regime contribution analysis

## Methodology

### Unsupervised Learning

The project uses unsupervised learning because market regimes are not provided as labeled classes in the input dataset. The model discovers groups of similar observations based on engineered financial features.

### KMeans Clustering

KMeans is used as the primary regime detection model. It groups observations into `K` clusters based on feature similarity. Each cluster is interpreted as a market regime.

The default number of clusters is 3, but the app includes tuning tools to compare different values of `K`.

### PCA Visualization

Principal Component Analysis is used for visualization. It compresses multiple selected features into two dimensions so users can inspect the overall structure of the feature space. PCA is used as an exploratory tool, not as the main regime detection model.

### Time-Series Validation

The app uses chronological splitting and TimeSeriesSplit instead of random splitting. This is important because financial data is time-dependent, and future observations should not leak into past training.

### Evaluation Metrics

Because this is an unsupervised clustering problem, classification metrics such as accuracy, precision, recall, and F1-score are not the primary evaluation metrics.

The app uses:

- Silhouette Score: measures how well separated the clusters are
- Inertia: measures within-cluster compactness
- Davies-Bouldin Score: measures cluster separation and compactness, where lower is better

### Markov-Style Regime Projection

The app estimates future regime probabilities from historical regime transitions. This does not predict stock prices. It estimates the likelihood of remaining in the current regime or moving to another regime based on observed transition behavior.

## How The System Works

1. Upload a CSV file.
2. Select the date column, price column, and feature columns.
3. Clean missing values and optionally remove IQR-based outliers.
4. Engineer financial features such as returns, rolling mean, rolling volatility, and momentum.
5. Select useful features using correlation and variance-based filtering.
6. Split the data chronologically into training and evaluation periods.
7. Train KMeans on valid feature rows.
8. Assign regime labels back to the processed dataframe.
9. Validate clustering quality using time-series validation and clustering metrics.
10. Visualize regimes, transitions, volatility, returns, and strategy behavior in the dashboard.

## Dashboard Modules

### Landing Page

Introduces the project as a financial analytics workspace and provides entry points into the dashboard.

### Dataset Overview

Used for uploading the dataset, selecting column roles, previewing the data, reviewing missing values, and viewing PCA-based feature-space structure.

### ML Pipeline

Contains the academic pipeline flow: problem setup, EDA, data cleaning, feature engineering, feature selection, time-based split, model training, validation, performance metrics, overfitting/underfitting interpretation, and hyperparameter tuning.

### Market Regime Analysis

The main regime monitoring page. It shows the price series with regime overlays, regime timeline, cluster distribution, transition matrix, Markov-style forecast, regime summary, and feature profile.

### Correlation Matrix

Shows relationships between selected model or financial features using a heatmap.

### Volatility Analysis

Analyzes rolling volatility using the selected price series and window size.

### Returns Analysis

Shows return distribution and cumulative returns for the selected market series.

### Strategy Desk

Evaluates a regime-aware strategy layer using performance metrics such as cumulative returns, drawdown, Sharpe ratio, Sortino ratio, CAGR, hit rate, and contribution by regime.

## How To Use The App

1. Open the app.
2. Go to Dataset Overview.
3. Upload a CSV file with historical market data.
4. Select the date column, price column, and numeric feature columns.
5. Open ML Pipeline.
6. Configure cleaning, feature engineering, feature selection, and the number of clusters.
7. Check clustering metrics such as silhouette score and Davies-Bouldin score.
8. Open Market Regime Analysis to review detected regimes and transition behavior.
9. Use Correlation Matrix, Volatility Analysis, Returns Analysis, and Strategy Desk for deeper quantitative review.

## Run Locally

Clone the repository:

```bash
git clone https://github.com/Prerna1313/Market-Regime-Detector.git
cd Market-Regime-Detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

If `streamlit` is not recognized, use:

```bash
python -m streamlit run app.py
```

## Project Structure

```text
app.py              Main Streamlit application and page routing
data_utils.py       Data loading, cleaning, and feature engineering helpers
eda_utils.py        EDA charts, PCA visualization, and model diagnostic charts
model_utils.py      KMeans training, validation, tuning, and regime estimation logic
quant_utils.py      Regime analytics, transition logic, strategy metrics, and quant charts
ui_utils.py         Dashboard styling, layout helpers, and UI rendering utilities
requirements.txt    Python dependencies
```
