# Semiconductor Pairs Trading Research Pipeline

A Python-based quantitative research project investigating a simple statistical pairs trading strategy within the semiconductor sector. The project implements an end-to-end research workflow covering market data collection, data cleaning, exploratory analysis, signal construction, strategy implementation, and out-of-sample evaluation.

The strategy tests whether a basic mean-reversion framework can identify relative mispricing between highly liquid semiconductor equities. Candidate pairs are selected using exploratory price and correlation analysis, then evaluated using a rule-based long/short strategy on a separate trading period.

The objective of the project is not to present a production-ready or consistently profitable trading system. Instead, it demonstrates practical quantitative research skills: building a reproducible research pipeline, working with real equity price data, separating in-sample analysis from out-of-sample testing, and interpreting weak or inconsistent performance with appropriate caution.

## What This Project Demonstrates

- Building a structured multi-stage research pipeline in Python
- Downloading and organizing historical market data
- Performing exploratory analysis to identify candidate trading pairs
- Constructing and testing a simple mean-reversion trading signal
- Implementing a rule-based strategy
- Separating research and trading periods for out-of-sample evaluation

## Workflow

1. Download and prepare historical semiconductor stock data
2. Split the dataset into analysis and trading periods
3. Explore relationships between stock prices and returns 
4. Select candidate pairs for relative-value testing by comparing correlation and cointegration
5. Implement a threshold-based pairs trading model
6. Evaluate results across multiple stock pairs

## Repository Structure

```text
S1_Get_Data.ipynb       # Downloads and prepares market data
S2_Data_analysis.ipynb  # Explores relationships between stocks and identifies candidate pairs
S3_Pairs_trading.py     # Contains the core pairs trading strategy implementation on the analysis data and test data set
S4_Evaluation.ipynb     # Evaluates strategy performance on selected pairs
Data/                   # Stores processed datasets used in the project
```
