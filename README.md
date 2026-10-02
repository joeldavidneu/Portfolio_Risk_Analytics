# Portfolio Risk Lab

A Python tool for measuring the one-day market risk of a portfolio containing BMW stock, gold and a bond ETF.

## Features

- Historical and Student-t Monte Carlo Value at Risk (VaR) and Expected Shortfall (ES).
- 100,000 simulated scenarios with a fixed random seed.
- Stress sensitivity analysis using a loss multiplier of three.
- Rolling historical VaR backtest with the Kupiec test.
- Automatic three-page PDF report with tables and charts.

## Installation

Install Python and the project dependencies:

```bash
pip install -r requirements.txt
```

## Input Files

Place these CSV files in the project folder:

- `history_BMW-stock.csv`
- `history_Gold.csv`
- `history_Bonds_ETF.csv`

Files must use `;` as the separator and include the columns `Datum` (date) and `Schluss` (closing price). Prices use German formatting, such as `1.234,56`. Supported date formats are `DD.MM.YYYY`, `DD.MM.YY` and `YYYY-MM-DD`.

The program sorts prices and selects common dates. Provide positive prices without duplicate dates and check currency consistency and adjustments for splits and distributions. Market data must be obtained separately unless redistribution is permitted.

## Usage

Run from the project folder:

```bash
python project.py
```

Enter:

1. Asset weights in percent, totalling 100%.
2. Confidence level in percent, between 0 and 100 (exclusive).
3. Current portfolio value in EUR, without thousands separators.

The program creates **`risk_report.pdf`** in the working directory and prints its location. An existing report is overwritten.

## Model Notes

Returns use constant portfolio weights, implying daily rebalancing. All risk estimates refer to one trading day. The stress test triples losses and gains, so VaR and ES also triple; it is not a historical crisis replay.

Student-t returns can imply infeasible losses above portfolio value. The Kupiec test checks historical VaR exception frequency, not exception clustering. Results depend on the input data and model assumptions.

Personal educational project, developed with AI assistance. Not investment advice.
