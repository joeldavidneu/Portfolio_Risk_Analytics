# Portfolio Risk Analytics

A Python application for analysing the **one-day market risk** of a portfolio containing BMW stock, gold and a bond ETF. It compares historical simulation with Student-t Monte Carlo simulation, evaluates a simplified stress scenario and backtests historical Value at Risk.

Results are exported automatically to a three-page PDF report with tables and charts.

## Features

- CSV import with chronological sorting and alignment on common dates.
- User-defined portfolio weights, confidence level and current portfolio value.
- Historical and Student-t Monte Carlo **Value at Risk (VaR)** and **Expected Shortfall (ES)**.
- **100,000 Monte Carlo scenarios** with a fixed random seed.
- Stress sensitivity analysis using a fixed loss multiplier of three.
- Rolling historical VaR backtest with the **Kupiec coverage test**.
- Automatic PDF reporting.

## Results Preview

### Historical and simulated losses

The charts compare observed portfolio loss scenarios with Student-t Monte Carlo scenarios, followed by baseline-versus-stress comparisons for each method. Dashed lines indicate VaR. Positive values represent losses; negative values represent gains.

<img src="images/loss_distributions.png" alt="Historical and Monte Carlo loss distributions, including threefold stress scenarios and VaR thresholds" width="760">

The stress scenario multiplies both losses and gains by three. Consequently, VaR and ES also triple. It is a transparent sensitivity exercise, not a replay of a historical crisis or a regulatory stress test.

sie
The historical VaR forecast is estimated from preceding observations and compared with each subsequent observed loss. Orange markers identify losses above the forecast threshold.

![Observed one-day losses against rolling historical VaR, with exceptions highlighted](images/var_backtest.png)

This example uses a **97% confidence level** and shows **three exceptions**. Their count alone does not establish model quality: the Kupiec test compares the observed frequency with the expected frequency over the test sample. It does not test whether exceptions cluster in time.

These images show one example run; results depend on the portfolio inputs and price history.

[View the example PDF report](risk_report.pdf)

## Installation

Install Python, then install the dependencies from the project folder:

```bash
pip install -r requirements.txt
```

## Input Data

Place these CSV files in the project folder:

| File | Asset |
|---|---|
| `history_BMW-stock.csv` | BMW stock |
| `history_Gold.csv` | Gold |
| `history_Bonds_ETF.csv` | Bond ETF |

Files must use semicolons (`;`) as separators and include the columns `Datum` and `Schluss`. Prices use German number formatting, such as `1.234,56`. Supported date formats are `DD.MM.YYYY`, `DD.MM.YY` and `YYYY-MM-DD`.

The program sorts observations and keeps common dates. Input prices must be positive and dates must not be duplicated. Verify currency consistency, split and distribution adjustments, and gaps in the common date sequence before interpreting results as one-day EUR risk. Data sources and redistribution permissions should be documented for the selected files.

## Usage

Run the main program from the project folder:

```bash
python calculator.py
```

Enter:

1. Asset weights in percent, totalling 100%.
2. Confidence level in percent, greater than 0 and less than 100.
3. Current portfolio value in EUR, without thousands separators.

The program saves **`risk_report.pdf`** in the working directory and prints its location. An existing report is overwritten.

## Methodology

**Portfolio returns:** Simple daily asset returns are combined using the selected weights. Constant weights imply daily rebalancing. Monetary loss is the negative portfolio return multiplied by current portfolio value.

**Historical simulation:** VaR and ES are calculated directly from observed portfolio loss scenarios without imposing a parametric distribution.

**Monte Carlo:** A Student-t distribution is fitted to historical portfolio returns using maximum likelihood. The model generates 100,000 independent one-day outcomes with seed 42. These are alternative daily scenarios, not a multi-day price path.

**Risk measures:** VaR is the selected loss quantile. ES is estimated as the mean of scenario losses at or above VaR. With finite samples and tied observations, this tail-average estimate may differ from an exact fractional-tail calculation.

**Backtesting:** A rolling historical VaR is compared with subsequent losses. A Kupiec p-value below 0.05 rejects the expected exception rate at the 5% significance level. A larger p-value does not prove the model is correct. Only historical VaR is backtested.

## Limitations

- Estimates depend on historical data and modelling assumptions; VaR is not a maximum loss.
- Student-t simple returns and scaled losses can imply losses above portfolio value, which is infeasible for an unleveraged long-only portfolio.
- Changing volatility, explicit stressed asset dependencies, liquidity risk, transaction costs and parameter uncertainty are not modelled.
- Sparse tail observations limit estimation and backtest reliability. The distribution charts may omit extreme tails for readability.

## Documentation

The detailed German project report explains the workflow, calculations, backtest and limitations.

<!-- After adding project_documentation.pdf to this folder, replace this comment with:
[Read the project documentation (German)](project_documentation.pdf)
-->

## Development

Personal educational project developed with AI assistance for parts of the implementation and explanations. Not investment advice.
