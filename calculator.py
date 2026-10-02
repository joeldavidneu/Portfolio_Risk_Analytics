import csv
import math
from datetime import datetime

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2, t

from pdf_report import create_pdf_report


# ---------------------------------------------------------------- Data


def parse_date(text):
    for fmt in ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unknown date format: {text}")


def parse_number(text):
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    return float(text)


def load_prices(filename):
    prices = {}

    with open(filename, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row in reader:
            prices[parse_date(row["Datum"])] = parse_number(row["Schluss"])

    return dict(sorted(prices.items()))


def align_prices(asset_data):
    """Keep only the days on which ALL assets have a price."""
    common_dates = set.intersection(*(set(prices) for prices in asset_data.values()))
    if len(common_dates) < 2:
        raise ValueError("The assets share fewer than two common dates.")
    return {
        asset: {date: prices[date] for date in sorted(common_dates)}
        for asset, prices in asset_data.items()
    }


# -------------------------------------------------------------- Inputs


def get_weights(files):
    if not files:
        raise ValueError("No assets available.")

    while True:
        percentages = {}

        for asset in files:
            while True:
                try:
                    percentage = float(
                        input(f"Enter portfolio share for {asset} (%): ").replace(",", ".")
                    )
                except ValueError:
                    print("Please enter a valid number.")
                    continue

                if not math.isfinite(percentage):
                    print("Please enter a finite number.")
                    continue

                if not 0 <= percentage <= 100:
                    print("The share must be between 0 and 100.")
                    continue

                percentages[asset] = percentage
                break

        total = sum(percentages.values())

        if not math.isclose(total, 100.0, rel_tol=0, abs_tol=1e-6):
            print(
                f"The shares add up to {total:.8f}%. "
                "They must total 100%. Please enter all shares again."
            )
            continue

        return {asset: percentage / 100 for asset, percentage in percentages.items()}


def get_confidence_level():
    while True:
        try:
            percentage = float(
                input("Enter confidence level in % (e.g. 97.5): ").replace(",", ".")
            )
        except ValueError:
            print("Please enter a valid number.")
            continue

        if not 0 < percentage < 100:
            print("The confidence level must be between 0 and 100.")
            continue

        return percentage / 100


def get_portfolio_value():
    while True:
        try:
            portfolio_value = float(
                input(
                    "Enter current portfolio value in EUR: "
                ).replace(",", ".")
            )
        except ValueError:
            print("Please enter a valid number without thousands separators.")
            continue

        if not math.isfinite(portfolio_value) or portfolio_value <= 0:
            print("Portfolio value must be a finite number greater than zero.")
            continue

        return portfolio_value


# ---------------------------------------------------------- Calculator


def calculate_returns(prices):
    closing_prices = list(prices.values())
    returns = []

    for i in range(1, len(closing_prices)):
        previous_price = closing_prices[i - 1]
        current_price = closing_prices[i]
        returns.append((current_price / previous_price) - 1)

    return returns


def calculate_portfolio_returns(asset_returns, weights):
    portfolio_returns = []

    lengths = {len(returns) for returns in asset_returns.values()}

    if len(lengths) != 1:
        raise ValueError("All assets must have the same number of returns.")

    number_of_days = len(next(iter(asset_returns.values())))

    for i in range(number_of_days):
        daily_return = 0.0

        for asset, returns in asset_returns.items():
            daily_return += weights[asset] * returns[i]

        portfolio_returns.append(daily_return)

    return portfolio_returns


def calculate_monte_carlo(portfolio_returns, confidence_level, portfolio_value):
    returns = np.asarray(portfolio_returns, dtype=float)

    if len(returns) < 2 or not np.all(np.isfinite(returns)):
        raise ValueError("At least two finite returns are required.")

    if np.std(returns) == 0:
        raise ValueError("Returns must not all be identical.")

    # Fit a Student-t distribution to historical portfolio returns.
    degrees_of_freedom, location, scale = t.fit(returns)

    if degrees_of_freedom <= 1:
        raise ValueError("The fitted distribution has no finite Expected Shortfall.")

    # Simulate independent one-day portfolio returns.
    simulated_returns = t.rvs(
        df=degrees_of_freedom,
        loc=location,
        scale=scale,
        size=100_000,
        random_state=42,
    )

    # Convert returns into monetary losses.
    simulated_losses = -simulated_returns * portfolio_value

    value_at_risk = np.quantile(simulated_losses, confidence_level)
    tail_losses = simulated_losses[simulated_losses >= value_at_risk]
    expected_shortfall = np.mean(tail_losses)

    return {
        "var": float(value_at_risk),
        "es": float(expected_shortfall),
        "simulated_losses": simulated_losses,
        "degrees_of_freedom": float(degrees_of_freedom),
    }


def calculate_historical(portfolio_returns, confidence_level, portfolio_value):
    returns = np.asarray(portfolio_returns, dtype=float)

    if len(returns) < 2 or not np.all(np.isfinite(returns)):
        raise ValueError("At least two finite returns are required.")

    historical_losses = -returns * portfolio_value
    value_at_risk = np.quantile(historical_losses, confidence_level)
    tail_losses = historical_losses[historical_losses >= value_at_risk]
    expected_shortfall = np.mean(tail_losses)

    return {
        "var": float(value_at_risk),
        "es": float(expected_shortfall),
        "historical_losses": historical_losses,
    }


def calculate_stress_test(losses, confidence_level, stress_factor=3.0):
    if stress_factor <= 0:
        raise ValueError("The stress factor must be positive.")

    stressed_losses = np.asarray(losses, dtype=float) * stress_factor
    value_at_risk = np.quantile(stressed_losses, confidence_level)
    tail_losses = stressed_losses[stressed_losses >= value_at_risk]
    expected_shortfall = np.mean(tail_losses)

    return {
        "var": float(value_at_risk),
        "es": float(expected_shortfall),
        "stressed_losses": stressed_losses,
    }


# ------------------------------------------------------------ Backtest


def kupiec_test(violations, observations, confidence_level):
    """Kupiec proportion-of-failures test.

    Checks whether the observed number of VaR exceedances fits the expected
    rate 1 - confidence_level. Returns (LR statistic, p-value).
    """
    if observations <= 0 or not 0 <= violations <= observations:
        raise ValueError("Need 0 <= violations <= observations and observations > 0.")

    expected_rate = 1 - confidence_level
    observed_rate = violations / observations

    log_likelihood_model = (
        xlogy(observations - violations, 1 - expected_rate)
        + xlogy(violations, expected_rate)
    )
    log_likelihood_observed = (
        xlogy(observations - violations, 1 - observed_rate)
        + xlogy(violations, observed_rate)
    )

    lr_statistic = max(float(-2 * (log_likelihood_model - log_likelihood_observed)), 0.0)
    p_value = float(chi2.sf(lr_statistic, df=1))

    return lr_statistic, p_value


def calculate_backtest(
    portfolio_returns, confidence_level, portfolio_value, window=250, dates=None
):
    returns = np.asarray(portfolio_returns, dtype=float)

    if window < 20:
        raise ValueError("The backtest window must be at least 20 days.")

    if len(returns) <= window or not np.all(np.isfinite(returns)):
        raise ValueError("More finite returns than the window length are required.")

    losses = -returns * portfolio_value

    var_series = np.array([
        np.quantile(losses[day - window:day], confidence_level)
        for day in range(window, len(losses))
    ])
    actual_losses = losses[window:]
    violations = actual_losses > var_series

    observations = len(actual_losses)
    violation_count = int(violations.sum())
    lr_statistic, p_value = kupiec_test(violation_count, observations, confidence_level)

    if dates is not None and len(dates) == len(returns):
        backtest_dates = list(dates)[window:]
    else:
        backtest_dates = None

    return {
        "window": window,
        "observations": observations,
        "violation_count": violation_count,
        "expected_violations": observations * (1 - confidence_level),
        "lr_statistic": lr_statistic,
        "p_value": p_value,
        "var_series": var_series,
        "actual_losses": actual_losses,
        "violations": violations,
        "dates": backtest_dates,
    }


# ---------------------------------------------------------------- main


def main():
    files = {
        "Stocks": "history_BMW-stock.csv",
        "Gold": "history_Gold.csv",
        "Bonds": "history_Bonds_ETF.csv",
    }

    asset_data = {asset: load_prices(filename) for asset, filename in files.items()}
    asset_data = align_prices(asset_data)

    weights = get_weights(files)
    confidence_level = get_confidence_level()
    portfolio_value = get_portfolio_value()

    asset_returns = {
        asset: calculate_returns(prices) for asset, prices in asset_data.items()
    }
    portfolio_returns = calculate_portfolio_returns(asset_returns, weights)

    # Return i belongs to the day after the first price, so drop the first date.
    return_dates = list(next(iter(asset_data.values())))[1:]

    monte_carlo_results = calculate_monte_carlo(
        portfolio_returns, confidence_level, portfolio_value
    )

    historical_results = calculate_historical(
        portfolio_returns, confidence_level, portfolio_value
    )

    monte_carlo_stress_results = calculate_stress_test(
        monte_carlo_results["simulated_losses"], confidence_level
    )

    historical_stress_results = calculate_stress_test(
        historical_results["historical_losses"], confidence_level
    )


    window = min(250, len(portfolio_returns) // 2)
    backtest_results = calculate_backtest(
        portfolio_returns, confidence_level, portfolio_value, window, return_dates
    )

    report_path = create_pdf_report(
        weights=weights,
        confidence_level=confidence_level,
        portfolio_value=portfolio_value,
        monte_carlo_results=monte_carlo_results,
        historical_results=historical_results,
        monte_carlo_stress_results=monte_carlo_stress_results,
        historical_stress_results=historical_stress_results,
        backtest_results=backtest_results,
    )

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
