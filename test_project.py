from datetime import date

import numpy as np
import pytest

from final_project.calculator import (
    align_prices,
    calculate_backtest,
    calculate_historical,
    calculate_monte_carlo,
    calculate_portfolio_returns,
    calculate_returns,
    calculate_stress_test,
    kupiec_test,
    load_prices,
)


def test_load_prices(tmp_path):
    file = tmp_path / "asset.csv"
    file.write_text(
        "Datum;Eröffnung;Hoch;Tief;Schluss\n"
        "03.01.2024;1;1;1;1.101,00\n"
        "02.01.2024;1;1;1;100,50\n",
        encoding="utf-8-sig",
    )
    prices = load_prices(file)
    assert list(prices) == [date(2024, 1, 2), date(2024, 1, 3)]  # sorted ascending
    assert prices[date(2024, 1, 2)] == pytest.approx(100.5)
    assert prices[date(2024, 1, 3)] == pytest.approx(1101.0)


def test_align_prices():
    d1, d2, d3 = date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)
    data = {"A": {d1: 1, d2: 2, d3: 3}, "B": {d2: 5, d3: 6}}
    aligned = align_prices(data)
    assert list(aligned["A"]) == [d2, d3]
    assert list(aligned["B"]) == [d2, d3]
    with pytest.raises(ValueError):
        align_prices({"A": {d1: 1}, "B": {d2: 1}})


def test_calculate_returns():
    prices = {date(2024, 1, 1): 100.0, date(2024, 1, 2): 110.0, date(2024, 1, 3): 99.0}
    assert calculate_returns(prices) == pytest.approx([0.10, -0.10])


def test_calculate_portfolio_returns():
    asset_returns = {"A": [0.01, 0.02], "B": [-0.01, 0.02]}
    weights = {"A": 0.5, "B": 0.5}
    assert calculate_portfolio_returns(asset_returns, weights) == pytest.approx([0.0, 0.02])
    with pytest.raises(ValueError):
        calculate_portfolio_returns({"A": [0.01], "B": [0.01, 0.02]}, weights)


def test_calculate_historical():
    returns = np.linspace(-0.10, 0.10, 201)  # losses -100 ... +100 EUR in 1 EUR steps
    result = calculate_historical(returns, 0.95, 1000)
    assert result["var"] == pytest.approx(90.0)
    assert result["es"] == pytest.approx(95.0)  # mean of losses 90..100
    with pytest.raises(ValueError):
        calculate_historical([0.01], 0.95, 1000)


def test_calculate_monte_carlo():
    returns = np.random.default_rng(1).normal(0, 0.01, 2000)
    result = calculate_monte_carlo(returns, 0.99, 1000)
    assert result["var"] == pytest.approx(23.3, rel=0.25)  # 1000 * 0.01 * 2.326
    assert result["es"] >= result["var"]
    assert result["degrees_of_freedom"] > 1
    assert calculate_monte_carlo(returns, 0.99, 1000)["var"] == result["var"]  # fixed seed
    with pytest.raises(ValueError):
        calculate_monte_carlo([0.01, 0.01, 0.01], 0.99, 1000)


def test_calculate_stress_test():
    losses = np.linspace(-100, 100, 201)
    result = calculate_stress_test(losses, 0.95)
    assert result["var"] == pytest.approx(3 * 90.0)
    assert result["es"] == pytest.approx(3 * 95.0)
    assert calculate_stress_test(losses, 0.95, stress_factor=2)["var"] == pytest.approx(2 * 90.0)
    with pytest.raises(ValueError):
        calculate_stress_test(losses, 0.95, stress_factor=0)


def test_kupiec_test():
    lr, p_value = kupiec_test(10, 1000, 0.99)
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert p_value == pytest.approx(1.0)
    # Far too many and far too few exceedances are both rejected
    assert kupiec_test(40, 1000, 0.99)[1] < 0.01
    assert kupiec_test(0, 1000, 0.99)[1] < 0.01
    with pytest.raises(ValueError):
        kupiec_test(5, 0, 0.99)


def test_calculate_backtest():
    returns = np.random.default_rng(3).normal(0, 0.01, 400)
    result = calculate_backtest(returns, 0.99, 1000, window=250)
    assert result["observations"] == 150
    assert result["violation_count"] == int(result["violations"].sum())
    losses = -returns * 1000
    assert result["var_series"][0] == pytest.approx(np.quantile(losses[:250], 0.99))

    calm_then_crash = [0.0] * 250 + [-0.05]
    crash = calculate_backtest(calm_then_crash, 0.99, 1000, window=250)
    assert crash["observations"] == 1
    assert crash["violation_count"] == 1

    with pytest.raises(ValueError):
        calculate_backtest(returns[:200], 0.99, 1000, window=250)
    with pytest.raises(ValueError):
        calculate_backtest(returns, 0.99, 1000, window=10)