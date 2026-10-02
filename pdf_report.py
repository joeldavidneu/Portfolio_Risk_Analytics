from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


BLUE = "#087F8C"
ORANGE = "#D46537"
DARK = "#172B4D"


def style_table(table):
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("white")

        if row == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#EFF4F8" if row % 2 else "white")


def draw_loss_distribution(
    ax,
    first,
    second,
    labels,
    title,
    confidence=None,
):
    """Compare two loss distributions using identical histogram bins."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    both = np.concatenate([first, second])
    low, high = np.quantile(both, [0.001, 0.999])

    bins = np.histogram_bin_edges(
        both,
        bins=60,
        range=(low, high),
    )

    ax.hist(
        first,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        color=BLUE,
        label=labels[0],
    )

    ax.hist(
        second,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        color=ORANGE,
        label=labels[1],
    )

    if confidence is not None:
        for losses, color, name in [
            (first, BLUE, labels[0]),
            (second, ORANGE, labels[1]),
        ]:
            ax.axvline(
                np.quantile(losses, confidence),
                color=color,
                linestyle="--",
                linewidth=1,
                label=f"VaR {name}",
            )

    ax.set_xlim(low, high)
    ax.set_title(
        title,
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("One-day loss (EUR)")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)


def create_pdf_report(
    weights,
    confidence_level,
    portfolio_value,
    monte_carlo_results,
    historical_results,
    monte_carlo_stress_results,
    historical_stress_results,
    backtest_results=None,
    filename="risk_report.pdf",
):
    """Save a risk report with two or three pages.

    The report uses already calculated monetary losses.
    The absolute path of the generated PDF is returned.
    """
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    historical_losses = historical_results["historical_losses"]
    simulated_losses = monte_carlo_results["simulated_losses"]

    historical_stress_losses = historical_stress_results[
        "stressed_losses"
    ]
    simulated_stress_losses = monte_carlo_stress_results[
        "stressed_losses"
    ]

    total_pages = 3 if backtest_results is not None else 2

    scenarios = [
        ("Historical", "Baseline", historical_results),
        ("Monte Carlo", "Baseline", monte_carlo_results),
        ("Historical", "Stress x3", historical_stress_results),
        ("Monte Carlo", "Stress x3", monte_carlo_stress_results),
    ]

    table_rows = []

    for method, scenario, result in scenarios:
        table_rows.append(
            [
                method,
                scenario,
                f"{result['var']:,.2f}",
                f"{result['var'] / portfolio_value:.2%}",
                f"{result['es']:,.2f}",
                f"{result['es'] / portfolio_value:.2%}",
            ]
        )

    style = {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": DARK,
        "text.color": DARK,
        "axes.titlecolor": DARK,
    }

    with plt.rc_context(style), PdfPages(output_path) as pdf:
        pdf.infodict()["Title"] = "Portfolio Risk Report"

        # Page 1
        fig = plt.figure(figsize=(8.27, 11.69))

        grid = fig.add_gridspec(
            5,
            1,
            height_ratios=[1.0, 1.6, 0.8, 2.0, 0.9],
            left=0.15,
            right=0.95,
            top=0.95,
            bottom=0.08,
            hspace=1.05,
        )

        header = fig.add_subplot(grid[0])
        header.axis("off")

        header.text(
            0,
            1,
            "Portfolio Risk Report",
            fontsize=21,
            fontweight="bold",
            va="top",
        )

        settings = (
            f"Portfolio value: EUR {portfolio_value:,.2f}"
            f"    |    Confidence: {confidence_level:.0%}\n"
            f"Risk horizon: 1 trading day"
            f"    |    Monte Carlo scenarios: "
            f"{len(simulated_losses):,}\n"
            f"Historical return observations: "
            f"{len(historical_losses):,}"
        )

        degrees_of_freedom = monte_carlo_results.get(
            "degrees_of_freedom"
        )

        if degrees_of_freedom is not None:
            settings += (
                f"    |    Fitted Student-t degrees of freedom: "
                f"{degrees_of_freedom:.2f}"
            )

        header.text(
            0,
            0.58,
            settings,
            fontsize=9,
            va="top",
            linespacing=1.6,
        )

        # Result table
        table_ax = fig.add_subplot(grid[1])
        table_ax.axis("off")
        table_ax.set_title(
            "Risk estimates",
            loc="left",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

        table = table_ax.table(
            cellText=table_rows,
            colLabels=[
                "Method",
                "Scenario",
                "VaR EUR",
                "VaR %",
                "ES EUR",
                "ES %",
            ],
            colWidths=[
                0.22,
                0.17,
                0.18,
                0.12,
                0.18,
                0.13,
            ],
            cellLoc="center",
            bbox=[0, 0, 1, 0.95],
        )

        style_table(table)

        # Chart 1: Portfolio allocation
        allocation_ax = fig.add_subplot(grid[2])
        allocation_ax.set_title(
            "01 / Portfolio allocation",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )

        left = 0.0
        allocation_colors = plt.get_cmap("tab10")

        for index, (asset, weight) in enumerate(weights.items()):
            allocation_ax.barh(
                0,
                weight * 100,
                left=left,
                color=allocation_colors(index % 10),
                label=f"{asset}: {weight:.1%}",
            )
            left += weight * 100

        allocation_ax.set_xlim(0, 100)
        allocation_ax.set_yticks([])
        allocation_ax.set_xlabel("Portfolio weight (%)")
        allocation_ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.7),
            ncol=min(3, len(weights)),
            fontsize=8,
            frameon=False,
        )

        # Chart 2: Risk estimates
        comparison_ax = fig.add_subplot(grid[3])
        positions = np.arange(len(scenarios))

        comparison_ax.barh(
            positions - 0.18,
            [
                result["var"]
                for _, _, result in scenarios
            ],
            height=0.34,
            color=BLUE,
            label="VaR",
        )

        comparison_ax.barh(
            positions + 0.18,
            [
                result["es"]
                for _, _, result in scenarios
            ],
            height=0.34,
            color=ORANGE,
            label="ES",
        )

        comparison_ax.set_yticks(
            positions,
            [
                "Historical",
                "Monte Carlo",
                "Historical x3",
                "Monte Carlo x3",
            ],
        )

        comparison_ax.invert_yaxis()
        comparison_ax.set_title(
            "02 / Baseline and stress comparison",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        comparison_ax.set_xlabel("Loss in EUR")
        comparison_ax.grid(axis="x", alpha=0.2)
        comparison_ax.legend(fontsize=8)

        notes_ax = fig.add_subplot(grid[4])
        notes_ax.axis("off")
        notes_ax.text(
            0,
            1,
            "VaR is a loss threshold, not a maximum possible loss.\n"
            "ES is the average scenario loss at or above VaR.\n\n"
            "Stress rule: multiply all losses and gains by 3.\n"
            "VaR and ES therefore triple exactly. This is a "
            "sensitivity\n"
            "analysis, not a historical crisis replay.",
            va="top",
            fontsize=9,
            linespacing=1.5,
        )

        fig.text(
            0.15,
            0.035,
            "Educational risk model",
            fontsize=8,
        )
        fig.text(
            0.90,
            0.035,
            f"1 / {total_pages}",
            fontsize=8,
        )

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(8.27, 11.69),
        )

        fig.subplots_adjust(
            left=0.15,
            right=0.95,
            top=0.88,
            bottom=0.22,
            hspace=0.8,
        )

        fig.text(
            0.15,
            0.95,
            "Loss distributions",
            fontsize=21,
            fontweight="bold",
        )

        fig.text(
            0.15,
            0.915,
            "One-day losses in EUR. Negative losses represent gains.",
            fontsize=9,
        )

        # Chart 3: Baseline method comparison
        draw_loss_distribution(
            axes[0],
            historical_losses,
            simulated_losses,
            labels=("Historical", "Monte Carlo"),
            title="03 / Historical vs. Monte Carlo",
            confidence=confidence_level,
        )

        # Chart 4: Monte Carlo stress
        draw_loss_distribution(
            axes[1],
            simulated_losses,
            simulated_stress_losses,
            labels=("Baseline", "Stress x3"),
            title="04 / Monte Carlo: baseline vs. stress",
            confidence=confidence_level,
        )

        # Chart 5: Historical stress
        draw_loss_distribution(
            axes[2],
            historical_losses,
            historical_stress_losses,
            labels=("Baseline", "Stress x3"),
            title="05 / Historical: baseline vs. stress",
            confidence=confidence_level,
        )

        fig.text(
            0.15,
            0.135,
            "Dashed lines mark VaR. Histograms use common bins "
            "within each chart.\n"
            "The x-axis is cropped to the 0.1% to 99.9% range "
            "of the plotted data.\n"
            "Input returns must use matching dates and a "
            "consistent EUR perspective.\n"
            "Constant daily weights imply daily rebalancing. "
            "Student-t simple returns\n"
            "and scaled losses can imply losses above the value "
            "of an unleveraged\n"
            "portfolio. Parameter uncertainty and liquidity risk "
            "are not modelled.",
            fontsize=8,
            linespacing=1.4,
            va="top",
        )

        fig.text(
            0.15,
            0.035,
            "Educational risk model",
            fontsize=8,
        )
        fig.text(
            0.90,
            0.035,
            f"2 / {total_pages}",
            fontsize=8,
        )

        pdf.savefig(fig)
        plt.close(fig)

        # Page 3
        if backtest_results is not None:
            fig = plt.figure(figsize=(8.27, 11.69))

            grid = fig.add_gridspec(
                4,
                1,
                height_ratios=[0.7, 2.0, 3.0, 1.9],
                left=0.15,
                right=0.95,
                top=0.95,
                bottom=0.08,
                hspace=0.5,
            )

            header = fig.add_subplot(grid[0])
            header.axis("off")

            header.text(
                0,
                1,
                "Backtest",
                fontsize=21,
                fontweight="bold",
                va="top",
            )

            header.text(
                0,
                0.45,
                "Rolling test of the historical VaR against the "
                "losses that actually occurred.",
                fontsize=9,
                va="top",
            )

            observations = backtest_results["observations"]
            violation_count = backtest_results[
                "violation_count"
            ]
            expected = backtest_results[
                "expected_violations"
            ]
            p_value = backtest_results["p_value"]

            verdict = (
                "Model rejected at the 5% level"
                if p_value < 0.05
                else "Model not rejected at the 5% level"
            )

            backtest_rows = [
                [
                    "Method",
                    "Historical VaR, rolling window of "
                    f"{backtest_results['window']} days",
                ],
                [
                    "Confidence level",
                    f"{confidence_level:.1%}",
                ],
                [
                    "Backtest observations",
                    f"{observations:,}",
                ],
                [
                    "VaR exceedances (observed)",
                    f"{violation_count}",
                ],
                [
                    "VaR exceedances (expected)",
                    f"{expected:.1f}",
                ],
                [
                    "Exceedance rate (observed / expected)",
                    f"{violation_count / observations:.2%} / "
                    f"{1 - confidence_level:.2%}",
                ],
                [
                    "Kupiec LR statistic",
                    f"{backtest_results['lr_statistic']:.3f}",
                ],
                [
                    "Kupiec p-value",
                    f"{p_value:.3f}",
                ],
                [
                    "Result",
                    verdict,
                ],
            ]

            table_ax = fig.add_subplot(grid[1])
            table_ax.axis("off")

            table = table_ax.table(
                cellText=backtest_rows,
                colLabels=["Metric", "Value"],
                colWidths=[0.45, 0.55],
                cellLoc="left",
                bbox=[0, 0, 1, 1],
            )

            style_table(table)

            chart_ax = fig.add_subplot(grid[2])

            actual_losses = np.asarray(
                backtest_results["actual_losses"],
                dtype=float,
            )

            var_series = np.asarray(
                backtest_results["var_series"],
                dtype=float,
            )

            violations = np.asarray(
                backtest_results["violations"],
                dtype=bool,
            )

            if backtest_results.get("dates") is not None:
                x_values = np.asarray(
                    backtest_results["dates"],
                    dtype=object,
                )
            else:
                x_values = np.arange(len(actual_losses))

            chart_ax.plot(
                x_values,
                actual_losses,
                color=BLUE,
                linewidth=0.7,
                label="Actual one-day loss",
            )

            chart_ax.plot(
                x_values,
                var_series,
                color=DARK,
                linewidth=1.3,
                label=(
                    f"VaR ({confidence_level:.1%}, rolling)"
                ),
            )

            chart_ax.scatter(
                x_values[violations],
                actual_losses[violations],
                color=ORANGE,
                s=22,
                zorder=3,
                label=f"Exceedances ({violation_count})",
            )

            chart_ax.set_title(
                "06 / Losses against rolling VaR",
                loc="left",
                fontsize=11,
                fontweight="bold",
            )

            chart_ax.set_ylabel("One-day loss (EUR)")
            chart_ax.grid(axis="y", alpha=0.2)

            # Leave space above the data for the legend.
            y_min, y_max = chart_ax.get_ylim()

            chart_ax.set_ylim(
                y_min,
                y_max + 0.3 * (y_max - y_min),
            )

            chart_ax.legend(
                fontsize=8,
                loc="upper left",
            )

            notes_ax = fig.add_subplot(grid[3])
            notes_ax.axis("off")

            notes_ax.text(
                0,
                1,
                "A VaR exceedance occurs when the actual loss is "
                "greater than the estimated VaR.\n"
                "The Kupiec test checks whether the observed "
                "exceedance rate matches the expected rate.\n"
                "A p-value below 0.05 indicates that the model "
                "is rejected.",
                va="top",
                fontsize=8.5,
                linespacing=1.5,
            )

            fig.text(
                0.15,
                0.035,
                "Educational risk model",
                fontsize=8,
            )

            fig.text(
                0.90,
                0.035,
                f"3 / {total_pages}",
                fontsize=8,
            )

            pdf.savefig(fig)
            plt.close(fig)

    return output_path.resolve()
