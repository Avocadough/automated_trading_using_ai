# src/eval/academic_report.py
"""
Academic-Grade Evaluation Report for Senior AI Project Thesis.

WHY COMPARE AGAINST BUY & HOLD?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Buy & Hold is the "null hypothesis" in quantitative finance.
Any trading strategy that does NOT beat Buy & Hold (after fees
and slippage) has zero value — the investor could simply hold
the asset with zero effort.

This is scientifically sound because:
1. It controls for market beta — if BTC goes up 100% and our agent
   makes 80%, we've actually LOST relative to doing nothing.
2. It includes ALL costs — the Buy & Hold baseline pays only 1 entry
   fee, while the agent pays fees on every trade.
3. It is the fairest benchmark for an active trading thesis: "Does
   the agent generate value beyond passive exposure?"

CASHFLOW THESIS:
The core academic argument is that the RL agent generates ACTIVE
monthly cashflow (realized profits) without needing to hold indefinitely,
unlike Buy & Hold which only realizes profit upon final liquidation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Metric Computations
# ============================================================

def compute_metrics(
    equity: np.ndarray,
    benchmark_equity: np.ndarray,
    trades: List[Dict],
    periods_per_year: int = 8760,
    initial_balance: float = 10_000.0,
) -> Dict:
    """Compute all academic-standard metrics."""
    n = len(equity)

    # Returns
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    bm_returns = np.diff(benchmark_equity) / benchmark_equity[:-1]
    bm_returns = np.nan_to_num(bm_returns, nan=0.0, posinf=0.0, neginf=0.0)

    # Basic
    total_return = (equity[-1] / equity[0]) - 1.0 if n > 1 else 0.0
    bm_total_return = (
        (benchmark_equity[-1] / benchmark_equity[0]) - 1.0
        if len(benchmark_equity) > 1 else 0.0
    )
    years = n / periods_per_year
    ann_return = (1 + total_return) ** (1 / max(years, 1e-6)) - 1 if total_return > -1 else 0.0
    ann_vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year) if len(returns) > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else 0.0
    sortino = ann_return / downside_std if downside_std > 1e-12 else 0.0

    # Drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    # Win/Loss from trades
    if trades:
        trade_rets = [t.get("return", 0) for t in trades]
        wins = [r for r in trade_rets if r > 1e-12]
        losses = [r for r in trade_rets if r < -1e-12]
        win_rate = len(wins) / max(len(wins) + len(losses), 1)
        gross_wins = sum(wins) if wins else 0
        gross_losses = abs(sum(losses)) if losses else 0
        profit_factor = gross_wins / max(gross_losses, 1e-12)
    else:
        win_rate = 0.0
        profit_factor = 0.0

    # Cashflow: average realized profit per month
    n_months = max(years * 12, 1)
    total_realized = sum(
        t.get("exit_equity", 0) - t.get("entry_equity", 0)
        for t in trades
        if t.get("return", 0) > 0
    )
    avg_monthly_cashflow = total_realized / n_months

    # Alpha (outperformance vs benchmark)
    alpha = total_return - bm_total_return

    return {
        "total_return": total_return,
        "bm_total_return": bm_total_return,
        "alpha": alpha,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": len(trades),
        "avg_monthly_cashflow": avg_monthly_cashflow,
        "eval_bars": n,
        "eval_years": years,
    }


def compute_train_sharpe(
    train_equity: np.ndarray,
    periods_per_year: int = 8760,
) -> float:
    """Compute Sharpe on training data for overfitting check."""
    if len(train_equity) < 2:
        return 0.0
    returns = np.diff(train_equity) / train_equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    years = len(returns) / periods_per_year
    total_ret = (train_equity[-1] / train_equity[0]) - 1.0
    ann_ret = (1 + total_ret) ** (1 / max(years, 1e-6)) - 1 if total_ret > -1 else 0.0
    ann_vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year) if len(returns) > 1 else 0.0
    return ann_ret / ann_vol if ann_vol > 1e-12 else 0.0


# ============================================================
# Visualization — Thesis-Presentation Ready
# ============================================================

class AcademicReport:
    """Generate clean, white-themed charts for thesis defense."""

    def __init__(self, strategy_name: str = "CNN+LSTM PPO Agent"):
        self.strategy_name = strategy_name

    def generate(
        self,
        equity: np.ndarray,
        benchmark_equity: np.ndarray,
        returns: np.ndarray,
        trades: List[Dict],
        metrics: Dict,
        train_sharpe: Optional[float] = None,
        periods_per_year: int = 8760,
        output_dir: Path = Path("data/eval/reports"),
    ):
        """Generate all academic plots and save to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("default")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 150,
        })

        # 1. Cumulative Returns Comparison
        self._plot_cumulative_returns(
            equity, benchmark_equity, metrics,
            output_dir / "cumulative_returns.png"
        )

        # 2. Underwater (Drawdown) Plot
        self._plot_underwater(
            returns,
            output_dir / "drawdown.png"
        )

        # 3. Monthly Returns Heatmap
        self._plot_monthly_heatmap(
            returns, periods_per_year,
            output_dir / "monthly_heatmap.png"
        )

        # 4. Full Summary Page (combined PDF)
        self._plot_full_report(
            equity, benchmark_equity, returns, trades,
            metrics, train_sharpe, periods_per_year,
            output_dir / "academic_report.pdf"
        )

        # 5. Console output
        self._print_console_report(metrics, train_sharpe)

    # ----------------------------------------------------------------
    # Plot 1: Cumulative Returns — Agent vs Buy & Hold
    # ----------------------------------------------------------------
    def _plot_cumulative_returns(self, equity, benchmark, metrics, path):
        fig, ax = plt.subplots(figsize=(12, 5))

        n = min(len(equity), len(benchmark))
        eq = equity[:n]
        bm = benchmark[:n]

        # Normalize to percentage return
        eq_pct = (eq / eq[0] - 1) * 100
        bm_pct = (bm / bm[0] - 1) * 100

        ax.plot(eq_pct, color="#1a73e8", linewidth=1.8,
                label=f'{self.strategy_name} ({metrics["total_return"]*100:+.1f}%)')
        ax.plot(bm_pct, color="#ea4335", linewidth=1.2, linestyle="--",
                label=f'Buy & Hold BTC ({metrics["bm_total_return"]*100:+.1f}%)')

        ax.fill_between(range(n), eq_pct, bm_pct,
                        where=eq_pct > bm_pct,
                        color="#34a853", alpha=0.15, label="Outperformance (Alpha)")
        ax.fill_between(range(n), eq_pct, bm_pct,
                        where=eq_pct < bm_pct,
                        color="#ea4335", alpha=0.10)

        ax.set_title("Cumulative Returns: Agent vs. Buy & Hold",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Time Steps (1H bars)")
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend(loc="upper left", fontsize=9)
        ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {path}")

    # ----------------------------------------------------------------
    # Plot 2: Underwater (Drawdown) Plot
    # ----------------------------------------------------------------
    def _plot_underwater(self, returns, path):
        fig, ax = plt.subplots(figsize=(12, 3.5))

        equity = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(equity)
        dd = ((equity - peak) / peak) * 100

        ax.fill_between(range(len(dd)), 0, dd,
                        color="#ea4335", alpha=0.4)
        ax.plot(dd, color="#ea4335", linewidth=0.6)
        ax.set_title("Drawdown Over Time (Underwater Plot)",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("Time Steps (1H bars)")
        ax.set_ylim(min(dd) * 1.15, 0.5)

        fig.tight_layout()
        fig.savefig(str(path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {path}")

    # ----------------------------------------------------------------
    # Plot 3: Monthly Returns Heatmap
    # ----------------------------------------------------------------
    def _plot_monthly_heatmap(self, returns, ppy, path):
        fig, ax = plt.subplots(figsize=(12, 4))

        hours_per_month = ppy / 12
        n_months = int(len(returns) / max(hours_per_month, 1))

        if n_months < 2:
            ax.text(0.5, 0.5, "Insufficient data for monthly heatmap",
                    transform=ax.transAxes, ha="center", va="center")
            fig.savefig(str(path), dpi=200, bbox_inches="tight")
            plt.close(fig)
            return

        monthly = []
        for i in range(n_months):
            s = int(i * hours_per_month)
            e = int(min((i + 1) * hours_per_month, len(returns)))
            if e > s:
                monthly.append(float(np.prod(1 + returns[s:e]) - 1) * 100)
            else:
                monthly.append(0.0)

        n_rows = max(1, (len(monthly) + 11) // 12)
        padded = monthly + [np.nan] * (n_rows * 12 - len(monthly))
        grid = np.array(padded).reshape(n_rows, 12)

        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto",
                       vmin=-8, vmax=8)
        plt.colorbar(im, ax=ax, label="Return (%)", shrink=0.8)

        for i in range(n_rows):
            for j in range(12):
                v = grid[i, j]
                if not np.isnan(v):
                    c = "white" if abs(v) > 4 else "black"
                    ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=c)

        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax.set_xticks(range(12))
        ax.set_xticklabels(months, fontsize=9)
        ax.set_title("Monthly Returns Heatmap (Consistency Check)",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Year")

        fig.tight_layout()
        fig.savefig(str(path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {path}")

    # ----------------------------------------------------------------
    # Full Combined Report (Multi-page PDF)
    # ----------------------------------------------------------------
    def _plot_full_report(self, equity, benchmark, returns, trades,
                          metrics, train_sharpe, ppy, path):
        with PdfPages(str(path)) as pdf:
            # --- Page 1: Charts ---
            fig = plt.figure(figsize=(14, 18))
            gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.40)

            # Title
            fig.suptitle(
                f"Evaluation Report: {self.strategy_name}\n"
                f"Senior AI Project — Out-of-Sample Results",
                fontsize=16, fontweight="bold", y=0.98
            )

            # Cumulative returns
            ax1 = fig.add_subplot(gs[0])
            n = min(len(equity), len(benchmark))
            eq_pct = (equity[:n] / equity[0] - 1) * 100
            bm_pct = (benchmark[:n] / benchmark[0] - 1) * 100
            ax1.plot(eq_pct, color="#1a73e8", lw=1.8,
                     label=f'Agent ({metrics["total_return"]*100:+.1f}%)')
            ax1.plot(bm_pct, color="#ea4335", lw=1.2, ls="--",
                     label=f'Buy & Hold ({metrics["bm_total_return"]*100:+.1f}%)')
            ax1.fill_between(range(n), eq_pct, bm_pct,
                             where=eq_pct > bm_pct,
                             color="#34a853", alpha=0.12)
            ax1.set_title("Cumulative Returns: Agent vs Buy & Hold",
                         fontweight="bold")
            ax1.set_ylabel("Return (%)")
            ax1.legend(fontsize=9)
            ax1.axhline(y=0, color="black", lw=0.5, alpha=0.3)

            # Drawdown
            ax2 = fig.add_subplot(gs[1])
            eq_curve = np.cumprod(1 + returns)
            pk = np.maximum.accumulate(eq_curve)
            dd = ((eq_curve - pk) / pk) * 100
            ax2.fill_between(range(len(dd)), 0, dd, color="#ea4335", alpha=0.4)
            ax2.plot(dd, color="#ea4335", lw=0.6)
            ax2.set_title("Drawdown (Underwater Plot)", fontweight="bold")
            ax2.set_ylabel("Drawdown (%)")
            ax2.set_ylim(min(dd) * 1.15, 0.5)

            # Monthly heatmap
            ax3 = fig.add_subplot(gs[2])
            hours_per_month = ppy / 12
            n_months = int(len(returns) / max(hours_per_month, 1))
            if n_months >= 2:
                monthly = []
                for i in range(n_months):
                    s = int(i * hours_per_month)
                    e = int(min((i + 1) * hours_per_month, len(returns)))
                    monthly.append(float(np.prod(1 + returns[s:e]) - 1) * 100 if e > s else 0.0)
                n_rows = max(1, (len(monthly) + 11) // 12)
                padded = monthly + [np.nan] * (n_rows * 12 - len(monthly))
                grid = np.array(padded).reshape(n_rows, 12)
                im = ax3.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=-8, vmax=8)
                plt.colorbar(im, ax=ax3, label="Return %", shrink=0.7)
                for i in range(n_rows):
                    for j in range(12):
                        v = grid[i, j]
                        if not np.isnan(v):
                            ax3.text(j, i, f"{v:.1f}%", ha="center", va="center",
                                     fontsize=7, fontweight="bold",
                                     color="white" if abs(v) > 4 else "black")
                months_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec"]
                ax3.set_xticks(range(12))
                ax3.set_xticklabels(months_labels, fontsize=8)
                ax3.set_title("Monthly Returns Heatmap", fontweight="bold")
            else:
                ax3.text(0.5, 0.5, "Insufficient data", ha="center",
                         va="center", transform=ax3.transAxes)

            # Metrics table
            ax4 = fig.add_subplot(gs[3])
            ax4.axis("off")
            overfitting = ""
            if train_sharpe is not None:
                degradation = (
                    (1 - metrics["sharpe_ratio"] / train_sharpe) * 100
                    if abs(train_sharpe) > 1e-6 else 0
                )
                overfitting = f"  |  Train→Test Degradation: {degradation:.0f}%"

            table_data = [
                ["Total Return (Agent)", f'{metrics["total_return"]*100:+.2f}%'],
                ["Total Return (Buy & Hold)", f'{metrics["bm_total_return"]*100:+.2f}%'],
                ["Alpha (Outperformance)", f'{metrics["alpha"]*100:+.2f}%'],
                ["Annualized Return", f'{metrics["annualized_return"]*100:.2f}%'],
                ["Annualized Volatility", f'{metrics["annualized_volatility"]*100:.2f}%'],
                ["Sharpe Ratio (OOS)", f'{metrics["sharpe_ratio"]:.3f}'],
                ["Sortino Ratio", f'{metrics["sortino_ratio"]:.3f}'],
                ["Max Drawdown", f'{metrics["max_drawdown"]*100:.2f}%'],
                ["Win Rate", f'{metrics["win_rate"]*100:.1f}%'],
                ["Profit Factor", f'{metrics["profit_factor"]:.3f}'],
                ["Total Trades", f'{metrics["n_trades"]}'],
                ["Avg Monthly Cashflow", f'${metrics["avg_monthly_cashflow"]:.2f}'],
            ]
            if train_sharpe is not None:
                table_data.append(["Sharpe (Train)", f"{train_sharpe:.3f}"])
                table_data.append(["Sharpe (Test/OOS)", f'{metrics["sharpe_ratio"]:.3f}'])

            tbl = ax4.table(cellText=table_data,
                           colLabels=["Metric", "Value"],
                           cellLoc="left", loc="center",
                           colWidths=[0.5, 0.3])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.4)
            for (row, col), cell in tbl.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#1a73e8")
                    cell.set_text_props(color="white", fontweight="bold")
                else:
                    cell.set_facecolor("#f8f9fa" if row % 2 == 0 else "white")
            ax4.set_title("Performance Summary" + overfitting,
                         fontweight="bold", fontsize=12)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"  [saved] {path}")

    # ----------------------------------------------------------------
    # Console Report
    # ----------------------------------------------------------------
    def _print_console_report(self, metrics, train_sharpe):
        print(f"\n{'='*60}")
        print(f" ACADEMIC EVALUATION REPORT — Out-Of-Sample")
        print(f"{'='*60}")
        print(f"  Agent Total Return  : {metrics['total_return']*100:+.2f}%")
        print(f"  Buy&Hold Return     : {metrics['bm_total_return']*100:+.2f}%")
        print(f"  Alpha               : {metrics['alpha']*100:+.2f}%")
        print(f"  —")
        print(f"  Annualized Return   : {metrics['annualized_return']*100:.2f}%")
        print(f"  Annualized Vol      : {metrics['annualized_volatility']*100:.2f}%")
        print(f"  Sharpe Ratio (OOS)  : {metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio       : {metrics['sortino_ratio']:.3f}")
        print(f"  Max Drawdown        : {metrics['max_drawdown']*100:.2f}%")
        print(f"  —")
        print(f"  Win Rate            : {metrics['win_rate']*100:.1f}%")
        print(f"  Profit Factor       : {metrics['profit_factor']:.3f}")
        print(f"  Total Trades        : {metrics['n_trades']}")
        print(f"  Avg Monthly Cashflow: ${metrics['avg_monthly_cashflow']:.2f}")

        if train_sharpe is not None:
            print(f"  —")
            print(f"  Sharpe (Train)      : {train_sharpe:.3f}")
            print(f"  Sharpe (Test/OOS)   : {metrics['sharpe_ratio']:.3f}")
            if abs(train_sharpe) > 1e-6:
                deg = (1 - metrics["sharpe_ratio"] / train_sharpe) * 100
                verdict = "✅ OK" if deg < 50 else "⚠️ Possible Overfit"
                print(f"  Degradation         : {deg:.0f}% {verdict}")
            else:
                print(f"  Degradation         : N/A (train Sharpe ≈ 0)")

        winner = "AGENT ✅" if metrics["alpha"] > 0 else "BUY & HOLD ❌"
        print(f"\n  >>> Verdict: {winner}")
        print(f"{'='*60}\n")
