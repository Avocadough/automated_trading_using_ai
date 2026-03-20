# src/eval/institutional_tearsheet.py
"""
Institutional-Grade Quantitative Tearsheet & Statistical Validation Suite.

Mathematical Foundation (CIO / PhD Financial Mathematics Perspective):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. WHY CVaR OVER VaR?
   VaR answers: "What's the worst loss at confidence level α?"
   CVaR answers: "Given we exceeded VaR, what is the EXPECTED loss?"
   For crypto (fat-tailed, leptokurtic distributions), VaR understates
   tail risk catastrophically. CVaR captures the entire tail shape.
   Basel III/IV mandates Expected Shortfall (CVaR) for regulatory capital.

2. WHY MONTE CARLO PERMUTATION TEST?
   A positive Sharpe could be:
   (a) Genuine alpha — the strategy exploits a persistent market inefficiency
   (b) Data snooping — random chance over the backtest period
   (c) Beta masquerading as alpha — strategy just follows BTC trends
   The permutation test shuffles trade returns 10,000 times, recomputing
   Sharpe each time. If observed Sharpe ranks in the top 1%, we reject
   H₀ (returns = random) at p < 0.01. This is the gold standard for
   academic alpha validation (Harvey, Liu & Zhu, 2016).

3. WHY OMEGA RATIO?
   Sharpe assumes returns are normally distributed. Crypto returns have
   significant skewness and excess kurtosis (fat tails). Omega captures
   the ENTIRE distribution: Ω(θ) = ∫[θ,∞] (1-F(r))dr / ∫[-∞,θ] F(r)dr
   Omega > 1 at θ=0 means gains outweigh losses probabilistically.

4. STRATEGY BETA DECOMPOSITION
   We regress: R_strategy = α + β·R_benchmark + ε
   - α (intercept) = pure alpha — must be positive and statistically
     significant (t-stat > 2.0)
   - β (slope) = benchmark exposure — we want low β to prove the strategy
     generates idiosyncratic returns, not leveraged BTC exposure
   - R² = how much of strategy variance is explained by BTC — low R² = good
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")     # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker


# ============================================================
# 1. RISK ANALYTICS ENGINE
# ============================================================

class RiskAnalytics:
    """Compute institutional-grade risk metrics from returns series."""

    def __init__(self, returns: np.ndarray, periods_per_year: int = 8760,
                 risk_free_rate: float = 0.0):
        self.returns = np.asarray(returns, dtype=np.float64)
        self.ppy = periods_per_year
        self.rf = risk_free_rate
        self.n = len(self.returns)

    # ---- Basic Stats ----
    def total_return(self) -> float:
        return float(np.prod(1 + self.returns) - 1)

    def annualized_return(self) -> float:
        total = self.total_return()
        years = self.n / self.ppy
        if years <= 0 or total <= -1:
            return 0.0
        return float((1 + total) ** (1 / years) - 1)

    def annualized_volatility(self) -> float:
        return float(np.std(self.returns, ddof=1) * np.sqrt(self.ppy))

    # ---- Sharpe Ratio (annualized) ----
    def sharpe_ratio(self) -> float:
        vol = self.annualized_volatility()
        if vol < 1e-12:
            return 0.0
        return float((self.annualized_return() - self.rf) / vol)

    # ---- Sortino Ratio ----
    def sortino_ratio(self) -> float:
        excess = self.returns - self.rf / self.ppy
        downside = excess[excess < 0]
        if len(downside) < 2:
            return 0.0
        downside_std = np.std(downside, ddof=1) * np.sqrt(self.ppy)
        if downside_std < 1e-12:
            return 0.0
        return float((self.annualized_return() - self.rf) / downside_std)

    # ---- Drawdown Analytics ----
    def _drawdown_series(self) -> np.ndarray:
        equity = np.cumprod(1 + self.returns)
        peak = np.maximum.accumulate(equity)
        return (equity - peak) / peak

    def max_drawdown(self) -> float:
        dd = self._drawdown_series()
        return float(np.min(dd))

    def calmar_ratio(self) -> float:
        mdd = abs(self.max_drawdown())
        if mdd < 1e-12:
            return 0.0
        return float(self.annualized_return() / mdd)

    # ---- VaR & CVaR (Expected Shortfall) ----
    def var(self, alpha: float = 0.05) -> float:
        """Historical VaR at confidence level (1 - alpha). Negative = loss."""
        return float(np.percentile(self.returns, alpha * 100))

    def cvar(self, alpha: float = 0.05) -> float:
        """
        Conditional VaR (Expected Shortfall).
        Average of returns BELOW the VaR threshold.
        This is what Basel III/IV mandates for tail risk measurement.
        """
        var_threshold = self.var(alpha)
        tail_returns = self.returns[self.returns <= var_threshold]
        if len(tail_returns) == 0:
            return var_threshold
        return float(np.mean(tail_returns))

    # ---- Distribution Moments ----
    def skewness(self) -> float:
        return float(sp_stats.skew(self.returns))

    def kurtosis(self) -> float:
        """Excess kurtosis (normal = 0). Crypto typically has kurtosis > 3."""
        return float(sp_stats.kurtosis(self.returns))

    # ---- Omega Ratio ----
    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Ω(θ) = E[max(R-θ, 0)] / E[max(θ-R, 0)]
        Omega > 1 at θ=0 means gains outweigh losses.
        """
        excess = self.returns - threshold
        gains = np.sum(excess[excess > 0])
        losses = np.abs(np.sum(excess[excess < 0]))
        if losses < 1e-12:
            return float("inf")
        return float(gains / losses)

    # ---- Win/Loss Stats ----
    def win_rate(self) -> float:
        wins = np.sum(self.returns > 0)
        total = np.sum(self.returns != 0)
        return float(wins / max(total, 1))

    def profit_factor(self) -> float:
        gains = np.sum(self.returns[self.returns > 0])
        losses = abs(np.sum(self.returns[self.returns < 0]))
        if losses < 1e-12:
            return float("inf")
        return float(gains / losses)

    # ---- Rolling Sharpe ----
    def rolling_sharpe(self, window: int = 0) -> pd.Series:
        """Rolling annualized Sharpe ratio."""
        if window <= 0:
            window = min(self.ppy // 2, self.n // 4)  # ~6 months
            window = max(window, 100)
        s = pd.Series(self.returns)
        roll_mean = s.rolling(window).mean()
        roll_std = s.rolling(window).std()
        return (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(self.ppy)

    def to_dict(self) -> Dict:
        return {
            "Total Return": f"{self.total_return() * 100:.2f}%",
            "Annualized Return": f"{self.annualized_return() * 100:.2f}%",
            "Annualized Volatility": f"{self.annualized_volatility() * 100:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio():.3f}",
            "Sortino Ratio": f"{self.sortino_ratio():.3f}",
            "Calmar Ratio": f"{self.calmar_ratio():.3f}",
            "Omega Ratio (θ=0)": f"{self.omega_ratio():.3f}",
            "Max Drawdown": f"{self.max_drawdown() * 100:.2f}%",
            "VaR (95%)": f"{self.var(0.05) * 100:.4f}%",
            "VaR (99%)": f"{self.var(0.01) * 100:.4f}%",
            "CVaR/ES (95%)": f"{self.cvar(0.05) * 100:.4f}%",
            "CVaR/ES (99%)": f"{self.cvar(0.01) * 100:.4f}%",
            "Skewness": f"{self.skewness():.3f}",
            "Excess Kurtosis": f"{self.kurtosis():.3f}",
            "Win Rate": f"{self.win_rate() * 100:.1f}%",
            "Profit Factor": f"{self.profit_factor():.3f}",
        }


# ============================================================
# 2. STATISTICAL VALIDATION
# ============================================================

class StatisticalValidator:
    """Statistical tests for alpha significance."""

    @staticmethod
    def bootstrap_sharpe_pvalue(returns: np.ndarray, observed_sharpe: float,
                                 n_simulations: int = 10_000,
                                 periods_per_year: int = 8760,
                                 seed: int = 42) -> Tuple[float, np.ndarray]:
        """
        Monte Carlo Bootstrap Test for Sharpe Ratio significance.

        H₀: The observed Sharpe ratio is due to random chance.
        Method: Shuffle returns (destroy temporal structure), compute Sharpe
                for each permutation. P-value = fraction of permuted Sharpes
                that equal or exceed the observed Sharpe.

        A p-value < 0.05 means: "There is < 5% probability this Sharpe
        arose from random luck."
        """
        rng = np.random.RandomState(seed)
        permuted_sharpes = np.zeros(n_simulations)

        # ศูนย์กลางผลตอบแทนให้กลายเป็น 0 (Null Hypothesis) เพื่อสุ่มใหม่
        centered_returns = returns - np.mean(returns)

        for i in range(n_simulations):
            # ใช้ Bootstrap (สุ่มใส่คืน) ไม่ใช่แค่การเรียงลำดับใหม่
            shuffled = rng.choice(centered_returns, size=len(returns), replace=True)
            mean_r = np.mean(shuffled)
            std_r = np.std(shuffled, ddof=1)
            if std_r > 1e-12:
                permuted_sharpes[i] = (mean_r / std_r) * np.sqrt(periods_per_year)
            else:
                permuted_sharpes[i] = 0.0

        p_value = float(np.mean(permuted_sharpes >= observed_sharpe))
        return p_value, permuted_sharpes

    @staticmethod
    def alpha_beta_decomposition(strategy_returns: np.ndarray,
                                  benchmark_returns: np.ndarray
                                  ) -> Dict[str, float]:
        """
        OLS Regression: R_strategy = α + β·R_benchmark + ε

        Returns:
          alpha:   annualized intercept (pure alpha)
          beta:    slope (benchmark exposure)
          r_squared: fraction of variance explained by benchmark
          alpha_tstat: t-statistic for alpha (> 2.0 = significant at 95%)
          alpha_pvalue: p-value for alpha significance
        """
        n = min(len(strategy_returns), len(benchmark_returns))
        y = strategy_returns[:n]
        x = benchmark_returns[:n]

        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)

        # t-stat for intercept (alpha)
        x_mean = np.mean(x)
        ss_xx = np.sum((x - x_mean) ** 2)
        y_hat = intercept + slope * x
        residuals = y - y_hat
        mse = np.sum(residuals ** 2) / max(n - 2, 1)
        se_intercept = np.sqrt(mse * (1.0 / n + x_mean ** 2 / max(ss_xx, 1e-12)))
        alpha_tstat = intercept / max(se_intercept, 1e-12)
        alpha_pval = 2 * (1 - sp_stats.t.cdf(abs(alpha_tstat), df=max(n - 2, 1)))

        return {
            "alpha_per_period": float(intercept),
            "beta": float(slope),
            "r_squared": float(r_value ** 2),
            "alpha_tstat": float(alpha_tstat),
            "alpha_pvalue": float(alpha_pval),
        }

    @staticmethod
    def jarque_bera_test(returns: np.ndarray) -> Tuple[float, float]:
        """Test if returns are normally distributed. High JB = non-normal."""
        jb_stat, jb_pval = sp_stats.jarque_bera(returns)
        return float(jb_stat), float(jb_pval)


# ============================================================
# 3. CAPACITY & TCA ESTIMATION
# ============================================================

class CapacityEstimator:
    """Estimate strategy capacity based on trade volume and market impact."""

    @staticmethod
    def estimate_capacity(
        equity_curve: np.ndarray,
        n_trades: int,
        avg_trade_notional: float,
        daily_volume_usd: float = 30_000_000_000.0,  # BTC ~$30B/day
        impact_coefficient: float = 0.1,   # sqrt market impact model
        current_sharpe: float = 1.0,
    ) -> Dict[str, float]:
        """
        Square-root market impact model (Almgren & Chriss, 2001):
          Impact = η · σ · √(V_trade / V_market)

        At what AUM does Sharpe decay to break-even (Sharpe ~ 0)?
        """
        if n_trades < 1 or daily_volume_usd < 1e-6:
            return {"capacity_usd": 0.0, "impact_bps_current": 0.0}

        # Current participation rate
        participation = avg_trade_notional / daily_volume_usd
        impact_bps = impact_coefficient * np.sqrt(participation) * 10_000

        # At what multiple does impact eat all alpha?
        # Sharpe degrades linearly with √(AUM):
        # Sharpe(AUM) ≈ Sharpe_observed - k * √(AUM / AUM_current)
        # Capacity = AUM where Sharpe → 0
        if current_sharpe > 0 and impact_bps > 0:
            initial_aum = equity_curve[-1] if len(equity_curve) > 0 else 10_000
            # Scale factor: how many multiples of current AUM before alpha → 0
            scale_factor = (current_sharpe / max(impact_bps * 1e-4, 1e-12)) ** 2
            capacity_usd = initial_aum * scale_factor
        else:
            capacity_usd = 0.0

        return {
            "capacity_usd": float(capacity_usd),
            "impact_bps_current": float(impact_bps),
            "participation_rate": float(participation),
        }


# ============================================================
# 4. PROFESSIONAL TEARSHEET GENERATOR
# ============================================================

class InstitutionalTearsheet:
    """Generate multi-page PDF institutional tearsheet."""

    # color scheme
    COLORS = {
        "primary": "#1a73e8",
        "secondary": "#34a853",
        "danger": "#ea4335",
        "warning": "#fbbc04",
        "bg": "#0d1117",
        "fg": "#c9d1d9",
        "grid": "#21262d",
        "accent1": "#58a6ff",
        "accent2": "#3fb950",
    }

    def __init__(self, strategy_name: str = "CNN+LSTM PPO Alpha Strategy"):
        self.strategy_name = strategy_name

    def generate(
        self,
        equity: np.ndarray,
        returns: np.ndarray,
        benchmark_equity: np.ndarray,
        benchmark_returns: np.ndarray,
        actions: np.ndarray,
        trades: List[Dict],
        risk: RiskAnalytics,
        stat_val: Dict,
        ab_decomp: Dict,
        capacity: Dict,
        output_path: Path,
        periods_per_year: int = 8760,
        sp500_equity: Optional[np.ndarray] = None,
        bm_risk: Optional[RiskAnalytics] = None,
        sp_risk: Optional[RiskAnalytics] = None,
    ):
        """Generate the complete multi-page PDF tearsheet."""
        plt.style.use("dark_background")
        plt.rcParams.update({
            "font.family": "monospace",
            "font.size": 9,
            "axes.facecolor": self.COLORS["bg"],
            "figure.facecolor": self.COLORS["bg"],
            "axes.edgecolor": self.COLORS["grid"],
            "axes.grid": True,
            "grid.color": self.COLORS["grid"],
            "grid.alpha": 0.3,
        })

        with PdfPages(str(output_path)) as pdf:
            # ---- PAGE 1: Executive Summary + Equity Curve ----
            fig = plt.figure(figsize=(16, 22))
            gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)

            self._plot_header(fig, risk, ab_decomp, stat_val, capacity)
            self._plot_equity_curve(fig.add_subplot(gs[1, :]), equity,
                                    benchmark_equity, sp500_equity)
            self._plot_underwater(fig.add_subplot(gs[2, :]), returns)
            self._plot_rolling_sharpe(fig.add_subplot(gs[3, :]), returns,
                                      periods_per_year)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ---- PAGE 2: Distribution + Statistical Tests ----
            fig = plt.figure(figsize=(16, 22))
            gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)

            self._plot_returns_distribution(fig.add_subplot(gs[0, 0]), returns)
            self._plot_qq_plot(fig.add_subplot(gs[0, 1]), returns)
            self._plot_monthly_heatmap(fig.add_subplot(gs[1, :]), returns,
                                       periods_per_year)
            self._plot_permutation_test(fig.add_subplot(gs[2, :]),
                                        stat_val, risk.sharpe_ratio())
            self._plot_action_distribution(fig.add_subplot(gs[3, 0]), actions)
            self._plot_trade_analysis(fig.add_subplot(gs[3, 1]), trades)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ---- PAGE 3: Risk Metrics Table ----
            fig = plt.figure(figsize=(16, 12))
            self._plot_metrics_table(fig, risk, ab_decomp, stat_val, capacity,
                                     bm_risk=bm_risk,
                                     sp_risk=sp_risk)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"[ok] Institutional tearsheet saved: {output_path}")

    # ================================================================
    # Individual Plot Methods
    # ================================================================

    def _plot_header(self, fig, risk, ab_decomp, stat_val, capacity):
        """Title block with key metrics."""
        fig.suptitle(
            f"INSTITUTIONAL QUANTITATIVE TEARSHEET\n{self.strategy_name}",
            fontsize=14, fontweight="bold", color=self.COLORS["accent1"],
            y=0.98
        )

        # Key metrics bar
        metrics_text = (
            f"Sharpe: {risk.sharpe_ratio():.2f}  |  "
            f"Sortino: {risk.sortino_ratio():.2f}  |  "
            f"Calmar: {risk.calmar_ratio():.2f}  |  "
            f"Max DD: {risk.max_drawdown()*100:.1f}%  |  "
            f"Alpha p-value: {stat_val.get('p_value', 'N/A')}  |  "
            f"Beta: {ab_decomp.get('beta', 0):.3f}"
        )
        fig.text(0.5, 0.94, metrics_text, ha="center", fontsize=10,
                 color=self.COLORS["fg"], fontfamily="monospace")

    def _plot_equity_curve(self, ax, equity, benchmark_equity,
                           sp500_equity=None):
        """Log-scale equity curve vs benchmark (Buy & Hold) + S&P 500."""
        n = min(len(equity), len(benchmark_equity))
        eq = equity[:n]
        bm = benchmark_equity[:n]

        ax.semilogy(eq, color=self.COLORS["accent1"], linewidth=1.5,
                    label="Strategy", alpha=0.9)
        ax.semilogy(bm, color=self.COLORS["danger"], linewidth=1.0,
                    label="Buy & Hold BTC", alpha=0.7)

        # S&P 500 third line
        if sp500_equity is not None and len(sp500_equity) > 1:
            sp_n = min(n, len(sp500_equity))
            ax.semilogy(sp500_equity[:sp_n], color=self.COLORS["warning"],
                        linewidth=1.0, linestyle=":",
                        label="S&P 500", alpha=0.8)

        ax.fill_between(range(n), eq, bm,
                        where=eq > bm,
                        color=self.COLORS["accent2"], alpha=0.1,
                        label="Outperformance")
        ax.fill_between(range(n), eq, bm,
                        where=eq < bm,
                        color=self.COLORS["danger"], alpha=0.1)

        title_suffix = " & S&P 500" if sp500_equity is not None else ""
        ax.set_title(f"Equity Curve (Log Scale) vs. Benchmark{title_suffix}",
                     color=self.COLORS["fg"], fontweight="bold")
        ax.set_ylabel("Equity ($)", color=self.COLORS["fg"])
        ax.legend(loc="upper left", fontsize=8)
        ax.tick_params(colors=self.COLORS["fg"])

    def _plot_underwater(self, ax, returns):
        """Underwater plot (drawdown chart)."""
        equity = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(equity)
        dd = ((equity - peak) / peak) * 100

        ax.fill_between(range(len(dd)), 0, dd,
                        color=self.COLORS["danger"], alpha=0.4)
        ax.plot(dd, color=self.COLORS["danger"], linewidth=0.5, alpha=0.8)

        ax.set_title("Underwater Plot (Drawdown %)",
                     color=self.COLORS["fg"], fontweight="bold")
        ax.set_ylabel("Drawdown %", color=self.COLORS["fg"])
        ax.set_ylim(min(dd) * 1.1, 1)
        ax.tick_params(colors=self.COLORS["fg"])

    def _plot_rolling_sharpe(self, ax, returns, ppy):
        """Rolling Sharpe to prove temporal stability."""
        window = min(ppy // 2, len(returns) // 4)
        window = max(window, 100)

        r = RiskAnalytics(returns, ppy)
        rolling_s = r.rolling_sharpe(window)

        ax.plot(rolling_s.values, color=self.COLORS["accent1"],
                linewidth=1.0, alpha=0.8)
        ax.axhline(y=0, color=self.COLORS["danger"], linestyle="--",
                   linewidth=0.5, alpha=0.5)
        ax.axhline(y=1.0, color=self.COLORS["accent2"], linestyle="--",
                   linewidth=0.5, alpha=0.5, label="Sharpe = 1.0")
        ax.axhline(y=2.0, color=self.COLORS["warning"], linestyle="--",
                   linewidth=0.5, alpha=0.5, label="Sharpe = 2.0")

        ax.set_title(f"Rolling Sharpe Ratio (window={window} bars ≈ {window/ppy*365:.0f}d)",
                     color=self.COLORS["fg"], fontweight="bold")
        ax.set_ylabel("Sharpe", color=self.COLORS["fg"])
        ax.legend(loc="upper right", fontsize=7)
        ax.tick_params(colors=self.COLORS["fg"])

    def _plot_returns_distribution(self, ax, returns):
        """Return distribution histogram vs Normal overlay."""
        ax.hist(returns * 100, bins=80, density=True, alpha=0.6,
                color=self.COLORS["accent1"], edgecolor="none",
                label="Strategy Returns")

        # Normal overlay
        mu, sigma = np.mean(returns * 100), np.std(returns * 100)
        if sigma > 1e-12:
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
            ax.plot(x, sp_stats.norm.pdf(x, mu, sigma),
                    color=self.COLORS["warning"], linewidth=1.5,
                    linestyle="--", label="Normal Distribution")

        ax.set_title("Return Distribution vs. Normal",
                     color=self.COLORS["fg"], fontweight="bold")
        ax.set_xlabel("Return (%)", color=self.COLORS["fg"])
        ax.legend(fontsize=7)
        ax.tick_params(colors=self.COLORS["fg"])

    def _plot_qq_plot(self, ax, returns):
        """Q-Q plot to visualize fat tails."""
        theoretical_q = sp_stats.norm.ppf(
            np.linspace(0.001, 0.999, min(1000, len(returns))))
        sorted_r = np.sort(returns)
        sample_q = np.percentile(sorted_r,
                                  np.linspace(0.1, 99.9, len(theoretical_q)))

        ax.scatter(theoretical_q, sample_q * 100, s=2,
                   color=self.COLORS["accent1"], alpha=0.5)
        # Reference line
        lim = max(abs(theoretical_q.min()), abs(theoretical_q.max()))
        ref_line = np.linspace(-lim, lim, 100)
        ax.plot(ref_line, ref_line * np.std(returns) * 100 + np.mean(returns) * 100,
                color=self.COLORS["danger"], linewidth=1, linestyle="--",
                alpha=0.6, label="Normal Reference")

        ax.set_title("Q-Q Plot (Fat Tail Detection)",
                     color=self.COLORS["fg"], fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles", color=self.COLORS["fg"])
        ax.set_ylabel("Sample Quantiles (%)", color=self.COLORS["fg"])
        ax.legend(fontsize=7)
        ax.tick_params(colors=self.COLORS["fg"])

    def _plot_monthly_heatmap(self, ax, returns, ppy):
        """Monthly returns heatmap."""
        # Approximate months from hourly data
        hours_per_month = ppy / 12
        n_months = int(len(returns) / max(hours_per_month, 1))

        if n_months < 2:
            ax.text(0.5, 0.5, "Insufficient data for monthly heatmap",
                    transform=ax.transAxes, ha="center", va="center",
                    color=self.COLORS["fg"])
            ax.set_title("Monthly Returns Heatmap",
                         color=self.COLORS["fg"], fontweight="bold")
            return

        monthly_returns = []
        for i in range(n_months):
            start = int(i * hours_per_month)
            end = int(min((i + 1) * hours_per_month, len(returns)))
            if end > start:
                m_ret = np.prod(1 + returns[start:end]) - 1
                monthly_returns.append(m_ret * 100)

        # Reshape into year × month grid
        months_per_row = 12
        n_rows = max(1, (len(monthly_returns) + months_per_row - 1) // months_per_row)
        padded = monthly_returns + [np.nan] * (n_rows * months_per_row - len(monthly_returns))
        grid = np.array(padded).reshape(n_rows, months_per_row)

        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto",
                       vmin=-10, vmax=10)
        plt.colorbar(im, ax=ax, label="Return %", shrink=0.6)

        # Annotate
        for i in range(n_rows):
            for j in range(months_per_row):
                val = grid[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 5 else "black"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")

        ax.set_title("Monthly Returns Heatmap (%)",
                     color=self.COLORS["fg"], fontweight="bold")
        ax.set_xlabel("Month", color=self.COLORS["fg"])
        ax.set_ylabel("Year", color=self.COLORS["fg"])
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_labels, fontsize=7, color=self.COLORS["fg"])
        ax.tick_params(colors=self.COLORS["fg"])

    def _plot_permutation_test(self, ax, stat_val, observed_sharpe):
        """Permutation test distribution with observed Sharpe marked."""
        perm_sharpes = stat_val.get("permuted_sharpes", np.array([]))
        p_value = stat_val.get("p_value", 1.0)

        if len(perm_sharpes) == 0 or np.std(perm_sharpes) < 1e-8:
            ax.text(0.5, 0.5, "Monte Carlo test invalid (zero variance)",
                    transform=ax.transAxes, ha="center", va="center",
                    color=self.COLORS["danger"])
            ax.set_title("Monte Carlo Permutation Test", color=self.COLORS["fg"], fontweight="bold")
            return

        ax.hist(perm_sharpes, bins=80, density=True, alpha=0.6,
                color=self.COLORS["fg"], edgecolor="none",
                label="Permuted Sharpes (H₀)")
        ax.axvline(x=observed_sharpe, color=self.COLORS["danger"],
                   linewidth=2, linestyle="-",
                   label=f"Observed Sharpe = {observed_sharpe:.3f}")

        # Significance regions
        p95 = np.percentile(perm_sharpes, 95)
        p99 = np.percentile(perm_sharpes, 99)
        ax.axvline(x=p95, color=self.COLORS["warning"], linewidth=1,
                   linestyle="--", alpha=0.7, label=f"95th pctl = {p95:.3f}")
        ax.axvline(x=p99, color=self.COLORS["accent2"], linewidth=1,
                   linestyle="--", alpha=0.7, label=f"99th pctl = {p99:.3f}")

        sig_color = self.COLORS["accent2"] if p_value < 0.05 else self.COLORS["danger"]
        sig_text = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
        ax.text(0.02, 0.95, f"p-value = {p_value:.4f} → {sig_text}",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                color=sig_color, verticalalignment="top")

        ax.set_title("Monte Carlo Permutation Test (10,000 simulations)",
                     color=self.COLORS["fg"], fontweight="bold")
        ax.set_xlabel("Sharpe Ratio", color=self.COLORS["fg"])
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(colors=self.COLORS["fg"])

    def _plot_action_distribution(self, ax, actions):
        """Action distribution pie chart."""
        labels = ["Short (0)", "Flat (1)", "Long (2)"]
        counts = [np.sum(actions == 0), np.sum(actions == 1), np.sum(actions == 2)]
        colors = [self.COLORS["danger"], self.COLORS["fg"],
                  self.COLORS["accent2"]]

        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, autopct="%1.1f%%",
            colors=colors, textprops={"fontsize": 8, "color": "white"},
            pctdistance=0.75
        )
        ax.set_title("Action Distribution",
                     color=self.COLORS["fg"], fontweight="bold")

    def _plot_trade_analysis(self, ax, trades):
        """Trade returns scatter plot."""
        if not trades:
            ax.text(0.5, 0.5, "No trades recorded",
                    transform=ax.transAxes, ha="center", va="center",
                    color=self.COLORS["fg"])
            return

        trade_returns = [t.get("return", 0) * 100 for t in trades]
        durations = [t.get("duration_steps", 0) for t in trades]
        colors_list = [self.COLORS["accent2"] if r > 0 else self.COLORS["danger"]
                       for r in trade_returns]

        ax.scatter(durations, trade_returns, c=colors_list, s=15, alpha=0.6)
        ax.axhline(y=0, color=self.COLORS["fg"], linewidth=0.5, alpha=0.3)

        ax.set_title("Trade Returns vs. Duration",
                     color=self.COLORS["fg"], fontweight="bold")
        ax.set_xlabel("Duration (bars)", color=self.COLORS["fg"])
        ax.set_ylabel("Return (%)", color=self.COLORS["fg"])
        ax.tick_params(colors=self.COLORS["fg"])

    def _plot_metrics_table(self, fig, risk, ab_decomp, stat_val, capacity,
                            bm_risk=None, sp_risk=None):
        """Full metrics table — Page 3."""
        fig.suptitle("QUANTITATIVE RISK METRICS — FULL DETAIL",
                     fontsize=14, fontweight="bold",
                     color=self.COLORS["accent1"], y=0.96)

        # Build the data
        perf_data = [
            ["PERFORMANCE METRICS", ""],
            ["Total Return", risk.to_dict()["Total Return"]],
            ["Annualized Return", risk.to_dict()["Annualized Return"]],
            ["Annualized Volatility", risk.to_dict()["Annualized Volatility"]],
            ["Sharpe Ratio", risk.to_dict()["Sharpe Ratio"]],
            ["Sortino Ratio", risk.to_dict()["Sortino Ratio"]],
            ["Calmar Ratio", risk.to_dict()["Calmar Ratio"]],
            ["Omega Ratio (θ=0)", risk.to_dict()["Omega Ratio (θ=0)"]],
            ["Profit Factor", risk.to_dict()["Profit Factor"]],
            ["Win Rate", risk.to_dict()["Win Rate"]],
            ["", ""],
            ["BENCHMARK COMPARISON", ""],
        ]
        if bm_risk is not None:
            perf_data += [
                ["BTC Buy & Hold Sharpe", f"{bm_risk.sharpe_ratio():.3f}"],
                ["BTC Buy & Hold Sortino", f"{bm_risk.sortino_ratio():.3f}"],
                ["BTC Buy & Hold Max DD", f"{bm_risk.max_drawdown()*100:.2f}%"],
            ]
        if sp_risk is not None:
            perf_data += [
                ["S&P 500 Sharpe", f"{sp_risk.sharpe_ratio():.3f}"],
                ["S&P 500 Sortino", f"{sp_risk.sortino_ratio():.3f}"],
                ["S&P 500 Max DD", f"{sp_risk.max_drawdown()*100:.2f}%"],
            ]
        if bm_risk is None and sp_risk is None:
            perf_data.append(["(Benchmark data not available)", ""])
        perf_data += [
            ["", ""],
            ["TAIL RISK METRICS", ""],
            ["Max Drawdown", risk.to_dict()["Max Drawdown"]],
            ["VaR (95%)", risk.to_dict()["VaR (95%)"]],
            ["VaR (99%)", risk.to_dict()["VaR (99%)"]],
            ["CVaR / Expected Shortfall (95%)", risk.to_dict()["CVaR/ES (95%)"]],
            ["CVaR / Expected Shortfall (99%)", risk.to_dict()["CVaR/ES (99%)"]],
            ["Skewness", risk.to_dict()["Skewness"]],
            ["Excess Kurtosis", risk.to_dict()["Excess Kurtosis"]],
            ["", ""],
            ["ALPHA DECOMPOSITION", ""],
            ["Alpha (per period)", f"{ab_decomp.get('alpha_per_period', 0):.6f}"],
            ["Alpha t-statistic", f"{ab_decomp.get('alpha_tstat', 0):.3f}"],
            ["Alpha p-value", f"{ab_decomp.get('alpha_pvalue', 1):.4f}"],
            ["Beta (benchmark exposure)", f"{ab_decomp.get('beta', 0):.4f}"],
            ["R² (benchmark correlation)", f"{ab_decomp.get('r_squared', 0):.4f}"],
            ["", ""],
            ["STATISTICAL VALIDATION", ""],
            ["Permutation Test p-value", f"{stat_val.get('p_value', 'N/A')}"],
            ["Significance (p < 0.05)", "YES ✅" if stat_val.get("p_value", 1) < 0.05 else "NO ❌"],
            ["", ""],
            ["CAPACITY ESTIMATION", ""],
            ["Est. Capacity (USD)", f"${capacity.get('capacity_usd', 0):,.0f}"],
            ["Current Market Impact", f"{capacity.get('impact_bps_current', 0):.2f} bps"],
            ["Participation Rate", f"{capacity.get('participation_rate', 0)*100:.6f}%"],
        ]

        ax = fig.add_subplot(111)
        ax.axis("off")

        table = ax.table(
            cellText=perf_data,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
            colWidths=[0.55, 0.35],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

        # Style the table
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(self.COLORS["grid"])
            if row == 0:
                cell.set_facecolor(self.COLORS["primary"])
                cell.set_text_props(color="white", fontweight="bold")
            elif perf_data[row - 1][1] == "":
                cell.set_facecolor("#1a1f2e")
                cell.set_text_props(color=self.COLORS["accent1"],
                                    fontweight="bold")
            else:
                cell.set_facecolor(self.COLORS["bg"])
                cell.set_text_props(color=self.COLORS["fg"])


# ============================================================
# 5. HIGH-LEVEL API — Run full evaluation pipeline
# ============================================================

def run_institutional_eval(
    equity: np.ndarray,
    actions: np.ndarray,
    trades: List[Dict],
    benchmark_prices: np.ndarray,
    initial_balance: float = 10_000.0,
    periods_per_year: int = 8760,
    n_trades_count: int = 0,
    output_dir: Path = Path("data/eval"),
    strategy_name: str = "CNN+LSTM PPO Alpha Strategy",
    sp500_equity: Optional[np.ndarray] = None,
) -> Dict:
    """
    Complete institutional evaluation pipeline.
    Returns dict of all metrics + generates PDF tearsheet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Compute returns ----
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    # Benchmark returns (Buy & Hold)
    bm_equity = initial_balance * (benchmark_prices / benchmark_prices[0])
    n = min(len(equity), len(bm_equity))
    bm_equity = bm_equity[:n]
    bm_returns = np.diff(benchmark_prices[:n]) / benchmark_prices[:n - 1]
    bm_returns = np.nan_to_num(bm_returns, nan=0.0, posinf=0.0, neginf=0.0)

    # Align lengths
    min_len = min(len(returns), len(bm_returns))
    returns = returns[:min_len]
    bm_returns = bm_returns[:min_len]

    # ---- 1. Risk Analytics ----
    print("[eval] Computing risk analytics...")
    risk = RiskAnalytics(returns, periods_per_year)

    # ---- 2. Statistical Validation ----
    print("[eval] Running Monte Carlo permutation test (10,000 simulations)...")
    p_value, perm_sharpes = StatisticalValidator.bootstrap_sharpe_pvalue(
        returns, risk.sharpe_ratio(), n_simulations=10_000,
        periods_per_year=periods_per_year
    )
    stat_val = {"p_value": p_value, "permuted_sharpes": perm_sharpes}

    print(f"       Observed Sharpe: {risk.sharpe_ratio():.3f}")
    print(f"       Permutation p-value: {p_value:.4f}")
    if p_value < 0.01:
        print("       → HIGHLY SIGNIFICANT (p < 0.01) ✅")
    elif p_value < 0.05:
        print("       → SIGNIFICANT (p < 0.05) ✅")
    else:
        print("       → NOT SIGNIFICANT (p ≥ 0.05) ⚠️")

    # ---- 3. Alpha/Beta Decomposition ----
    print("[eval] Computing Alpha-Beta decomposition...")
    ab_decomp = StatisticalValidator.alpha_beta_decomposition(returns, bm_returns)
    print(f"       Alpha t-stat: {ab_decomp['alpha_tstat']:.3f} "
          f"(significant if > 2.0)")
    print(f"       Beta: {ab_decomp['beta']:.4f} "
          f"(low = idiosyncratic alpha)")
    print(f"       R²: {ab_decomp['r_squared']:.4f} "
          f"(low = uncorrelated with BTC)")

    # ---- 3b. Benchmark RiskAnalytics (for table on Page 3) ----
    bm_risk = RiskAnalytics(bm_returns, periods_per_year)
    sp_risk = None
    if sp500_equity is not None and len(sp500_equity) > 1:
        sp_rets = np.diff(sp500_equity) / sp500_equity[:-1]
        sp_rets = np.nan_to_num(sp_rets, nan=0.0, posinf=0.0, neginf=0.0)
        sp_risk = RiskAnalytics(sp_rets, periods_per_year)
        print(f"[eval] S&P 500 Sharpe: {sp_risk.sharpe_ratio():.3f}, "
              f"Sortino: {sp_risk.sortino_ratio():.3f}")
    print(f"[eval] BTC B&H Sharpe: {bm_risk.sharpe_ratio():.3f}, "
          f"Sortino: {bm_risk.sortino_ratio():.3f}")

    # ---- 4. Jarque-Bera test ----
    jb_stat, jb_pval = StatisticalValidator.jarque_bera_test(returns)
    print(f"[eval] Jarque-Bera: stat={jb_stat:.1f}, p={jb_pval:.4f} "
          f"({'Non-normal ✅' if jb_pval < 0.05 else 'Normal ❌'})")

    # ---- 5. Capacity Estimation ----
    print("[eval] Estimating strategy capacity...")
    avg_trade_notional = equity[-1] * 0.30 if len(equity) > 0 else 3000
    capacity = CapacityEstimator.estimate_capacity(
        equity[:n], n_trades_count, avg_trade_notional,
        current_sharpe=risk.sharpe_ratio()
    )
    print(f"       Est. Capacity: ${capacity['capacity_usd']:,.0f}")
    print(f"       Current Impact: {capacity['impact_bps_current']:.2f} bps")

    # ---- 6. Generate PDF Tearsheet ----
    print("[eval] Generating institutional PDF tearsheet...")
    tearsheet = InstitutionalTearsheet(strategy_name)
    pdf_path = output_dir / "institutional_tearsheet.pdf"

    tearsheet.generate(
        equity=equity[:n],
        returns=returns,
        benchmark_equity=bm_equity,
        benchmark_returns=bm_returns,
        actions=actions[:n] if len(actions) >= n else actions,
        trades=trades,
        risk=risk,
        stat_val=stat_val,
        ab_decomp=ab_decomp,
        capacity=capacity,
        output_path=pdf_path,
        periods_per_year=periods_per_year,
        sp500_equity=sp500_equity,
        bm_risk=bm_risk,
        sp_risk=sp_risk,
    )

    # ---- 7. Print Summary ----
    print(f"\n{'='*70}")
    print(f" INSTITUTIONAL EVALUATION COMPLETE — {strategy_name}")
    print(f"{'='*70}")
    metrics = risk.to_dict()
    for k, v in metrics.items():
        print(f"  {k:30s}  {v}")
    print(f"{'='*70}")
    print(f"  PDF Tearsheet: {pdf_path}")
    print(f"{'='*70}")

    return {
        "risk_metrics": metrics,
        "stat_validation": {
            "p_value": p_value,
            "alpha_tstat": ab_decomp["alpha_tstat"],
            "beta": ab_decomp["beta"],
            "r_squared": ab_decomp["r_squared"],
        },
        "capacity": capacity,
        "pdf_path": str(pdf_path),
    }
