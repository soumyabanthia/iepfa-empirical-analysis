import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})


def load_historical_series():
    claims_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    claims_filed = [4026, 19188, 16166, 14032, 28647, 37920, 55031, 51449]

    approvals_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    claims_approved = [155, 712, 6989, 7262, 10472, 10989, 16985, 12749, 49505]

    div_years = [2022, 2023, 2024, 2025, 2026]
    div_refunded_cr = [10.53, 11.54, 16.59, 24.97, 49.71]

    shares_years = [2022, 2023, 2024, 2025, 2026]
    shares_restituted_m = [6.12, 7.65, 11.04, 7.87, 28.58]

    df_claims = pd.DataFrame({"year": claims_years, "claims_filed": claims_filed}).set_index("year")
    df_approvals = pd.DataFrame({"year": approvals_years, "claims_approved": claims_approved}).set_index("year")
    df_div = pd.DataFrame({"year": div_years, "dividend_refunded_cr": div_refunded_cr}).set_index("year")
    df_shares = pd.DataFrame({"year": shares_years, "shares_restituted_m": shares_restituted_m}).set_index("year")

    return df_claims, df_approvals, df_div, df_shares


def fit_best_arima(series, p_range=(0, 2), d_range=(1, 2), q_range=(0, 2), trend='t'):
    best_aic = float("inf")
    best_order = None
    best_model_fit = None

    for p in range(p_range[0], p_range[1] + 1):
        for d in range(d_range[0], d_range[1] + 1):
            for q in range(q_range[0], q_range[1] + 1):
                try:
                    model = ARIMA(series, order=(p, d, q), trend=trend)
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model_fit = fitted
                except Exception:
                    continue

    if best_model_fit is None:
        for p in range(p_range[0], p_range[1] + 1):
            for d in range(d_range[0], d_range[1] + 1):
                for q in range(q_range[0], q_range[1] + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q), trend=None)
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            best_model_fit = fitted
                    except Exception:
                        continue

    return best_order, best_model_fit, best_aic


def generate_forecast(fitted_model, last_year, steps=5):
    forecast_res = fitted_model.get_forecast(steps=steps)
    forecast_df = forecast_res.summary_frame(alpha=0.05)

    future_years = [last_year + i for i in range(1, steps + 1)]
    forecast_df.index = future_years
    return forecast_df


def plot_single_forecast(history_series, forecast_df, title, ylabel, filename, color="#1f77b4"):
    plt.figure(figsize=(9, 5.5))

    plt.plot(history_series.index, history_series.values, marker="o", linewidth=2.5,
             color=color, label="Historical Actuals")

    plt.plot(forecast_df.index, forecast_df["mean"], marker="s", linestyle="--",
             linewidth=2.5, color="#d62728", label="ARIMA Forecast (2026–2030)")

    plt.fill_between(forecast_df.index, forecast_df["mean_ci_lower"], forecast_df["mean_ci_upper"],
                     color="#d62728", alpha=0.18, label="95% Confidence Interval")

    plt.plot([history_series.index[-1], forecast_df.index[0]],
             [history_series.values[-1], forecast_df["mean"].iloc[0]],
             linestyle=":", color="#d62728", linewidth=1.8)

    plt.title(title, pad=12, fontweight="bold")
    plt.xlabel("Financial Year (End)", labelpad=8)
    plt.ylabel(ylabel, labelpad=8)
    plt.xticks(list(history_series.index) + list(forecast_df.index))
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    shutil.copy(save_path, os.path.join(PAPER_DIR, filename))
    print(f"Saved: {save_path}")


def plot_combined_panel(df_claims, f_claims, df_app, f_app, df_div, f_div):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].plot(df_claims.index, df_claims["claims_filed"] / 1000, marker="o", color="#1f77b4", linewidth=2, label="Historical")
    axes[0].plot(f_claims.index, f_claims["mean"] / 1000, marker="s", linestyle="--", color="#d62728", linewidth=2, label="Forecast")
    axes[0].fill_between(f_claims.index, f_claims["mean_ci_lower"] / 1000, f_claims["mean_ci_upper"] / 1000, color="#d62728", alpha=0.15)
    axes[0].plot([df_claims.index[-1], f_claims.index[0]], [df_claims["claims_filed"].iloc[-1] / 1000, f_claims["mean"].iloc[0] / 1000], linestyle=":", color="#d62728")
    axes[0].set_title("(a) Form IEPF-5 Claims Submitted ('000s)", fontweight="bold")
    axes[0].set_xlabel("Financial Year")
    axes[0].set_ylabel("Claims Submitted ('000s)")
    axes[0].legend(loc="upper left")

    axes[1].plot(df_app.index, df_app["claims_approved"] / 1000, marker="o", color="#2ca02c", linewidth=2, label="Historical")
    axes[1].plot(f_app.index, f_app["mean"] / 1000, marker="s", linestyle="--", color="#d62728", linewidth=2, label="Forecast")
    axes[1].fill_between(f_app.index, f_app["mean_ci_lower"] / 1000, f_app["mean_ci_upper"] / 1000, color="#d62728", alpha=0.15)
    axes[1].plot([df_app.index[-1], f_app.index[0]], [df_app["claims_approved"].iloc[-1] / 1000, f_app["mean"].iloc[0] / 1000], linestyle=":", color="#d62728")
    axes[1].set_title("(b) Claims Approved / Sanctioned ('000s)", fontweight="bold")
    axes[1].set_xlabel("Financial Year")
    axes[1].set_ylabel("Approved Claims ('000s)")
    axes[1].legend(loc="upper left")

    axes[2].plot(df_div.index, df_div["dividend_refunded_cr"], marker="o", color="#ff7f0e", linewidth=2, label="Historical")
    axes[2].plot(f_div.index, f_div["mean"], marker="s", linestyle="--", color="#d62728", linewidth=2, label="Forecast")
    axes[2].fill_between(f_div.index, f_div["mean_ci_lower"], f_div["mean_ci_upper"], color="#d62728", alpha=0.15)
    axes[2].plot([df_div.index[-1], f_div.index[0]], [df_div["dividend_refunded_cr"].iloc[-1], f_div["mean"].iloc[0]], linestyle=":", color="#d62728")
    axes[2].set_title("(c) Dividend Amount Refunded (INR Cr)", fontweight="bold")
    axes[2].set_xlabel("Financial Year")
    axes[2].set_ylabel("Refunded Amount (INR Cr)")
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    comb_path = os.path.join(PLOTS_DIR, "arima_combined_forecast.png")
    plt.savefig(comb_path, dpi=300)
    plt.close()

    shutil.copy(comb_path, os.path.join(PAPER_DIR, "arima_combined_forecast.png"))
    print(f"Saved: {comb_path}")


def run_arima_pipeline():
    df_claims, df_approvals, df_div, df_shares = load_historical_series()

    order_claims, fit_claims, aic_claims = fit_best_arima(df_claims["claims_filed"], p_range=(0, 2), d_range=(1, 2), q_range=(0, 2))
    f_claims = generate_forecast(fit_claims, last_year=2025, steps=5)
    print(f"Claim Filings: ARIMA{order_claims} (AIC: {aic_claims:.2f})")

    order_app, fit_app, aic_app = fit_best_arima(df_approvals["claims_approved"], p_range=(0, 2), d_range=(1, 2), q_range=(0, 2))
    f_app = generate_forecast(fit_app, last_year=2026, steps=4)
    print(f"Approved Claims: ARIMA{order_app} (AIC: {aic_app:.2f})")

    order_div, fit_div, aic_div = fit_best_arima(df_div["dividend_refunded_cr"], p_range=(0, 1), d_range=(1, 2), q_range=(0, 1))
    f_div = generate_forecast(fit_div, last_year=2026, steps=4)
    print(f"Dividend Restitution: ARIMA{order_div} (AIC: {aic_div:.2f})")

    plot_single_forecast(df_claims["claims_filed"], f_claims,
                         "ARIMA Forecast: Form IEPF-5 Annual Claim Submissions (2026–2030)",
                         "Form IEPF-5 Submissions", "arima_claims_forecast.png", color="#1f77b4")

    plot_single_forecast(df_div["dividend_refunded_cr"], f_div,
                         "ARIMA Forecast: Dividend Restitution Quantum (2026–2030)",
                         "Dividend Refunded (INR Crores)", "arima_refunds_forecast.png", color="#ff7f0e")

    plot_combined_panel(df_claims, f_claims, df_approvals, f_app, df_div, f_div)

    summary_path = os.path.join(PLOTS_DIR, "arima_forecast_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("IEPFA ARIMA Time Series Forecasting Projections\n")
        f.write("=================================================================\n\n")

        f.write("1. FORM IEPF-5 CLAIMS SUBMITTED\n")
        f.write(f"Model: ARIMA{order_claims} | AIC: {aic_claims:.2f}\n")
        f.write("Historical Actuals:\n")
        for y, val in df_claims["claims_filed"].items():
            f.write(f"  FY {y-1}-{str(y)[2:]}: {val:,.0f}\n")
        f.write("Projected Forecast (with 95% Confidence Interval):\n")
        for y, row in f_claims.iterrows():
            f.write(f"  FY {y-1}-{str(y)[2:]}: {row['mean']:,.0f}  [95% CI: {row['mean_ci_lower']:,.0f} to {row['mean_ci_upper']:,.0f}]\n")
        f.write("\n" + "-"*65 + "\n\n")

        f.write("2. CLAIMS APPROVED / SANCTIONED FOR REFUND\n")
        f.write(f"Model: ARIMA{order_app} | AIC: {aic_app:.2f}\n")
        f.write("Historical Actuals:\n")
        for y, val in df_approvals["claims_approved"].items():
            f.write(f"  FY {y-1}-{str(y)[2:]}: {val:,.0f}\n")
        f.write("Projected Forecast (with 95% Confidence Interval):\n")
        for y, row in f_app.iterrows():
            f.write(f"  FY {y-1}-{str(y)[2:]}: {row['mean']:,.0f}  [95% CI: {row['mean_ci_lower']:,.0f} to {row['mean_ci_upper']:,.0f}]\n")
        f.write("\n" + "-"*65 + "\n\n")

        f.write("3. DIVIDEND AMOUNT REFUNDED (INR CRORES)\n")
        f.write(f"Model: ARIMA{order_div} | AIC: {aic_div:.2f}\n")
        f.write("Historical Actuals:\n")
        for y, val in df_div["dividend_refunded_cr"].items():
            f.write(f"  FY {y-1}-{str(y)[2:]}: INR {val:,.2f} Cr\n")
        f.write("Projected Forecast (with 95% Confidence Interval):\n")
        for y, row in f_div.iterrows():
            f.write(f"  FY {y-1}-{str(y)[2:]}: INR {row['mean']:,.2f} Cr  [95% CI: INR {row['mean_ci_lower']:,.2f} to INR {row['mean_ci_upper']:,.2f} Cr]\n")
        f.write("\n=================================================================\n")

    print(f"Saved forecast summary: {summary_path}")


if __name__ == "__main__":
    run_arima_pipeline()

