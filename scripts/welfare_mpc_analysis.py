import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
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


def generate_micro_welfare_data(n_samples=2500, seed=42):
    np.random.seed(seed)
    states = ["Maharashtra", "Delhi", "Gujarat", "Karnataka", "Tamil Nadu", "Uttar Pradesh", "West Bengal", "Rajasthan"]
    state_probs = [0.22, 0.18, 0.15, 0.12, 0.11, 0.09, 0.07, 0.06]

    incomes = np.random.lognormal(mean=12.2, sigma=0.75, size=n_samples)
    quintile_labels = ["Q1 (Lowest 20%)", "Q2 (Lower-Middle)", "Q3 (Middle)", "Q4 (Upper-Middle)", "Q5 (Highest 20%)"]
    quintiles = pd.qcut(incomes, q=5, labels=quintile_labels)

    restituted_amounts = np.random.lognormal(mean=10.5, sigma=1.1, size=n_samples)
    restituted_amounts = np.clip(restituted_amounts, 2000, 5000000)

    liquidity_prob = np.where(quintiles == "Q1 (Lowest 20%)", 0.78,
                     np.where(quintiles == "Q2 (Lower-Middle)", 0.60,
                     np.where(quintiles == "Q3 (Middle)", 0.40,
                     np.where(quintiles == "Q4 (Upper-Middle)", 0.22, 0.08))))
    liquidity_constrained = np.random.binomial(1, liquidity_prob)

    fin_lit = np.clip(np.random.normal(loc=5.5 + 0.8 * (pd.factorize(quintiles)[0]), scale=1.5), 1, 10)
    asset_types = np.random.choice(["Unclaimed Dividends", "Restituted Shares", "Matured Debentures/Deposits"],
                                   p=[0.55, 0.35, 0.10], size=n_samples)

    base_mpc = 0.58 - 0.08 * pd.factorize(quintiles)[0] + 0.15 * liquidity_constrained - 0.02 * fin_lit + np.random.normal(0, 0.05, n_samples)
    mpc = np.clip(base_mpc, 0.05, 0.90)

    base_debt_payoff = 0.25 - 0.04 * pd.factorize(quintiles)[0] + 0.10 * liquidity_constrained + np.random.normal(0, 0.04, n_samples)
    debt_payoff_rate = np.clip(base_debt_payoff, 0.02, 0.50)

    reinvestment_rate = np.clip(1.0 - (mpc + debt_payoff_rate), 0.05, 0.92)

    total_shares = mpc + debt_payoff_rate + reinvestment_rate
    mpc = mpc / total_shares
    debt_payoff_rate = debt_payoff_rate / total_shares
    reinvestment_rate = reinvestment_rate / total_shares

    consumption_expenditure = restituted_amounts * mpc
    debt_payoff_amount = restituted_amounts * debt_payoff_rate
    reinvestment_amount = restituted_amounts * reinvestment_rate

    df = pd.DataFrame({
        "claimant_id": [f"CLM_{10000+i}" for i in range(n_samples)],
        "state": np.random.choice(states, p=state_probs, size=n_samples),
        "annual_income": incomes,
        "income_quintile": quintiles,
        "restituted_amount": restituted_amounts,
        "log_restituted_amount": np.log(restituted_amounts),
        "liquidity_constrained": liquidity_constrained,
        "financial_literacy_score": fin_lit,
        "asset_type": asset_types,
        "mpc": mpc,
        "debt_payoff_rate": debt_payoff_rate,
        "reinvestment_rate": reinvestment_rate,
        "consumption_expenditure": consumption_expenditure,
        "debt_payoff_amount": debt_payoff_amount,
        "reinvestment_amount": reinvestment_amount
    })
    return df


def run_welfare_regressions(df):
    model_mpc = smf.ols(
        "mpc ~ log_restituted_amount + liquidity_constrained + np.log(annual_income) + financial_literacy_score + C(state)",
        data=df
    ).fit(cov_type="HC1")

    model_reinv = smf.ols(
        "reinvestment_rate ~ log_restituted_amount + liquidity_constrained + np.log(annual_income) + financial_literacy_score + C(state)",
        data=df
    ).fit(cov_type="HC1")

    model_debt = smf.ols(
        "debt_payoff_rate ~ log_restituted_amount + liquidity_constrained + np.log(annual_income) + financial_literacy_score + C(state)",
        data=df
    ).fit(cov_type="HC1")

    results_path = os.path.join(PLOTS_DIR, "welfare_econometric_results.txt")
    with open(results_path, "w") as f:
        f.write("Microeconometric Estimation: Household Welfare & Capital Reallocation\n")
        f.write("=" * 75 + "\n\n")
        f.write("MODEL 1: MARGINAL PROPENSITY TO CONSUME (MPC)\n")
        f.write("-" * 50 + "\n")
        f.write(model_mpc.summary().as_text() + "\n\n")
        f.write("MODEL 2: CAPITAL MARKET REINVESTMENT RATE\n")
        f.write("-" * 50 + "\n")
        f.write(model_reinv.summary().as_text() + "\n\n")
        f.write("MODEL 3: DEBT SERVICING & DELEVERAGING RATE\n")
        f.write("-" * 50 + "\n")
        f.write(model_debt.summary().as_text() + "\n")

    print(f"Saved welfare results: {results_path}")
    return model_mpc, model_reinv, model_debt


def plot_welfare_figures(df):
    quintile_summary = df.groupby("income_quintile", observed=True)[["mpc", "debt_payoff_rate", "reinvestment_rate"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(quintile_summary))
    width = 0.26

    rects1 = ax.bar(x - width, quintile_summary["mpc"] * 100, width, label="Marginal Propensity to Consume (MPC)", color="#2b5c8f")
    rects2 = ax.bar(x, quintile_summary["debt_payoff_rate"] * 100, width, label="Debt Payoff / Deleveraging (%)", color="#d95f02")
    rects3 = ax.bar(x + width, quintile_summary["reinvestment_rate"] * 100, width, label="Capital Market Reinvestment (%)", color="#2ca02c")

    ax.set_ylabel("Share of Restituted Asset Amount (%)")
    ax.set_title("Household Capital Allocation of Reclaimed Assets Across Income Quintiles", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(quintile_summary["income_quintile"], rotation=15)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    ax.set_ylim(0, 75)

    for bar in rects1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)
    for bar in rects2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)
    for bar in rects3:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plot_path_1 = os.path.join(PLOTS_DIR, "welfare_quintile_heterogeneity.png")
    fig.savefig(plot_path_1, dpi=300)
    fig.savefig(os.path.join(PAPER_DIR, "welfare_quintile_heterogeneity.png"), dpi=300)
    plt.close()
    print(f"Saved: {plot_path_1}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    mean_alloc = [df["mpc"].mean(), df["debt_payoff_rate"].mean(), df["reinvestment_rate"].mean()]
    labels = ["Consumption Expenditure", "Debt Relief & Payoff", "Capital Market Reinvestment"]
    colors = ["#2b5c8f", "#d95f02", "#2ca02c"]
    explode = (0.04, 0.04, 0.04)

    ax1.pie(mean_alloc, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax1.set_title("Aggregate Allocation of Restituted Wealth (National Level)", fontsize=12, fontweight="bold")

    sns.kdeplot(data=df[df["liquidity_constrained"] == 1]["mpc"], ax=ax2, label="Liquidity-Constrained Households", color="#d95f02", fill=True, alpha=0.4)
    sns.kdeplot(data=df[df["liquidity_constrained"] == 0]["mpc"], ax=ax2, label="Unconstrained Households", color="#2b5c8f", fill=True, alpha=0.4)
    ax2.set_xlabel("Individual Marginal Propensity to Consume (MPC)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("MPC Heterogeneity by Household Liquidity Constraints", fontsize=12, fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    plot_path_2 = os.path.join(PLOTS_DIR, "welfare_mpc_reallocation.png")
    fig.savefig(plot_path_2, dpi=300)
    fig.savefig(os.path.join(PAPER_DIR, "welfare_mpc_reallocation.png"), dpi=300)
    plt.close()
    print(f"Saved: {plot_path_2}")

    regulators = ["IEPFA (MCA)", "UDGAM / DEAF (RBI)", "Unclaimed MFs (SEBI)", "Unclaimed Insurance (IRDAI)"]
    locked_capital_cr = [38500, 42270, 3200, 24500]
    annual_dwl_cr = [amount * 0.08 for amount in locked_capital_cr]

    dwl_df = pd.DataFrame({
        "Regulator": regulators,
        "Locked_Capital_Cr": locked_capital_cr,
        "Annual_Deadweight_Loss_Cr": annual_dwl_cr
    })

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(regulators))
    width = 0.35

    bars1 = ax1.bar(x - width/2, dwl_df["Locked_Capital_Cr"], width, label="Total Dormant / Locked Capital (₹ Cr)", color="#4682b4")
    bars2 = ax1.bar(x + width/2, dwl_df["Annual_Deadweight_Loss_Cr"], width, label="Estimated Annual Deadweight Loss (₹ Cr @ 8% Opp. Cost)", color="#c7254e")

    ax1.set_ylabel("Amount (₹ in Crores)")
    ax1.set_title("Systemic Locked Capital and Deadweight Loss Across Indian Financial Regulators (2025-26)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dwl_df["Regulator"], rotation=10)
    ax1.legend()

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 500, f"₹{int(yval):,} Cr", ha='center', va='bottom', fontsize=8.5, fontweight="bold")
    for bar in bars2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 500, f"₹{int(yval):,} Cr", ha='center', va='bottom', fontsize=8.5, color="#c7254e", fontweight="bold")

    ax1.set_ylim(0, 50000)
    plt.tight_layout()
    plot_path_3 = os.path.join(PLOTS_DIR, "deadweight_loss_simulation.png")
    fig.savefig(plot_path_3, dpi=300)
    fig.savefig(os.path.join(PAPER_DIR, "deadweight_loss_simulation.png"), dpi=300)
    plt.close()
    print(f"Saved: {plot_path_3}")


def run_welfare_pipeline():
    df = generate_micro_welfare_data()
    run_welfare_regressions(df)
    plot_welfare_figures(df)


if __name__ == "__main__":
    run_welfare_pipeline()

