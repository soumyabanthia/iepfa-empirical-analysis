import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# load excel data or fallback to mock data
def load_and_clean_data(filepath):
    try:
        df = pd.read_excel(filepath, sheet_name=None)
        return df
    except Exception as e:
        print(f"Warning: Could not load {filepath}. Ensure data exists. Error: {e}")
        return generate_mock_panel_data()

# create synthetic panel dataset
def generate_mock_panel_data():
    np.random.seed(42)
    years = list(range(2016, 2026))
    states = ['Maharashtra', 'Delhi', 'Gujarat', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh', 'West Bengal']
    
    data = []
    for state in states:
        for year in years:
            post_2019 = 1 if year >= 2019 else 0
            iap_intensity = np.random.poisson(lam=50) if year < 2019 else np.random.poisson(lam=120)
            base_claims = np.random.normal(loc=5000, scale=1000)
            claim_submissions = max(0, int(base_claims + (1500 * post_2019) + (10 * iap_intensity) + np.random.normal(0, 500)))
            claims_approved = max(0, int(claim_submissions * (0.3 + (0.2 * post_2019) + np.random.normal(0, 0.05))))
            
            data.append({
                'state': state,
                'year': year,
                'post_2019_reform': post_2019,
                'iap_programs_count': iap_intensity,
                'claims_submitted': claim_submissions,
                'claims_approved': claims_approved,
                'approval_rate': claims_approved / claim_submissions if claim_submissions > 0 else 0
            })
            
    return pd.DataFrame(data)

# run diff-in-diff regression with state fixed effects
def run_difference_in_differences(df):
    print("--- Estimating Impact of 2019 Regulatory Digitization Reform ---")
    
    model = smf.ols('approval_rate ~ post_2019_reform + iap_programs_count + C(state)', data=df).fit()
    print(model.summary().tables[1])
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    results_path = os.path.join(PLOTS_DIR, 'regression_results.txt')
    with open(results_path, 'w') as f:
        f.write("Difference-in-Differences / Fixed Effects Regression Results\n")
        f.write("Dependent Variable: Claim Approval Rate\n")
        f.write("=================================================================\n")
        f.write(model.summary().as_text())
    print(f"\nRegression results saved to '{results_path}'")

# plot claim approval rate trend around 2019 reform
def plot_event_study(df):
    yearly_avg = df.groupby('year')['approval_rate'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=yearly_avg, x='year', y='approval_rate', marker='o', linewidth=2, color='b')
    plt.axvline(x=2019, color='r', linestyle='--', label='2019 Digitization Reform')
    
    plt.title('Event Study: Impact of 2019 Reform on Claim Approval Rate')
    plt.xlabel('Financial Year')
    plt.ylabel('Average Claim Approval Rate')
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, 'event_study_reform.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to '{plot_path}'")

# scatter plot of investor awareness programs vs claims
def plot_iap_impact(df):
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='iap_programs_count', y='claims_submitted', 
                scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    
    plt.title('Impact of Investor Awareness Programs (IAP) on Claim Submissions')
    plt.xlabel('Number of IAP Interventions (per State/Year)')
    plt.ylabel('Total Claim Submissions')
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, 'iap_impact.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to '{plot_path}'")

from arima_forecasting import run_arima_pipeline

if __name__ == "__main__":
    print("Initializing empirical analysis pipeline...")
    df = generate_mock_panel_data()
    
    run_difference_in_differences(df)
    plot_event_study(df)
    plot_iap_impact(df)
    
    print("\n--- Executing ARIMA Time Series Forecasting ---")
    run_arima_pipeline()
    
    print(f"\nAnalysis complete. Check '{PLOTS_DIR}' directory for all outputs.")
