import pandas as pd
from scipy.stats import spearmanr

# Load your CSV
df = pd.read_csv('mqm_scores_with_spearman.csv')

# df = df.dropna(subset=['domain'])
# List of LLM MQM columns
llm_columns = ['mqm_claude', 'mqm_gemini', 'mqm_gpt']

# Group by domain
grouped = df.groupby('domain')

# Store results
correlations = []

for domain, group in grouped:
    domain_result = {'domain': domain}
    for col in llm_columns:
        # Drop rows with NaN in relevant columns
        valid_data = group[['mqm_human', col]].dropna()
        if len(valid_data) >= 2:
            corr, _ = spearmanr(valid_data['mqm_human'], valid_data[col])
        else:
            corr = None  # Not enough data
        domain_result[col] = corr
    correlations.append(domain_result)

# Convert to DataFrame for display
correlation_df = pd.DataFrame(correlations)

print(correlation_df)