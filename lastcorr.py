import pandas as pd
from scipy.stats import spearmanr

# Load the CSV file
df = pd.read_csv('last_test_mqm.csv')

df = df.rename(columns={
    'og_score': 'comet_score',
    'ours_score': 'cometh_score',
    'c.mqm': 'claude_score',
    'g.mqm': 'gemini_score',
    'o.mqm': 'gpt_score'
})

df['domain'] = df['domain'].fillna('education')
aug_df = pd.read_csv('predict_aug.csv')
aug_df.drop_duplicates(subset=['src', 'mt'], inplace=True)

# Merge the two DataFrames on 'src' and 'mt'
df = pd.merge(df, aug_df[['src', 'mt', 'aug_cometh_score']], on=['src', 'mt'], how='left')


# #save the dataframe to a csv file
df.to_csv('im_done_bitch_mqm.csv', index=False)


# Compute Spearman correlations
mqm_columns = ['comet_score','cometh_score','aug_cometh_score','claude_score', 'gemini_score', 'gpt_score']

for col in mqm_columns:
    corr, pval = spearmanr(df['score'], df[col], nan_policy='omit')
    print(f"Spearman correlation between human and {col}: {corr:.4f}")


#print proportions of domain

# for nan values in the column "domain" replace with "education"

print("=== Proportions of each domain ===")
print(df['domain'].value_counts(normalize=True) * 100)

mqm_columns = ['score','comet_score','cometh_score','aug_cometh_score','claude_score', 'gemini_score', 'gpt_score']

# Correlation per domain
domain_corr = df.groupby('domain')[mqm_columns].corr().loc[:, 'score'].unstack()
print("=== Correlation with 'score' per domain ===")
print(domain_corr)

# # Correlation per system
# system_corr = df.groupby('system')[corr_columns].corr().loc[:, 'score'].unstack()
# print("\n=== Correlation with 'score' per system ===")
# print(system_corr)


