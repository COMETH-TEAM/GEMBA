import pandas as pd

# Load the CSV files
test_df = pd.read_csv("predict_cometh.csv")
claude_cometh_df = pd.read_csv("COMETH_DATASET_CLAUDE.csv")
gpt_gemini_df = pd.read_csv("cleaned_gpt_gemini_claude_mqm.csv")
not_null_df = pd.read_csv("not_null_rows_2.csv")

gpt_gemini_df.rename(columns={
    "sourceText": "src",
    "translatedText": "mt",
}, inplace=True)

claude_cometh_df = claude_cometh_df.rename(columns={
    "minor": "c.minor",
    "major": "c.major",
    "critical": "c.critical"
})

#size of the dataframes
print("Test shape:", test_df.shape)

claude_cometh_df.drop_duplicates(subset=["src", "mt"], inplace=True)
gpt_gemini_df.drop_duplicates(subset=["src", "mt"], inplace=True)

merged_df = pd.merge(
    test_df,
    claude_cometh_df,
    on=["src", "mt"],
    how="left"
)

merged_df = pd.merge(
    merged_df,
    gpt_gemini_df[['src','mt','domain', 'critical', 'major', 'minor', 'system', 'g.critical', 'g.major', 'g.minor','o.critical', 'o.major', 'o.minor']],
    on=["src", "mt"],
    how="left"
)

# # print gpt_gemin_df column "id"  value =68
# print(gpt_gemini_df[gpt_gemini_df['id'] == 14].isnull().any())

# #save rows that have null values in the column "g.critical" 
# null_df = merged_df[merged_df['g.critical'].isnull()]
# null_df.to_csv("null_rows_2.csv", index=False)


#drop rows that have null values in the column "g.critical"
merged_df = merged_df.dropna(subset=['g.critical'])
merged_df = pd.concat([merged_df, not_null_df], ignore_index=True)

# Define weights
weights = {'critical': 10, 'major': 5, 'minor': 1}

# Sentence length (in characters)
merged_df['src_len'] = merged_df['src'].apply(lambda x: len(x.split()))

# Overall MQM score
merged_df['mqm'] = 1 - (
    (merged_df['critical'] * weights['critical']
     + merged_df['major'] * weights['major']
     + merged_df['minor'] * weights['minor'])
    / merged_df['src_len']
)

# Category-wise MQM scores
merged_df['c.mqm'] = 1 - (
    (merged_df['c.critical'] * weights['critical']
     + merged_df['c.major'] * weights['major']
     + merged_df['c.minor'] * weights['minor'])
    / merged_df['src_len']
)

merged_df['g.mqm'] = 1 - (
    (merged_df['g.critical'] * weights['critical']
     + merged_df['g.major'] * weights['major']
     + merged_df['g.minor'] * weights['minor'])
    / merged_df['src_len']
)

merged_df['o.mqm'] = 1 - (
    (merged_df['o.critical'] * weights['critical']
     + merged_df['o.major'] * weights['major']
     + merged_df['o.minor'] * weights['minor'])
    / merged_df['src_len']
)

# each mqm score cannot be less than 0
merged_df['mqm'] = merged_df['mqm'].clip(lower=0)
merged_df['c.mqm'] = merged_df['c.mqm'].clip(lower=0)
merged_df['g.mqm'] = merged_df['g.mqm'].clip(lower=0)
merged_df['o.mqm'] = merged_df['o.mqm'].clip(lower=0)

#savw csv
merged_df.to_csv("last_test_mqm.csv", index=False)
