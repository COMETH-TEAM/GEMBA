import pandas as pd

df = pd.read_csv("im_done_bitch_mqm.csv")


# stats = df.groupby('system')['score'].agg(['mean', 'size'])
# stats['mean'] = (stats['mean'] * 100).round(3)


# stats.rename(columns={'mean': 'avg mqm score'}, inplace=True)
# stats = stats.sort_values(by='avg mqm score', ascending=False)
# print(stats)


# Step 1: Replace all typhoon variants with 'typhoon-v1.5x-70b-instruct'
df['system'] = df['system'].replace({
    'typhoon-v1.5x-70b-instruct\\t': 'typhoon-v1.5x-70b-instruct',
    'typhoon-1.5v-instruct': 'typhoon-v1.5x-70b-instruct',
})

# Step 2: Define allowed systems
allowed_systems = [
    'xai/grok-beta',
    'claude 3.5 sonnet',
    'typhoon-v1.5x-70b-instruct\t',
    'openai/gpt-4o-mini',
    'ggt-sheet',
    'aisingapore/gemma2-9b-cpt-sea-lionv3-instruct',
    'typhoon-v1.5x-70b-instruct',
    'airesearch/LLaMa3-8b-WangchanX-sft-Full',
    'Qwen/Qwen2.5-72B-Instruct',
    'openthaigpt/openthaigpt1.5-72b-instruct',
    'facebook/nllb-200-1.3B'
]


df = df[df['system'].isin(allowed_systems)]

avg_scores = df.groupby('system')['score'].mean().reset_index()
# avg_scores['score'] = (avg_scores['score'] * 100).round(3)
avg_scores['score'] = avg_scores['score'].round(4)
avg_scores.rename(columns={'score': 'avg mqm score'}, inplace=True)
avg_scores = avg_scores.sort_values(by='avg mqm score', ascending=False)
# Step 3: Print the average scores
print("=== Average MQM Scores ===") 
print(avg_scores)
