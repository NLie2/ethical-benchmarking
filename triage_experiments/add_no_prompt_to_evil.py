import pandas as pd 

triage_evil_path_gpt = "triage_experiments/datasets/results/2024-04-02_11_53_triage_results.csv"
triage_evil_path_haiku = "triage_experiments/datasets/results/2024-04-29_21_56_triage_evil_results_claude_haiku.csv"

triage_path_gpt = "triage_experiments/datasets/results/2024-03-27_15_28_triage_results(main).csv"
triage_path_claude = "triage_experiments/datasets/results/2024-04-29_19_49_triage_results_claude_haiku.csv"
triage_path_claude = "triage_experiments/datasets/results/2024-04-29_19_49_triage_results_claude_opus.csv"


df_gpt = pd.read_csv(triage_path_gpt)
df_gpt_evil = pd.read_csv(triage_evil_path_gpt)

df_claude = pd.read_csv(triage_path_claude)
df_claude_evil = pd.read_csv(triage_evil_path_haiku)

df_gpt = df_gpt.rename(columns={'Unnamed: 0.1': 'question_id'})
df_gpt_evil = df_gpt_evil.rename(columns={'Unnamed: 0.1': 'question_id'})

df_claude = df_claude.rename(columns={'Unnamed: 0.1': 'question_id'})
df_claude_evil = df_claude_evil.rename(columns={'Unnamed: 0.1': 'question_id'})

# Filter df for columns containing 'no_prompt' or 'question_id'
df_gpt_no_prompt = df_gpt.filter(regex='no_prompt|question_id')
df_claude_no_prompt = df_claude.filter(regex='no_prompt|question_id')

df_gpt_evil = pd.merge(df_gpt_evil, df_gpt_no_prompt, on='question_id', how='inner')
df_claude_evil = pd.merge(df_claude_evil, df_claude_no_prompt, on='question_id', how='inner')

df_gpt_evil.to_csv('context_changes/datasets/gpt_evil_and_no_prompt.csv')
df_claude_evil.to_csv('context_changes/datasets/claude_haiku_evil_and_no_prompt.csv')
