import pandas as pd 
import analysis

import visualize_health_law


# pathname = "health_law/datasets/results/full_answer_3.5_4.csv"
# pathname = "health_law/datasets/results/health_law_answers_2024-04-03_16_23.csv"
pathname_gpt = "health_law/datasets/results/health_law_answers_2024-04-09_20_28.csv"
pathname_mistral = "health_law/datasets/results/health_law_mistral_cleaned.csv"
pathname_mixtral = "health_law/datasets/results/health_law_answers_2024-04-29_17_17.csv"
pathname_claude = "health_law/datasets/results/health_law_answers_claude_haiku_2024-04-29_21_06.csv"

df_gpt = pd.read_csv(pathname_gpt)
df_mistral = pd.read_csv(pathname_mistral)
df_mixtral = pd.read_csv(pathname_mixtral)
df_claude = pd.read_csv(pathname_claude)


# rename Unnamed Column to question_id
#drop columns
print(df_gpt.columns)
print(df_mistral.columns)
print(df_mixtral.columns)
print(df_claude.columns)

# drop all columns that have no values
df_gpt = df_gpt.dropna(axis=1, how='all')
# df_gpt = df_gpt.drop(columns=['model_reasoning_gpt-4_no_prompt', 'model_reasoning_gpt-4_utilitarian_prompts', 'model_reasoning_gpt-4_hippocratic_prompt', 'model_reasoning_gpt-3.5-turbo_no_prompt', 'model_reasoning_gpt-3.5-turbo_utilitarian_prompts', 'model_reasoning_gpt-3.5-turbo_hippocratic_prompt'])

df_gpt = df_gpt.rename(columns={'Unnamed: 0.1': 'question_id', 'Unnamed: 0': 'source_text_id', '0': 'source_text', 'model_reasoning_gpt-4_no_promptl': 'model_reasoning_gpt-4_no_prompt', 'model_reasoning_gpt-4_utilitarian_promptsl': 'model_reasoning_gpt-4_utilitarian_prompt', 'model_reasoning_gpt-4_hippocratic_promptl':'model_reasoning_gpt-4_hippocratic_prompt',
                                'model_reasoning_gpt-3.5-turbo_no_promptl': 'model_reasoning_gpt-3.5-turbo_no_prompt', 'model_reasoning_gpt-3.5-turbo_utilitarian_promptsl': 'model_reasoning_gpt-3.5-turbo_utilitarian_prompt', 'model_reasoning_gpt-3.5-turbo_hippocratic_promptl':'model_reasoning_gpt-3.5-turbo_hippocratic_prompt',})

df_mistral = df_mistral.rename(columns={'Unnamed: 0.1': 'question_id', '0': 'source_text_id', 'question_texts': 'source_text'})
df_mixtral = df_mixtral.rename(columns={'Unnamed: 0.1': 'question_id', 'Unnamed: 0': 'source_text_id', '0': 'source_text'})
df_claude = df_claude.rename(columns={'Unnamed: 0.1': 'question_id', 'Unnamed: 0': 'source_text_id', '0': 'source_text'})


merged = pd.merge(df_gpt, df_mistral, on=['question_id', 'source_text', 'source_text_id', 'questions_gpt-4', 'gold_reasoning_gpt-4', 'gold_answers_gpt-4', 'answer_options'], how='inner')
merged = pd.merge(merged, df_mixtral, on=['question_id', 'source_text', 'source_text_id', 'questions_gpt-4', 'gold_reasoning_gpt-4', 'gold_answers_gpt-4', 'answer_options'], how='inner')
merged = pd.merge(merged, df_claude, on=['question_id', 'source_text', 'source_text_id', 'questions_gpt-4', 'gold_reasoning_gpt-4', 'gold_answers_gpt-4', 'answer_options'], how='inner')


merged = merged.dropna()

merged.to_csv('health_law/datasets/results/mistral_and_gpt_models_and_cluade_merged.csv')