# * Evaluate correctness of answers on the health_law dataset

import pandas as pd 
import analysis

import visualize_health_law


# pathname = "health_law/datasets/results/full_answer_3.5_4.csv"
# pathname = "health_law/datasets/results/health_law_answers_2024-04-03_16_23.csv"
pathname = "context_changes/datasets/health_law/health_law_answers_evil_2024-04-11_23_17.csv"
pathname_no_prompt = "health_law/datasets/results/health_law_answers_2024-04-09_20_28.csv"
df = pd.read_csv(pathname)
df_no_prompt = pd.read_csv(pathname_no_prompt)

df.dropna()
df_no_prompt.dropna()


# rename Unnamed Column to question_id
df = df.rename(columns={'Unnamed: 0': 'question_id', '0': 'source_text'})
df_no_prompt = df_no_prompt.rename(columns={'Unnamed: 0': 'question_id', '0': 'source_text'})
df_no_prompt = df_no_prompt.filter(regex='no_prompt|question_id')

df = pd.merge(df, df_no_prompt, on='question_id', how='inner')

# df = df.rename(columns={'Unnamed: 0.1': 'question_id'})
# df = df.drop(columns=df.filter(regex='Unnamed|reasoning').columns )
# df = df.drop(columns=['question', 'action', 'class'])


# melt dataframe
melted_df = df.melt(id_vars=['question_id', 'questions_gpt-4', 'gold_answers_gpt-4', 'gold_reasoning_gpt-4', 'source_text', 'answer_options'], var_name='column', value_name='response')
melted_df['model'] = melted_df['column'].str.extract('(gpt-3.5|gpt-4)')
melted_df['response_type'] = melted_df['column'].str.contains('reasoning').map({True: 'reasoning', False: 'answer'})
# melted_df['prompt_type'] = melted_df['column'].str.contains('ethical').map({True: 'ethical', False: 'no_prompt'})
melted_df['prompt_type'] = melted_df['column'].str.extract('(mad_scientist|healthcare_assistant|doctor|no_prompt)')

# Filter rows with unwanted data
melted_df = melted_df.drop(columns=['column'])
melted_df = melted_df[melted_df['response_type'] == 'answer']
melted_df = melted_df[melted_df['response'].isin(['A', 'B'])]
melted_df = melted_df.dropna(subset=['response'])

# Check correctness
melted_df['correct_answer'] = melted_df['gold_answers_gpt-4'] == melted_df['response']
melted_df.dropna(subset=['model'])
# save to CSV
melted_df.to_csv('context_changes/datasets/health_law/melted_df_for_mixed_model.csv')

# output summary
summary = analysis.analyse_health_law('context_changes/datasets/health_law/melted_df_for_mixed_model.csv')

variation = analysis.check_question_variation('context_changes/datasets/health_law/melted_df_for_mixed_model.csv')

visualize_health_law.visualize(summary)