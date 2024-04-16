# * Evaluate correctness of answers on the health_law dataset

import pandas as pd 
import analysis

import visualize_health_law


# pathname = "health_law/datasets/results/full_answer_3.5_4.csv"
# pathname = "health_law/datasets/results/health_law_answers_2024-04-03_16_23.csv"
# pathname = "/Users/nathaliekirch/THESIS/ethical-benchmarking/health_law/datasets/results/health_law_answers_2024-04-09_20_28.csv"
# pathname = "/Users/nathaliekirch/THESIS/ethical-benchmarking/health_law/datasets/results/health_law_mistral_cleaned.csv"
pathname = "/Users/nathaliekirch/THESIS/ethical-benchmarking/health_law/datasets/results/mistral_gpt_merged.csv"


df = pd.read_csv(pathname)
df.dropna()


# rename Unnamed Column to question_id
# df = df.rename(columns={'Unnamed: 0': 'question_id', '0': 'source_text'})

# melt dataframe
# melted_df = df.melt(id_vars=['question_id', 'questions_gpt-4', 'gold_answers_gpt-4', 'gold_reasoning_gpt-4', 'source_text', 'answer_options'], var_name='column', value_name='response')
melted_df = df.melt(id_vars=['question_id', 'questions_gpt-4', 'gold_answers_gpt-4', 'gold_reasoning_gpt-4', 'answer_options'], var_name='column', value_name='response')
melted_df['model'] = melted_df['column'].str.extract('(gpt-3.5|gpt-4|mistral)')
melted_df['response_type'] = melted_df['column'].str.contains('reasoning').map({True: 'reasoning', False: 'answer'})
# melted_df['prompt_type'] = melted_df['column'].str.contains('ethical').map({True: 'ethical', False: 'no_prompt'})
melted_df['prompt_type'] = melted_df['column'].str.extract('(no_prompt|utilitarian|hippocratic)')

# Filter rows with unwanted data
melted_df = melted_df.drop(columns=['column'])
melted_df = melted_df[melted_df['response_type'] == 'answer']
melted_df = melted_df[melted_df['response'].isin(['A', 'B'])]
melted_df = melted_df.dropna(subset=['response'])

# Check correctness
melted_df['correct_answer'] = melted_df['gold_answers_gpt-4'] == melted_df['response']
melted_df.dropna(subset=['model'])
# save to CSV
melted_df.to_csv('health_law/datasets/melted_df_for_mixed_model_mistral_and_gpt.csv')

# output summary
# summary = analysis.analyse_health_law('health_law/datasets/melted_df_for_mixed_model.csv')
summary = analysis.analyse_health_law('health_law/datasets/melted_df_for_mixed_model_mistral_and_gpt.csv')

# variation = analysis.check_question_variation('health_law/datasets/melted_df_for_mixed_model.csv')
variation = analysis.check_question_variation('health_law/datasets/melted_df_for_mixed_model_mistral_and_gpt.csv')

# visualize_health_law.visualize_alt(summary)
visualize_health_law.visualize(summary)