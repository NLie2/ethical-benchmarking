# * Evaluate correctness of answers on the health_law dataset

import pandas as pd 
import analysis


pathname = "health_law/datasets/full_answer_3.5_4.csv"
df = pd.read_csv(pathname)


# rename Unnamed Column to question_id
df = df.rename(columns={'Unnamed: 0': 'question_id', '0': 'source_text'})

# melt dataframe
melted_df = df.melt(id_vars=['question_id', 'questions_gpt-4', 'gold_answers_gpt-4', 'gold_reasoning_gpt-4', 'source_text', 'answer_options'], var_name='column', value_name='response')
melted_df['model'] = melted_df['column'].str.extract('(gpt-3.5|gpt-4)')
melted_df['response_type'] = melted_df['column'].str.contains('reasoning').map({True: 'reasoning', False: 'answer'})
melted_df['prompt_type'] = "no_prompt" 

# Filter rows with unwanted data
melted_df = melted_df.drop(columns=['column'])
melted_df = melted_df[melted_df['response_type'] == 'answer']
melted_df = melted_df.dropna(subset=['response'])

# Check correctness
melted_df['correct_answer'] = melted_df['gold_answers_gpt-4'] == melted_df['response']

# save to CSV
melted_df.to_csv('health_law/datasets/melted_df_for_mixed_model.csv')

# output summary
analysis.analyse_health_law()