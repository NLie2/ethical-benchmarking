# * evaluate correctness of responses and output summary 
import pandas as pd 
import sys

import analysis
from triage_zone_mapping import class_to_color

triage_path = "triage_experiments/datasets/full_set_with_answers_gpt3.5_gpt4.csv"

pathname = triage_path

def check_match(row):
    # Get the model answer for the current gold answer from the mapping
    # Use `mapping.get(row['gold_answer'], None)` to avoid KeyError if the gold_answer is not in the mapping
    expected_model_answer = class_to_color.get(row['response'])
    # Return True if the model answer matches the expected model answer, False otherwise
    return row['triage_zone'] == expected_model_answer



df = pd.read_csv(pathname)

df = df.rename(columns={'Unnamed: 0.1': 'question_id'})
df = df.drop(columns=df.filter(regex='Unnamed|reasoning').columns )
df = df.drop(columns=['question', 'action', 'class'])

# Dataframe like this has many results per row. 
# I need it to be one result per row, where model and prompt are listed as "conditions"
# Melt the DataFrame
melted_df = df.melt(id_vars=['question_id', 'triage_zone'], var_name='column', value_name='response')

# Extract information from 'column' into new columns
melted_df['model'] = melted_df['column'].str.extract('(gpt35|gpt4)')
melted_df['syntax'] = melted_df['column'].str.extract('(paper|outcome|action)')
melted_df['prompt_type'] = melted_df['column'].str.contains('machEthics').map({True: 'machEthics', False: 'CoT'})
melted_df['response_type'] = melted_df['column'].str.contains('reasoning').map({True: 'reasoning', False: 'answer'})

# Filter out rows without responses  
melted_df.dropna(subset=['response'])

# Check if the model answer matches the expected model answer
melted_df['correct_answer'] = melted_df.apply(check_match, axis=1)

# Drop columns that are no longer needed
melted_df = melted_df.drop(columns=['column', 'triage_zone', 'response', 'response_type'])

melted_df.to_csv('triage_experiments/datasets/melted_df_for_mixed_model.csv')

# print summary
analysis.analyse_triage()