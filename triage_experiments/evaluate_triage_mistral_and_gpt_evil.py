# * evaluate correctness of responses and output summary 
import pandas as pd 
import sys

import analysis
from triage_experiments.visualize_triage import visualize, visualize_alt, misclassification

from triage_experiments.triage_zone_mapping import class_to_color, color_to_class

triage_evil_path_mistral = "triage_experiments/datasets/results/triage_mistral_evil_cleaned.csv"
#pathname_evil = triage_evil_path_mistral

triage_evil_path_gpt = "context_changes/datasets/gpt_evil_and_no_prompt.csv"
# pathname = triage_path_gpt

def check_match(row):
    # Get the model answer for the current gold answer from the mapping
    # Use `mapping.get(row['gold_answer'], None)` to avoid KeyError if the gold_answer is not in the mapping
    expected_model_answer = class_to_color.get(row['response'])
    # Return True if the model answer matches the expected model answer, False otherwise
    return row['triage_zone'] == expected_model_answer

def filter_columns(df):
    keywords = ['healthcare', 'doctor',  'no_prompt' ]
    conditions = ['from_paper', 'answer']
    
    filtered_cols = [
        col for col in df.columns 
        if any(keyword in col for keyword in keywords) and all(condition in col for condition in conditions)
        or col in ['question_id', 'triage_zone']
    ]
    return filtered_cols


df_mistral_evil = pd.read_csv(triage_evil_path_mistral)
df_gpt_evil = pd.read_csv(triage_evil_path_gpt)

# df = df.rename(columns={'Unnamed: 0.1': 'question_id'})
df_mistral_evil = df_mistral_evil.rename(columns={'Unnamed: 0.1': 'question_id'})
# df_gpt_evil = df_gpt_evil.rename(columns={'Unnamed: 0.1': 'question_id'})

df_mistral_evil = df_mistral_evil[filter_columns(df_mistral_evil)]
df_gpt_evil = df_gpt_evil[filter_columns(df_gpt_evil)]

# # Filter df for columns containing 'no_prompt' or 'question_id'
# df_no_prompt = df.filter(regex='no_prompt|question_id')
df = pd.merge(df_mistral_evil, df_gpt_evil, on=['question_id', 'triage_zone'], how='inner')

# df = df.rename(columns={'Unnamed: 0.1': 'question_id'})
# df = df.drop(columns=df.filter(regex='Unnamed|reasoning').columns )
# df = df.drop(columns=['question', 'action', 'class'])


# Dataframe like this has many results per row. 
# I need it to be one result per row, where model and prompt are listed as "conditions"
# Melt the DataFrame
melted_df = df.melt(id_vars=['question_id', 'triage_zone'], var_name='column', value_name='response')

# Extract information from 'column' into new columns
melted_df['model'] = melted_df['column'].str.extract('(gpt-3.5|gpt-4|mistral)')
# melted_df['syntax'] = melted_df['column'].str.extract('(paper|outcome|action)')
melted_df['prompt_type'] = melted_df['column'].str.extract('(doctor|mad|healthcare|no_prompt)')
melted_df['response_type'] = melted_df['column'].str.contains('reasoning').map({True: 'reasoning', False: 'answer'})

# Filter out rows without responses  
melted_df.dropna(subset=['response'])

# Check if the model answer matches the expected model answer
melted_df['correct_answer'] = melted_df.apply(check_match, axis=1)

# filter melted_df for rows that contain "answer" in response_tye
merged = melted_df[melted_df['response_type'] == 'answer']

# Drop columns that are no longer needed
melted_df = melted_df.drop(columns=['column', 'triage_zone', 'response', 'response_type'])

melted_df.to_csv('context_changes/datasets/melted_df_for_mixed_model_mistral_and_gpt_evil.csv')

# print summary
summary = analysis.analyse_triage('context_changes/datasets/melted_df_for_mixed_model_mistral_evil.csv')
# question_variation = analysis.check_question_variation("triage_experiments/datasets/melted_df_for_mixed_model.csv")



visualize(summary)
# visualize_alt(summary)




# print(gpt_35.head(5)["qustion_id"], gpt_4.head(5)["question_id"], paper.head(5)["question_id"], action.head(5)["question_id"], outcome.head(5)["question_id"], no_prompt.head(5)["question_id"], deontology.head(5)["question_id"], utilitarianism.head(5)["question_id"])

# merge all dataframes
# add all columns except question_id and triage_zone
columns = [column for column in df.columns if "answer" in column]
misclassification(df, columns)

