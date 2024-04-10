# * evaluate correctness of responses and output summary 
import pandas as pd 
import sys

import analysis
from visualize_triage import visualize, misclassification
from triage_zone_mapping import class_to_color, color_to_class


triage_path = "/Users/nathaliekirch/THESIS/ethical-benchmarking/triage_experiments/datasets/results/2024-03-27_15_28_triage_results.csv"

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
melted_df['model'] = melted_df['column'].str.extract('(gpt-3.5|gpt-4)')
melted_df['syntax'] = melted_df['column'].str.extract('(paper|outcome|action)')
melted_df['prompt_type'] = melted_df['column'].str.extract('(no_prompt|deontology|utilitarianism)')
melted_df['response_type'] = melted_df['column'].str.contains('reasoning').map({True: 'reasoning', False: 'answer'})

# Filter out rows without responses  
melted_df.dropna(subset=['response'])

# Check if the model answer matches the expected model answer
melted_df['correct_answer'] = melted_df.apply(check_match, axis=1)

# filter melted_df for rows that contain "answer" in response_tye
merged = melted_df[melted_df['response_type'] == 'answer']

# Drop columns that are no longer needed
melted_df = melted_df.drop(columns=['column', 'triage_zone', 'response', 'response_type'])

melted_df.to_csv('triage_experiments/datasets/melted_df_for_mixed_model.csv')

# print summary
summary = analysis.analyse_triage()
question_variation = analysis.check_question_variation("triage_experiments/datasets/melted_df_for_mixed_model.csv")



visualize(summary)



# create separate dataframes for each model
gpt_35 = merged[merged['model'] == 'gpt-3.5']
gpt_4 = merged[merged['model'] == 'gpt-4']
#create separate dataframe for each syntax
paper = merged[merged['syntax'] == 'paper']
outcome = merged[merged['syntax'] == 'outcome']
action = merged[merged['syntax'] == 'action']
# create separate dataframe for each prompt type
no_prompt = merged[melted_df['prompt_type'] == 'no_prompt']
deontology = melted_df[merged['prompt_type'] == 'deontology']
utilitarianism = merged[merged['prompt_type'] == 'utilitarianism']

# print first 5 rows of gpt_35 dataframe
print(gpt_4.head(5))
# print(gpt_35.head(5)["qustion_id"], gpt_4.head(5)["question_id"], paper.head(5)["question_id"], action.head(5)["question_id"], outcome.head(5)["question_id"], no_prompt.head(5)["question_id"], deontology.head(5)["question_id"], utilitarianism.head(5)["question_id"])

# merge all dataframes
merged = pd.merge(gpt_35, gpt_4, paper, action, outcome, no_prompt,deontology, utilitarianism, on='question_id')
misclassification(merged, merged.columns)

