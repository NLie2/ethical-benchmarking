# * evaluate correctness of responses and output summary 
import pandas as pd 
import sys

import analysis
from triage_experiments.visualize_triage import visualize, misclassification

from triage_experiments.triage_zone_mapping import class_to_color, color_to_class

triage_evil_path_mistral = "context_changes/datasets/triage/triage_mistral_evil_cleaned.csv"
#pathname_evil = triage_evil_path_mistral
triage_evil_path_gpt = "context_changes/datasets/triage/gpt_evil_and_no_prompt.csv"
# pathname = triage_path_gpt
triage_evil_path_mixtral = "context_changes/datasets/triage/mixtral_evil_and_no_prompt.csv"
triage_evil_path_claude = "context_changes/datasets/triage/claude_haiku_evil_and_no_prompt.csv"

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
df_mixtral_evil = pd.read_csv(triage_evil_path_mixtral)
df_gpt_evil = pd.read_csv(triage_evil_path_gpt)
df_claude_evil = pd.read_csv(triage_evil_path_claude)


# df = df.rename(columns={'Unnamed: 0.1': 'question_id'})
df_mistral_evil = df_mistral_evil.rename(columns={'Unnamed: 0.1': 'question_id'})
# df_gpt_evil = df_gpt_evil.rename(columns={'Unnamed: 0.1': 'question_id'})
# df_mixtral_evil = df_mixtral_evil.rename(columns={'Unnamed: 0': 'question_id'})


df_mistral_evil = df_mistral_evil[filter_columns(df_mistral_evil)]
df_mixtral_evil = df_mixtral_evil[filter_columns(df_mixtral_evil)]
df_gpt_evil = df_gpt_evil[filter_columns(df_gpt_evil)]
df_claude_evil = df_claude_evil[filter_columns(df_claude_evil)]


# # Filter df for columns containing 'no_prompt' or 'question_id'
# df_no_prompt = df.filter(regex='no_prompt|question_id')
# df = pd.merge([df_mistral_evil, df_mixtral_evil, df_gpt_evil], on=['question_id', 'triage_zone'], how='inner')
df = pd.merge(df_mistral_evil, df_mixtral_evil, on=['question_id', 'triage_zone'], how='inner')
df = pd.merge(df, df_gpt_evil, on=['question_id', 'triage_zone'], how='inner')
df = pd.merge(df, df_claude_evil, on=['question_id', 'triage_zone'], how='inner')


# DELETE ALL ROWS WHERE A COLUMN CONTAINING THE WORD "answer" IS NOT IN ["IMMEDIATE", "DELAYED", "MINOR", "MAJOR", "EXPECTANT/DECEASED"]
# answer_cols = [col for col in df.columns if 'answer' in col]
# for col in answer_cols:
#     print(col)
#     words = ["IMMEDIATE", "immediate", "DELAYED", "delayed", "MINOR", "minor", "EXPECTANT/DECEASED", "expectant/deceased" ]

#     # Add a row in the dataframe that extracts the words ["IMMEDIATE", "DELAYED", "MINOR", "MAJOR", "EXPECTANT/DECEASED"] from the column
#     df[col] = df[col].apply(lambda x: [word for word in words if word in str(x)][0] if isinstance(x, str) and len([word for word in words if word in x]) == 1 else None)    
#     print(len(df[df[col].isin(["IMMEDIATE", "immediate", "DELAYED", "delayed", "MINOR", "minor", "EXPECTANT/DECEASED", "expectant/deceased" ])]))

# Filter to only columns that contain "from_paper"
from_paper_cols = [col for col in df.columns if 'action' not in col and 'outcome' not in col]
df = df[from_paper_cols]
df = df.dropna()
# print(len(df))


# df = df.rename(columns={'Unnamed: 0.1': 'question_id'})
# df = df.drop(columns=df.filter(regex='Unnamed|reasoning').columns )
# df = df.drop(columns=['question', 'action', 'class'])


# Dataframe like this has many results per row. 
# I need it to be one result per row, where model and prompt are listed as "conditions"
# Melt the DataFrame
melted_df = df.melt(id_vars=['question_id', 'triage_zone'], var_name='column', value_name='response')

# Extract information from 'column' into new columns
melted_df['model'] = melted_df['column'].str.extract('(gpt-3.5|gpt-4|mistral|mixtral|claude)')
# melted_df['syntax'] = melted_df['column'].str.extract('(paper|outcome|action)')
melted_df['prompt_type'] = melted_df['column'].str.extract('(doctor|mad|healthcare|no_prompt)')
melted_df['response_type'] = melted_df['column'].str.contains('reasoning').map({True: 'reasoning', False: 'answer'})

# Filter out rows without responses  
melted_df.dropna(subset=['response'])

# Check if the model answer matches the expected model answer
# Questions that are not answered in the right format will be interpreted as FALSE 
melted_df['correct_answer'] = melted_df.apply(check_match, axis=1)

# filter melted_df for rows that contain "answer" in response_tye
merged = melted_df[melted_df['response_type'] == 'answer']

# Drop columns that are no longer needed
melted_df = melted_df.drop(columns=['column', 'triage_zone', 'response', 'response_type'])

melted_df.to_csv('context_changes/datasets/melted_df_for_mixed_model_triage_evil_mistral_gpt_claude.csv')

# print summary
summary = analysis.analyse_triage('context_changes/datasets/melted_df_for_mixed_model_triage_evil_mistral_gpt_claude.csv')
# question_variation = analysis.check_question_variation("triage_experiments/datasets/melted_df_for_mixed_model.csv")



visualize(summary)
# visualize_alt(summary)




# print(gpt_35.head(5)["qustion_id"], gpt_4.head(5)["question_id"], paper.head(5)["question_id"], action.head(5)["question_id"], outcome.head(5)["question_id"], no_prompt.head(5)["question_id"], deontology.head(5)["question_id"], utilitarianism.head(5)["question_id"])

# merge all dataframes
# add all columns except question_id and triage_zone
columns = [column for column in df.columns if "answer" in column]
misclassification(df, columns)

