# * evaluate correctness of responses and output summary 
import pandas as pd 
import sys

import analysis
from triage_experiments.visualize_triage import visualize

from triage_experiments.triage_zone_mapping import class_to_color, color_to_class

gpt_triage_evil_path = "context_changes/datasets/gpt_evil_and_no_prompt.csv"
mistral_triage_evil_path = "context_changes/datasets/triage/triage_mistral_evil_cleaned.csv"
mixtral_triage_evil_path = "context_changes/datasets/mixtral_evil_and_no_prompt.csv"
claudeHaiku_triage_evil_path = "context_changes/datasets/claude_haiku_evil_and_no_prompt.csv"
claudeOpus_triage_evil_path = "context_changes/datasets/triage/triage_evil_results_claude_opus.csv"


gpt_evil = pd.read_csv(gpt_triage_evil_path)

mistral_evil = pd.read_csv(mistral_triage_evil_path)
mixtral_evil = pd.read_csv(mixtral_triage_evil_path)
haiku_evil = pd.read_csv(claudeHaiku_triage_evil_path)
opus_evil = pd.read_csv(claudeOpus_triage_evil_path)
# opus_evil = opus_evil[[col for col in opus_evil.columns if not "outcome_oriented" in col]]

def check_match(row):
    # Get the model answer for the current gold answer from the mapping
    # Use `mapping.get(row['gold_answer'], None)` to avoid KeyError if the gold_answer is not in the mapping
    expected_model_answer = class_to_color.get(row['response'])
    # Return True if the model answer matches the expected model answer, False otherwise
    return row['triage_zone'] == expected_model_answer

opus_evil = opus_evil.rename(columns={'Unnamed: 0.1': 'question_id', '0': 'source_text_id', 'question_texts': 'source_text'})
# mixtral_evil = mixtral_evil.rename(columns={'Unnamed: 0.1': 'question_id', '0': 'source_text_id', 'question_texts': 'source_text'})
mistral_evil = mistral_evil.rename(columns={'Unnamed: 0.1': 'question_id', '0': 'source_text_id', 'question_texts': 'source_text'})

df_all_syntax = pd.merge(gpt_evil, haiku_evil, on=['question_id', 'question', 'triage_zone', 'class', 'action'], how='inner')
df_all_syntax = pd.merge(df_all_syntax, opus_evil, on=['question_id', 'question', 'triage_zone', 'class', 'action'], how='inner')

df_all_syntax = df_all_syntax[[column for column in df_all_syntax.columns if not "outcome" in column and not "mad" in column]]

df_all_syntax = df_all_syntax.drop(columns= df_all_syntax.filter(regex='Unnamed|reasoning').columns )

df_from_paper = df_all_syntax[[col for col in df_all_syntax.columns if "action_oriented" not in col]]

for df in [mixtral_evil, mistral_evil]:
  df_from_paper = df_from_paper.drop(columns= df_from_paper.filter(regex='Unnamed|reasoning').columns )
  df_from_paper = pd.merge(df_from_paper, df, on=['question_id', 'question', 'triage_zone', 'class', 'action'], how='inner')
  df_from_paper = df_from_paper[[column for column in df_from_paper.columns if not "mad" in column and not "raw" in column and not "reasoning" in column]]



print(len(df_all_syntax))
# df_all_syntax = df_all_syntax.drop(columns=['question', 'action', 'class'])

df_from_paper = df_from_paper.drop(columns= df_from_paper.filter(regex='Unnamed').columns )
df_from_paper = df_from_paper.drop(columns=['question', 'action', 'class'])

df_all_syntax = df_all_syntax.drop(columns= df_all_syntax.filter(regex='Unnamed').columns )
df_all_syntax = df_all_syntax.drop(columns=['question', 'action', 'class'])


# Dataframe like this has many results per row. 
# I need it to be one result per row, where model and prompt are listed as "conditions"
# Melt the DataFrame

df = df_all_syntax
# df = df_from_paper


melted_df = df.melt(id_vars=['question_id', 'triage_zone'], var_name='column', value_name='response')

# Extract information from 'column' into new columns
melted_df['model'] = melted_df['column'].str.extract('(gpt-3.5|gpt-4|haiku|opus|mistral|mixtral)')
melted_df['syntax'] = melted_df['column'].str.extract('(paper|outcome|action)')
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


melted_df.to_csv('context_changes/datasets/melted_df_for_mixed_model_triage_evil_all_syntax.csv')
# melted_df.to_csv('context_changes/datasets/melted_df_for_mixed_model_triage_evil_from_paper.csv')

# print summary
# summary = analysis.analyse_triage('context_changes/datasets/melted_df_for_mixed_model_triage_evil_from_paper.csv')
summary = analysis.analyse_triage('context_changes/datasets/melted_df_for_mixed_model_triage_evil_all_syntax.csv')


visualize(summary)


