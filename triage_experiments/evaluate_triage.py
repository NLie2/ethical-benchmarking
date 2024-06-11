# * evaluate correctness of responses and output summary 
import pandas as pd 
import sys

import analysis
from visualize_triage import visualize, misclassification
from triage_zone_mapping import class_to_color, color_to_class


# gpt_triage_path = "/Users/nathaliekirch/THESIS/ethical-benchmarking/triage_experiments/datasets/results/2024-03-27_15_28_triage_results(main).csv"
# mistral_triage_path = "triage_experiments/datasets/results/triage_mistral_cleaned.csv"
# mixtral_triage_path = "triage_experiments/datasets/results/triage_results_mixtral_from_paper.csv"
# haiku_triage_path = "triage_experiments/datasets/results/2024-04-29_19_49_triage_results_claude_haiku.csv"
# opus_triage_path = "triage_experiments/datasets/results/triage_results_claude_opus_from_paper.csv"


gpt_triage_path = "triage_experiments/datasets/results/2024-03-27_15_28_triage_results(main).csv"
mistral_triage_path = "triage_experiments/datasets/results/triage_mistral_cleaned.csv"
mixtral_triage_path = "triage_experiments/datasets/results/triage_results_mixtral_from_paper.csv"
claudeHaiku_triage_path = "triage_experiments/datasets/results/2024-04-29_19_49_triage_results_claude_haiku.csv"
claudeOpus_triage_path = "triage_experiments/datasets/results/triage_results_claude_opus_from_paper.csv"

gpt = pd.read_csv(gpt_triage_path)
gpt35 = gpt[[col for col in gpt if "4" not in col]]
gpt4 = gpt[[col for col in gpt if "3.5" not in col]]

mistral = pd.read_csv(mistral_triage_path)
mixtral = pd.read_csv(mixtral_triage_path)
haiku = pd.read_csv(claudeHaiku_triage_path)
opus = pd.read_csv(claudeOpus_triage_path)
opus = opus[[col for col in opus.columns if not "action" in col]]

# pathname = opus_triage_path

models = {
    'gpt35': gpt35,
    'gpt4': gpt4,
    'mistral': mistral,
    'mixtral': mixtral,
    'haiku': haiku,
    'opus': opus
}
def check_match(row):
    # Get the model answer for the current gold answer from the mapping
    # Use `mapping.get(row['gold_answer'], None)` to avoid KeyError if the gold_answer is not in the mapping
    expected_model_answer = class_to_color.get(row['response'])
    # Return True if the model answer matches the expected model answer, False otherwise
    return row['triage_zone'] == expected_model_answer



# df = pd.read_csv(pathname)

for name, df in models.items(): 
  
  # only rename if question_id is not present in colnames
  if 'question_id' not in df.columns:
    df = df.rename(columns={'Unnamed: 0.1': 'question_id'})
  df = df.drop(columns=df.filter(regex='Unnamed|reasoning').columns )
  #df = df.drop(columns=['question', 'action', 'class'])


  # Dataframe like this has many results per row. 
  # I need it to be one result per row, where model and prompt are listed as "conditions"
  # Melt the DataFrame
  melted_df = df.melt(id_vars=['question_id', 'triage_zone'], var_name='column', value_name='response')

  # Extract information from 'column' into new columns
  melted_df['model'] = melted_df['column'].str.extract('(gpt-3.5|gpt-4|Mistral|mixtral|haiku|opus)')
  melted_df['syntax'] = melted_df['column'].str.extract('(paper|outcome|action)')
  melted_df['prompt_type'] = melted_df['column'].str.extract('(no_prompt|deontology|utilitarianism)')
  melted_df['response_type'] = melted_df['column'].str.extract('(reasoning|answer|raw)')
  # melted_df['response_type'] = melted_df['column'].str.contains('reasoning').map({True: 'reasoning', False: 'answer'})

  # Filter out rows without responses  
  melted_df.dropna(subset=['response'])

  # Check if the model answer matches the expected model answer
  melted_df['correct_answer'] = melted_df.apply(check_match, axis=1)

  # filter melted_df for rows that contain "answer" in response_tye
  melted_df = melted_df[melted_df['response_type'] == 'answer']


  # Drop columns that are no longer needed
  melted_df = melted_df.drop(columns=['column', 'triage_zone', 'response', 'response_type'])

  melted_df.to_csv(f'triage_experiments/datasets/results/melted_dfs_for_mixed_models/melted_df_for_mixed_model_{name}.csv')

  # print summary
  # summary = analysis.analyse_triage("triage_experiments/datasets/melted_df_for_mixed_model_claude_opus.csv")
  summary = analysis.analyse_triage(f'triage_experiments/datasets/results/melted_dfs_for_mixed_models/melted_df_for_mixed_model_{name}.csv')


  # question_variation = analysis.check_question_variation("triage_experiments/datasets/melted_df_for_mixed_model.csv")



  visualize(summary)


