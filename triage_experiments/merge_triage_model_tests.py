import pandas as pd 
import analysis
from triage_experiments.visualize_triage import visualize, misclassification
from triage_experiments.triage_zone_mapping import class_to_color, color_to_class


# MISTRAL, GPT 4, AND GPT 3.5 TURBO WERE ALL TESTED ON THE PAPER CONDITION

gpt35_path = 'triage_experiments/datasets/results/melted_dfs_for_mixed_models/melted_df_for_mixed_model_gpt35.csv'
gpt4_path = 'triage_experiments/datasets/results/melted_dfs_for_mixed_models/melted_df_for_mixed_model_gpt4.csv'
mistral_path = 'triage_experiments/datasets/results/melted_dfs_for_mixed_models/melted_df_for_mixed_model_mistral.csv'
mixtral_path = 'triage_experiments/datasets/results/melted_dfs_for_mixed_models/melted_df_for_mixed_model_mixtral.csv'
claude_haiku_path = 'triage_experiments/datasets/results/melted_dfs_for_mixed_models/melted_df_for_mixed_model_haiku.csv'
claude_opus_path = 'triage_experiments/datasets/results/melted_dfs_for_mixed_models/melted_df_for_mixed_model_opus.csv'



gpt35 = pd.read_csv(gpt35_path)
gpt_4 = pd.read_csv(gpt4_path)
mistral = pd.read_csv(mistral_path)
mixtral = pd.read_csv(mixtral_path)
haiku = pd.read_csv(claude_haiku_path)
opus = pd.read_csv(claude_opus_path)


# COMPARE ALL MODELS ON FROM_PAPER SYNTAX VARIATION
gpt35_paper = gpt35[gpt35['syntax'].str.contains('paper')]
gpt4_paper = gpt_4[gpt_4['syntax'].str.contains('paper')]
haiku_paper = haiku[haiku['syntax'].str.contains('paper')]
opus_paper = opus[opus['syntax'].str.contains('paper')]


# append msitral to gpt_models_paper
merged = pd.concat([gpt35_paper, gpt4_paper, mistral, mixtral, haiku_paper, opus_paper], ignore_index=True)

# df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

merged.to_csv('triage_experiments/datasets/melted_df_for_mixed_model_gpts_mistrals_claudes_from_paper.csv')

summary = analysis.analyse_triage('triage_experiments/datasets/melted_df_for_mixed_model_gpts_mistrals_claudes_from_paper.csv')

visualize(summary)

#COMPARE GPT AND CLAUDE ON ALL SYNTAX VARIATIONS
merged = pd.concat([gpt35, gpt_4, haiku], ignore_index=True)
merged.to_csv('triage_experiments/datasets/melted_df_for_mixed_model_gpts_claude_haiku.csv')


summary = analysis.analyse_triage('triage_experiments/datasets/melted_df_for_mixed_model_gpts_claude_haiku.csv')

visualize(summary)





