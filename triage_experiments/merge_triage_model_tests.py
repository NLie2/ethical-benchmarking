import pandas as pd 
import analysis
from triage_experiments.visualize_triage import visualize, visualize_alt, misclassification
from triage_experiments.triage_zone_mapping import class_to_color, color_to_class


# MISTRAL, GPT 4, AND GPT 3.5 TURBO WERE ALL TESTED ON THE PAPER CONDITION

gpt_models_path = 'triage_experiments/datasets/melted_df_for_mixed_model.csv'
mistral_path = 'triage_experiments/datasets/melted_df_for_mixed_model_mistral.csv'
mixtral_path = 'triage_experiments/datasets/melted_df_for_mixed_model_mixtral.csv'
claude_haiku_path = 'triage_experiments/datasets/melted_df_for_mixed_model_claude_haiku.csv'
claude_opus_path = 'triage_experiments/datasets/melted_df_for_mixed_model_claude_opus.csv'


gpt_models = pd.read_csv(gpt_models_path)
mistral = pd.read_csv(mistral_path)
mixtral = pd.read_csv(mixtral_path)
haiku = pd.read_csv(claude_haiku_path)
opus = pd.read_csv(claude_opus_path)


# COMPARE ALL MODELS ON FROM_PAPER SYNTAX VARIATION
gpt_models_paper = gpt_models[gpt_models['syntax'].str.contains('paper')]
haiku_paper = haiku[haiku['syntax'].str.contains('paper')]
opus_paper = opus[opus['syntax'].str.contains('paper')]


# append msitral to gpt_models_paper
merged = pd.concat([gpt_models_paper, mistral, mixtral, haiku_paper, opus_paper], ignore_index=True)

# df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

merged.to_csv('triage_experiments/datasets/melted_df_for_mixed_model_gpts_mistrals_claudes_from_paper.csv')

summary = analysis.analyse_triage('triage_experiments/datasets/melted_df_for_mixed_model_gpts_mistrals_claudes_from_paper.csv')

visualize(summary)

#COMPARE GPT AND CLAUDE ON ALL SYNTAX VARIATIONS
merged = pd.concat([gpt_models, haiku], ignore_index=True)
merged.to_csv('triage_experiments/datasets/melted_df_for_mixed_model_gpts_claude_haiku.csv')


summary = analysis.analyse_triage('triage_experiments/datasets/melted_df_for_mixed_model_gpts_claude_haiku.csv')

visualize(summary)





