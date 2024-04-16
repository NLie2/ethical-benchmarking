import pandas as pd 


# MISTRAL, GPT 4, AND GPT 3.5 TURBO WERE ALL TESTED ON THE PAPER CONDITION

gpt_models_path = 'triage_experiments/datasets/melted_df_for_mixed_model.csv'
mistral_path = 'triage_experiments/datasets/melted_df_for_mixed_model_mistral.csv'

gpt_models = pd.read_csv(gpt_models_path)
mistral = pd.read_csv(mistral_path)

gpt_models_paper = gpt_models[gpt_models['syntax'].str.contains('paper')]
mistral = pd.read_csv(mistral_path)

# apend msitral to gpt_models_paper
merged = pd.concat([gpt_models_paper, mistral], ignore_index=True)
# df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

merged.to_csv('triage_experiments/datasets/melted_df_for_mixed_model_gpt_and_mistral.csv')