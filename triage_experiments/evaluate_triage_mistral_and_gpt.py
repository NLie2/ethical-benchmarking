import analysis
from visualize_triage import visualize, misclassification


# summary = analysis.analyse_triage("triage_experiments/datasets/melted_df_for_mixed_model_gpt_and_mistral.csv")
summary = analysis.analyse_triage("triage_experiments/datasets/melted_df_for_mixed_model_gpts_mistrals_claudes_from_paper.csv")

visualize(summary)

