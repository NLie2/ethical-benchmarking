import analysis
from visualize_triage import visualize, misclassification


# summary = analysis.analyse_triage("triage_experiments/datasets/melted_df_for_mixed_model_gpt_and_mistral.csv")
summary = analysis.analyse_triage("/Users/nathaliekirch/THESIS/ethical-benchmarking/triage_experiments/datasets/melted_df_for_mixed_model_gpt_and_mistral_and_mixtral.csv")

visualize(summary)

