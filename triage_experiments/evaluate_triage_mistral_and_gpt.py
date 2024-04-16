import analysis
from visualize_triage import visualize, misclassification


summary = analysis.analyse_triage("triage_experiments/datasets/melted_df_for_mixed_model_gpt_and_mistral.csv")


visualize(summary)

# columns = [column for column in df.columns if column not in ["question_id", "triage_zone", 'mistralai/Mistral-7B-Instruct-v0.1_from_paper_deontology_raw', 'mistralai/Mistral-7B-Instruct-v0.1_from_paper_utilitarianism_raw', 'mistralai/Mistral-7B-Instruct-v0.1_from_paper_no_prompt_raw']]
# misclassification(df, columns)