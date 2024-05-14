# * evaluate correctness of responses and output summary 
import pandas as pd 
import sys

import analysis
from visualize_triage import visualize, misclassification
from triage_zone_mapping import class_to_color, color_to_class




gpt_triage_path = "triage_experiments/datasets/results/2024-03-27_15_28_triage_results(main).csv"
mistral_triage_path = "triage_experiments/datasets/results/triage_mistral_cleaned.csv"
mixtral_triage_path = "triage_experiments/datasets/results/triage_results_mixtral_from_paper.csv"
claudeHaiku_triage_path = "triage_experiments/datasets/results/2024-04-29_19_49_triage_results_claude_haiku.csv"
claudeOpus_triage_path = "triage_experiments/datasets/results/triage_results_claude_opus_from_paper.csv"

gpt_triage_evil_path = "context_changes/datasets/gpt_evil_and_no_prompt.csv"
gpt_triage_evil_path1 = "context_changes/datasets/triage/2024-04-02_11_40_triage_results.csv"
gpt_triage_evil_path2 = "context_changes/datasets/triage/2024-04-02_11_53_triage_evil_results_gpt.csv"
mistral_triage_evil_path = "context_changes/datasets/triage/triage_mistral_evil_cleaned.csv"
mixtral_triage_evil_path = "context_changes/datasets/mixtral_evil_and_no_prompt.csv"
claudeHaiku_triage_evil_path = "context_changes/datasets/claude_haiku_evil_and_no_prompt.csv"
claudeOpus_triage_evil_path = "context_changes/datasets/triage/triage_evil_results_claude_opus.csv"


gpt = pd.read_csv(gpt_triage_path)
gpt35 = gpt[[col for col in gpt if "4" not in col]]
gpt4 = gpt[[col for col in gpt if "3.5" not in col]]

mistral = pd.read_csv(mistral_triage_path)
mixtral = pd.read_csv(mixtral_triage_path)
haiku = pd.read_csv(claudeHaiku_triage_path)
opus = pd.read_csv(claudeOpus_triage_path)
opus = opus[[col for col in opus.columns if not "action" in col]]


# for df in [ gpt35, gpt4, mistral, mixtral, haiku, opus]: 
#   columns = [column for column in df.columns if "answer" in column]

#   misclassification(df, columns)

gpt1 = pd.read_csv(gpt_triage_evil_path1)
gpt2 = pd.read_csv(gpt_triage_evil_path2)
gpt_evil = pd.read_csv(gpt_triage_evil_path)
gpt35_evil = gpt_evil[[col for col in gpt_evil if "4" not in col]]
gpt4_evil = gpt_evil[[col for col in gpt_evil if "3.5" not in col]]

mistral_evil = pd.read_csv(mistral_triage_evil_path)
mixtral_evil = pd.read_csv(mixtral_triage_evil_path)
haiku_evil = pd.read_csv(claudeHaiku_triage_evil_path)
opus_evil = pd.read_csv(claudeOpus_triage_evil_path)
opus_evil = opus_evil[[col for col in opus_evil.columns if not "outcome_oriented" in col]]


for df in [gpt35_evil, gpt4_evil, mistral_evil, mixtral_evil, haiku_evil, opus_evil]: 
  columns = [column for column in df.columns if "answer" in column and not "outcome" in column and not "mad" in column]

  misclassification(df, columns)