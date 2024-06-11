# * evaluate correctness of responses and output summary 
import pandas as pd 
import sys
import matplotlib.pyplot as plt
import numpy as np

import analysis
from visualize_triage import visualize, misclassification, error_analysis, avg_error_analysis, visualize_avg_errors, filter_data_for_conditions
from triage_zone_mapping import class_to_color, color_to_class




gpt_triage_path = "triage_experiments/datasets/results/2024-03-27_15_28_triage_results(main).csv"
mistral_triage_path = "triage_experiments/datasets/results/triage_mistral_cleaned.csv"
mixtral_triage_path = "triage_experiments/datasets/results/triage_results_mixtral_from_paper.csv"
claudeHaiku_triage_path = "triage_experiments/datasets/results/2024-04-29_19_49_triage_results_claude_haiku.csv"
claudeOpus_triage_path = "triage_experiments/datasets/results/triage_results_claude_opus_from_paper.csv"

gpt_triage_evil_path = "context_changes/datasets/triage/gpt_evil_and_no_prompt.csv"
# gpt_triage_evil_path1 = "context_changes/datasets/triage/2024-04-02_11_40_triage_results.csv"
gpt_triage_evil_path2 = "context_changes/datasets/triage/2024-04-02_11_53_triage_evil_results_gpt.csv"
mistral_triage_evil_path = "context_changes/datasets/triage/triage_mistral_evil_cleaned.csv"
mixtral_triage_evil_path = "context_changes/datasets/triage/mixtral_evil_and_no_prompt.csv"
claudeHaiku_triage_evil_path = "context_changes/datasets/triage/claude_haiku_evil_and_no_prompt.csv"
claudeOpus_triage_evil_path = "context_changes/datasets/triage/triage_evil_results_claude_opus.csv"


gpt = pd.read_csv(gpt_triage_path)
gpt35 = gpt[[col for col in gpt if "4" not in col]]
gpt4 = gpt[[col for col in gpt if "3.5" not in col]]

mistral = pd.read_csv(mistral_triage_path)
mixtral = pd.read_csv(mixtral_triage_path)
haiku = pd.read_csv(claudeHaiku_triage_path)
opus = pd.read_csv(claudeOpus_triage_path)
opus = opus[[col for col in opus.columns if not "action" in col]]


# gpt1 = pd.read_csv(gpt_triage_evil_path1)
gpt2 = pd.read_csv(gpt_triage_evil_path2)
gpt_evil = pd.read_csv(gpt_triage_evil_path)
gpt35_evil = gpt_evil[[col for col in gpt_evil if "4" not in col]]
gpt4_evil = gpt_evil[[col for col in gpt_evil if "3.5" not in col]]

mistral_evil = pd.read_csv(mistral_triage_evil_path)
mixtral_evil = pd.read_csv(mixtral_triage_evil_path)
haiku_evil = pd.read_csv(claudeHaiku_triage_evil_path)
opus_evil = pd.read_csv(claudeOpus_triage_evil_path)
opus_evil = opus_evil[[col for col in opus_evil.columns if not "outcome_oriented" in col]]


models = {
    'gpt35': [gpt35],
    'gpt4': [gpt4],
    'mistral': [mistral],
    'mixtral': [mixtral],
    'haiku': [haiku],
    'opus': [opus]
}

models = {
    'gpt35': [gpt35],
    'gpt4': [gpt4],
    'haiku': [haiku],
    'opus': [opus]
}


model_results = {}


conditions_results = {cond: {"total_errors": [], "instruction_not_followed": [], "overcaring": [], "neglecting": []} for cond in ["no_prompt", "utilitarianism", "deontology"]}

for model_name, datasets in models.items():
    aggregated_results = {key: [] for key in ['total_errors', 'instruction_not_followed', 'overcaring', 'neglecting']}
    for data in datasets:
        data['triage_zone_name'] = data['triage_zone'].apply(lambda x: color_to_class[x])
        columns = [col for col in data.columns if "answer" in col and not any(exclude in col for exclude in ['outcome', 'mad'])]  # Ensure only desired columns are used
        if columns:  # Check if columns list is not empty
            results = error_analysis(data, columns)
            for key in aggregated_results:
                aggregated_results[key].append(results[key])
        
        # Aggregate over conditions 
        for condition in conditions_results.keys(): 
          # save index of column that contains conditions_results
          condition_index = [i for i, c in enumerate(columns) if condition in c]
          for index in condition_index: 
            for error_type in ['total_errors', 'instruction_not_followed', 'overcaring', 'neglecting']:
              conditions_results[condition][error_type].append(results[error_type][index])
        
    # Compute and store the average for each error type
    model_results[model_name] = {key: np.mean(values) for key, values in aggregated_results.items() if values}  # Ensure values list is not empty
    print(model_results.keys())

# Debug print to check aggregated results before plotting
print(model_results)

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))

n_models = len(model_results)
index = np.arange(n_models)
bar_width = 0.2

visualize_avg_errors(model_results, n_models, bar_width)


# take average of errors for each condition
condition_results = {cond: {key: np.mean(values) for key, values in errors.items()} for cond, errors in conditions_results.items()}

visualize_avg_errors(condition_results, 3, bar_width)


models = {
    'gpt35': [gpt35_evil],
    'gpt4': [gpt4_evil],
    'mistral': [mistral_evil],
    'mixtral': [mixtral_evil],
    'haiku': [haiku_evil],
    'opus': [opus_evil]
}


model_results = {}

conditions_results = {cond: {"total_errors": [], "instruction_not_followed": [], "overcaring": [], "neglecting": []} for cond in ["no_prompt", "doctor_assistant", "healthcare_assistant"]}

for model_name, datasets in models.items():
    aggregated_results = {key: [] for key in ['total_errors', 'instruction_not_followed', 'overcaring', 'neglecting']}
    for data in datasets:
        data['triage_zone_name'] = data['triage_zone'].apply(lambda x: color_to_class[x])

        columns = [col for col in data.columns if "answer" in col and not any(exclude in col for exclude in ['outcome', 'mad'])]  # Ensure only desired columns are used
        if columns:  # Check if columns list is not empty
            results = error_analysis(data, columns)
            for key in aggregated_results:
                aggregated_results[key].append(results[key])
        

        # Aggregate over conditions 
        for condition in conditions_results.keys(): 
          # save index of column that contains conditions_results
          condition_index = [i for i, c in enumerate(columns) if condition in c]
          for index in condition_index: 
            for error_type in ['total_errors', 'instruction_not_followed', 'overcaring', 'neglecting']:
              conditions_results[condition][error_type].append(results[error_type][index])

    # Compute and store the average for each error type
    model_results[model_name] = {key: np.mean(values) for key, values in aggregated_results.items() if values}  # Ensure values list is not empty

# Debug print to check aggregated results before plotting
print(model_results)


n_models = len(model_results)
index = np.arange(n_models)
bar_width = 0.2

visualize_avg_errors(model_results, n_models, bar_width)

# take average of errors for each condition
condition_results = {cond: {key: np.mean(values) for key, values in errors.items()} for cond, errors in conditions_results.items()}


visualize_avg_errors(condition_results, 3, bar_width)


for df in [ gpt35, gpt4, mistral, mixtral, haiku, opus]: 
  columns = [column for column in df.columns if "answer" in column]

  # misclassification(df, columns)
  # error_analysis(df, columns, average=True)




# conditions_of_interest = ["deontology", "no prompt", "utilitarianism"]
# model_datasets = {
#     'gpt35': gpt35,
#     'gpt4': gpt4,
#     'mistral': mistral,
#     'mixtral': mixtral,
#     'haiku': haiku,
#     'opus': opus
# }

# condition_results = {cond: [] for cond in ["deontology", "no prompt", "utilitarianism"]}

# for model_name, datasets in model_datasets.items():
#     for data in datasets:
#         filtered_data = filter_data_for_conditions(data)
#         results = error_analysis(filtered_data)
        
#         for key, value in results.items():
#             if any(cond in key for cond in condition_results):
#                 condition_results[key].append(value)

# # Calculate the average error for each condition across all models
# average_results = {cond: np.mean(errors) for cond, errors in condition_results.items()}

# fig, ax = plt.subplots(figsize=(10, 6))
# bar_width = 0.3
# labels = list(average_results.keys())
# values = [average_results[label] for label in labels]

# indices = np.arange(len(labels))

# plt.bar(indices, values, bar_width, color=['blue', 'green', 'red'])
# plt.xlabel('Conditions')
# plt.ylabel('Average Error Rate')
# plt.title('Average Error Rates by Condition Across Models')
# plt.xticks(indices, labels, rotation=45)
# plt.show()

for df in [gpt35_evil, gpt4_evil, mistral_evil, mixtral_evil, haiku_evil, opus_evil]: 
  columns = [column for column in df.columns if "answer" in column and not "outcome" in column and not "mad" in column]

  # misclassification(df, columns)
  #error_analysis(df, columns, average = True)