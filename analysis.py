# * evaluate correctness of responses and output summary 
import pandas as pd 
import sys

# Analyse correctness of health_law dataset
def analyse_health_law():
  # save to CSV
  df = pd.read_csv('health_law/datasets/melted_df_for_mixed_model.csv')
  
  # output summary
  summary = df.groupby(['model', 'prompt_type'])['correct_answer'].mean().reset_index(name='proportion_correct')
  print(summary)


# Analyse correctness of triage dataset
def analyse_triage():
  df = pd.read_csv('triage_experiments/datasets/melted_df_for_mixed_model.csv')

  # print summary
  summary = df.groupby(['model', 'prompt_type', 'syntax'])['correct_answer'].mean().reset_index(name='proportion_correct')
  print(summary)
  return summary




# make melt-datframe function callable as main function
if __name__ == "__main__":
    if len(sys.argv) > 1:
        function_name = sys.argv[1]
        if function_name == "analyse_triage" and len(sys.argv) == 2:
            analyse_triage()
        elif function_name == "analyse_health_law" and len(sys.argv) == 2:
            analyse_health_law()
        else:
            print("Invalid function name or arguments")
    else:
        print("No function name provided")
  