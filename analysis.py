# * evaluate correctness of responses and output summary 
import pandas as pd 
import sys

# Analyse correctness of health_law dataset
def analyse_health_law(path):
  # save to CSV
  df = pd.read_csv(path)
  
  # output summary
  summary = df.groupby(['model', 'prompt_type'])['correct_answer'].mean().reset_index(name='proportion_correct')
  # summary = df.groupby(['model', 'prompt_type'])['correct_answer'].agg(['mean', 'std']).reset_index()
  # Rename the columns for clarity
  summary.columns = ['model', 'prompt_type', 'proportion_correct']
    
  print(summary)
  return summary


# Analyse correctness of triage dataset
def analyse_triage(path):
  df = pd.read_csv(path)

  # print summary
  # summary = df.groupby(['model', 'prompt_type', 'syntax'])['correct_answer'].mean().reset_index(name='proportion_correct')
  summary = df.groupby(['model', 'prompt_type'])['correct_answer'].mean().reset_index(name='proportion_correct')

  # summary = df.groupby(['model', 'prompt_type'])['correct_answer'].agg(['mean', 'std']).reset_index()
  # summary.columns = ['model', 'prompt_type', 'syntax','proportion_correct']
  summary.columns = ['model', 'prompt_type','proportion_correct']



  print(summary)
  return summary

def check_question_variation(path): 
  df = pd.read_csv(path)
  summary = df.groupby(['question_id', 'model'])['correct_answer'].mean().reset_index(name='proportion_correct')

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
  