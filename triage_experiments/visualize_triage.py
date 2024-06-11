import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from triage_zone_mapping import class_to_color, color_to_class
import math
import pandas as pd


labels = ["MINOR", "EXPECTANT/DECEASED", "DELAYED", "IMMEDIATE"]

# Pivot the DataFrame for the heatmap
def visualize(summary):
    # First, calculate the mean proportion correct for each model across all prompt types if necessary
    # This assumes summary already includes the 'proportion_correct' data aggregated as needed
    model_means = summary.groupby('model')['proportion_correct'].mean().sort_values(ascending=True)

    # Use the calculated means to sort the index of the pivot table
    # Sorting is done before pivot to ensure the heatmap uses this order
    summary['model'] = pd.Categorical(summary['model'], categories=model_means.index, ordered=True)
    pivot_df = summary.pivot_table(index='model', columns='prompt_type', values='proportion_correct')

    plt.figure(figsize=(12, 10))
    # color pattern should be red to green

    sns.heatmap(pivot_df, annot=True, cmap='Greens', fmt=".2f", vmin=0.25, vmax= 1)
    plt.title('Proportion of Correct Answers by Model and Ethics Prompt Type')
    plt.xlabel('Ethics Prompt')
    plt.ylabel('Model')
    plt.show()

def heatmap(data, title, ax):
    # Assuming 'data' is the confusion matrix and 'title' is the title for the heatmap
    # 'ax' is the axes object where the heatmap will be drawn
    sns.heatmap(data, annot=True, fmt="g", ax=ax, cmap="Reds", vmin=0, vmax=0.25, xticklabels=["m", "e", "d", "i"], yticklabels=["m", "e", "d", "i"])
    ax.set_title(title, fontsize = 7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')


def misclassification(dataset, categories):
  num_categories = len(categories)
  cols = int(math.ceil(math.sqrt(num_categories)))  # Square root strategy for columns
  rows = int(math.ceil(num_categories / cols))  # Calculate rows based on columns

  category_order = ['healthcare', 'doctor', 'no_prompt']
  # Filter and order categories according to the predefined list
  categories = sorted(categories, key=lambda x: next((i for i, target in enumerate(category_order) if target in x), float('inf')))

  category_order = ['from_paper', 'action_oriented']
  # Filter and order categories according to the predefined list
  categories = sorted(categories, key=lambda x: next((i for i, target in enumerate(category_order) if target in x), float('inf')))

  print(categories)


  fig, axs = plt.subplots(rows, cols, figsize=(20, 15))  # Adjust the figure size as needed

  x, y = 0, 0
  print(categories)

  # Create a confusion matrix
  dataset["triage_zone_name"] = [color_to_class[zone] for zone in dataset["triage_zone"]]
  # remove non values
  dataset = dataset.dropna()

  # Make this a 2x6 grid
  num_categories = len(categories) 
  cols = int(math.ceil(math.sqrt(num_categories)))  # Square root strategy for columns
  rows = int(math.ceil(num_categories / cols))  # Calculate rows based on columns

  fig, axs = plt.subplots(rows, cols, figsize=(20, 15))
    
  # Ensure axs is a 2D array for consistency
  if num_categories == 1:
        axs = np.array([[axs]])
  elif cols == 1 or rows == 1:
        axs = np.reshape(axs, (rows, cols))

  print(categories)

  for i, category in enumerate(categories):

    print(category)
    print(i)
    # If entry in datset[category] is not in datast[triage_zone_name] then it should be classified as "other"
    # dataset[category] = dataset.apply(lambda row: row['triage_zone_name'] if row['triage_zone_name'] in categories else 'other', axis=1)

    # display a score for how many entries in dataset[category] are NOT in dataset[triage_zone_name]
    score_instruction_not_followed = round(len(dataset[~dataset[category].isin(dataset['triage_zone_name'])]) / len(dataset), 2)
    print(f"Score for {category}: {score_instruction_not_followed}")
    # add an extra row to the subplot that displays score_instruction_not_followed

    cm = confusion_matrix(dataset['triage_zone_name'], dataset[category], labels=labels)
    cm = np.round(cm / len(dataset),2)

    np.fill_diagonal(cm, 0)
    print(cm.view())
    row, col = divmod(i, cols)
    heatmap(cm, category + " " + str(score_instruction_not_followed), axs[row, col])   

  # Turn off unused axes
  for j in range(i + 1, rows * cols):
      row, col = divmod(j, cols)
      axs[row, col].axis('off')  # Hide unused sub

  plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area
  plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.6)
  plt.show()



def error_analysis(dataset, categories, previous_error_pattern = None): 

  num_categories = len(categories)
  cols = int(math.ceil(math.sqrt(num_categories)))  # Square root strategy for columns
  rows = int(math.ceil(num_categories / cols))  # Calculate rows based on columns

  category_order = ['healthcare', 'doctor', 'no_prompt']
  # Filter and order categories according to the predefined list
  categories = sorted(categories, key=lambda x: next((i for i, target in enumerate(category_order) if target in x), float('inf')))

  category_order = ['from_paper', 'action_oriented']
  # Filter and order categories according to the predefined list
  categories = sorted(categories, key=lambda x: next((i for i, target in enumerate(category_order) if target in x), float('inf')))

  print(categories)


  fig, axs = plt.subplots(rows, cols, figsize=(20, 15))  # Adjust the figure size as needed

  x, y = 0, 0
  print(categories)


  # remove non values
  # dataset = dataset.dropna()

  # Make this a 2x6 grid
  num_categories = len(categories) 
  cols = int(math.ceil(math.sqrt(num_categories)))  # Square root strategy for columns
  rows = int(math.ceil(num_categories / cols))  # Calculate rows based on columns

  fig, axs = plt.subplots(rows, cols, figsize=(20, 15))
    
  # Ensure axs is a 2D array for consistency
  if num_categories == 1:
        axs = np.array([[axs]])
  elif cols == 1 or rows == 1:
        axs = np.reshape(axs, (rows, cols))

  # Error Patterns 
  error_proportions = []
  instruction_not_followed = []
  overcaring_errors = []
  neglecting_errors = []

  # * Define Overcaring and neglecting

  labels = ["MINOR", "EXPECTANT/DECEASED", "DELAYED", "IMMEDIATE"]

  # Overcaring Errors
  # Overcaring: resource unintensive -> resource intesive 
  # m -> e, m -> d, m -> i
  # e -> d, e -> i
  # d -> i
        

  overcaring = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
  
  print("overcaring", overcaring)

  # Neglecting Errors 
  # Neglecting: resource intensive  -> resource unintesive 
  # i -> d, i -> e, i -> m
  # d -> e, d -> m
  # e -> m

  neglecting = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i - 1, -1, -1)]

  print("neglecting", neglecting)

        

  for category in categories:
      
      # Overallproportoin of errors
      error_proportions.append(len(dataset[dataset[category] != dataset['triage_zone_name']]) / len(dataset))
  
      # Instuction following Errors
      instruction_not_followed.append(len(dataset[~dataset[category].isin(dataset['triage_zone_name'])]) / len(dataset))
      
      
      # check whether predicted category and actual category are in the overcaring pattern
      overcare_err_count = dataset.apply(lambda x: (x['triage_zone_name'], x[category]) in overcaring, axis=1).sum()
      overcaring_errors.append(overcare_err_count / len(dataset))      
      # overcaring_errors.append((dataset[category], dataset["triage_zone_name"]).isin("overcaring"))
          
      neglect_err_count = dataset.apply(lambda x: (x['triage_zone_name'], x[category]) in neglecting, axis=1).sum()
      neglecting_errors.append(neglect_err_count / len(dataset))      
      # neglecting_errors.append((dataset[category], dataset["triage_zone_name"]).isin("overcaring"))

      print("overcaring errors", len(overcaring_errors), overcaring_errors)
      print("neglecting errors", len(neglecting_errors), neglecting_errors)

  # Shorten each value in category to only include the parts that are in ["haiku", "opus", "gpt-3.5-turbo", "gpt-4", "mistral", "mixtral", "no_prompt", "healthcare", "doctor", "from_paper", "action_oriented", "utilitarian", "deontological"]
  desired_labels =  ["haiku", "opus", "gpt-3.5-turbo", "gpt-4", "mistral", "mixtral", "paper", "action", "outcome", "no", "utilitarianism", "deontology", "healthcare", "doctor"]
  labels = [ "_".join([l for l in label.split("_") if l in desired_labels])for label in categories]
  print("LABELS", labels)

  # Plotting
  plt.figure(figsize=(12, 6))

  x = np.arange(len(categories))
  bar_width = 0.35  # Width of the bars

  # Plot total errors as a separate bar for comparison
  bars_total = plt.bar(x, error_proportions, width=bar_width, label='Total Errors', color='grey')
  annotate_bars(bars_total)

  # Stacking and plotting specific error types
  bars_instruction = plt.bar(x + bar_width, instruction_not_followed, label='Instruction Not Followed', width=bar_width, color='aqua')
  bars_overcaring = plt.bar(x + bar_width, overcaring_errors, bottom=instruction_not_followed, label='Overcaring Errors', width=bar_width, color='lightgreen')
  neglect_bottoms = np.array(instruction_not_followed) + np.array(overcaring_errors)
  bars_neglecting = plt.bar(x + bar_width, neglecting_errors, bottom=neglect_bottoms, label='Neglecting Errors', width=bar_width, color='lightcoral')

  # Annotate specific error bars
  annotate_bars(bars_instruction)
  annotate_bars(bars_overcaring, previous_heights=instruction_not_followed)
  annotate_bars(bars_neglecting, previous_heights=neglect_bottoms)

  plt.xlabel('Category')
  plt.ylim(0, 1)
  plt.ylabel('Proportion of Errors')
  plt.title(f'Error Analysis by Category (Dataset size: {len(dataset)})')  # Add dataset size to title  
  plt.xticks(x + bar_width / 2, categories, rotation=90)
  plt.legend()
  plt.tight_layout()
  plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.55, wspace=0.4, hspace=0.6)

  plt.show()

  return {"total_errors": error_proportions, "instruction_not_followed": instruction_not_followed, "overcaring": overcaring_errors, "neglecting": neglecting_errors}


def error_analysis(dataset, categories, previous_error_pattern = None, average = None): 

  num_categories = len(categories)
  cols = int(math.ceil(math.sqrt(num_categories)))  # Square root strategy for columns
  rows = int(math.ceil(num_categories / cols))  # Calculate rows based on columns

  category_order = ['healthcare', 'doctor', 'no_prompt']
  # Filter and order categories according to the predefined list
  categories = sorted(categories, key=lambda x: next((i for i, target in enumerate(category_order) if target in x), float('inf')))

  category_order = ['from_paper', 'action_oriented']
  # Filter and order categories according to the predefined list
  categories = sorted(categories, key=lambda x: next((i for i, target in enumerate(category_order) if target in x), float('inf')))

  print(categories)


  fig, axs = plt.subplots(rows, cols, figsize=(20, 15))  # Adjust the figure size as needed

  x, y = 0, 0
  print(categories)

  # Create a confusion matrix
  dataset["triage_zone_name"] = [color_to_class[zone] for zone in dataset["triage_zone"]]
  
  # remove non values
  # dataset = dataset.dropna()

  # Make this a 2x6 grid
  num_categories = len(categories) 
  cols = int(math.ceil(math.sqrt(num_categories)))  # Square root strategy for columns
  rows = int(math.ceil(num_categories / cols))  # Calculate rows based on columns

  fig, axs = plt.subplots(rows, cols, figsize=(20, 15))
    
  # Ensure axs is a 2D array for consistency
  if num_categories == 1:
        axs = np.array([[axs]])
  elif cols == 1 or rows == 1:
        axs = np.reshape(axs, (rows, cols))

  # Error Patterns 
  error_proportions = []
  instruction_not_followed = []
  overcaring_errors = []
  neglecting_errors = []

  # * Define Overcaring and neglecting

  labels = ["MINOR", "EXPECTANT/DECEASED", "DELAYED", "IMMEDIATE"]

  # Overcaring Errors
  # Overcaring: resource unintensive -> resource intesive 
  # m -> e, m -> d, m -> i
  # e -> d, e -> i
  # d -> i
        

  overcaring = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
  
  print("overcaring", overcaring)

  # Neglecting Errors 
  # Neglecting: resource intensive  -> resource unintesive 
  # i -> d, i -> e, i -> m
  # d -> e, d -> m
  # e -> m

  neglecting = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i - 1, -1, -1)]

  print("neglecting", neglecting)

        

  for category in categories:
      
      # Overallproportoin of errors
      error_proportions.append(len(dataset[dataset[category] != dataset['triage_zone_name']]) / len(dataset))
  
      # Instuction following Errors
      instruction_not_followed.append(len(dataset[~dataset[category].isin(dataset['triage_zone_name'])]) / len(dataset))
      
      
      # check whether predicted category and actual category are in the overcaring pattern
      overcare_err_count = dataset.apply(lambda x: (x['triage_zone_name'], x[category]) in overcaring, axis=1).sum()
      overcaring_errors.append(overcare_err_count / len(dataset))      
      # overcaring_errors.append((dataset[category], dataset["triage_zone_name"]).isin("overcaring"))
          
      neglect_err_count = dataset.apply(lambda x: (x['triage_zone_name'], x[category]) in neglecting, axis=1).sum()
      neglecting_errors.append(neglect_err_count / len(dataset))      
      # neglecting_errors.append((dataset[category], dataset["triage_zone_name"]).isin("overcaring"))

      print("overcaring errors", len(overcaring_errors), overcaring_errors)
      print("neglecting errors", len(neglecting_errors), neglecting_errors)
  

  # Plotting
  plt.figure(figsize=(12, 6))

  x = np.arange(len(categories))
  bar_width = 0.35  # Width of the bars

  # Plot total errors as a separate bar for comparison
  bars_total = plt.bar(x, error_proportions, width=bar_width, label='Total Errors', color='grey')
  annotate_bars(bars_total)

  # Stacking and plotting specific error types
  bars_instruction = plt.bar(x + bar_width, instruction_not_followed, label='Instruction Not Followed', width=bar_width, color='aqua')
  bars_overcaring = plt.bar(x + bar_width, overcaring_errors, bottom=instruction_not_followed, label='Overcaring Errors', width=bar_width, color='lightgreen')
  neglect_bottoms = np.array(instruction_not_followed) + np.array(overcaring_errors)
  bars_neglecting = plt.bar(x + bar_width, neglecting_errors, bottom=neglect_bottoms, label='Neglecting Errors', width=bar_width, color='lightcoral')

  # Annotate specific error bars
  annotate_bars(bars_instruction)
  annotate_bars(bars_overcaring, previous_heights=instruction_not_followed)
  annotate_bars(bars_neglecting, previous_heights=neglect_bottoms)

  plt.xlabel('Category')
  plt.ylim(0, 1)
  plt.ylabel('Proportion of Errors')
  plt.title(f'Error Analysis by Category (Dataset size: {len(dataset)})')  # Add dataset size to title  
  plt.xticks(x + bar_width / 2, categories, rotation=90)
  plt.legend()
  plt.tight_layout()
  plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.55, wspace=0.4, hspace=0.6)

  plt.show()

  return {"total_errors": error_proportions, "instruction_not_followed": instruction_not_followed, "overcaring": overcaring_errors, "neglecting": neglecting_errors}

def annotate_bars(bars, previous_heights=None):
    """ Helper function to annotate bars in the bar chart. """
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        y_offset = bar.get_y()
        # Calculate the cumulative height for positioning the text above the stack
        if previous_heights is not None and np.any(previous_heights):            
            cumulative_height = previous_heights[idx]
        else:
            cumulative_height = 0
        
        total_height = cumulative_height + height
        plt.text(bar.get_x() + bar.get_width() / 2.0, total_height, f'{height:.2f}', ha='center', va='top')




def avg_error_analysis(dataset, categories):
    overcaring = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]

    neglecting = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i - 1, -1, -1)]

    # Same as before but ensure numeric operations are performed correctly
    error_counts = {key: 0 for key in ['total_errors', 'instruction_not_followed', 'overcaring', 'neglecting']}
    
    num_entries = len(dataset)
    
    for category in categories:
        error_counts['total_errors'] += np.sum(dataset[category] != dataset['triage_zone_name']) / num_entries
        error_counts['instruction_not_followed'] += np.sum(~dataset[category].isin(dataset['triage_zone_name'])) / num_entries
        
        #check whether predicted category and actual category are in the overcaring pattern
        overcare_err_count = dataset.apply(lambda x: (x['triage_zone_name'], x[category]) in overcaring, axis=1).sum()
        error_counts['overcaring'] += overcare_err_count / num_entries     
        # overcaring_errors.append((dataset[category], dataset["triage_zone_name"]).isin("overcaring"))
            
        neglect_err_count = dataset.apply(lambda x: (x['triage_zone_name'], x[category]) in neglecting, axis=1).sum()
        error_counts['neglecting'] += neglect_err_count / num_entries   
        # neglecting_errors.append((dataset[category], dataset["triage_zone_name"]).isin("overcaring"))

        # Assumes `overcaring` and `neglecting` are defined outside this function or passed as parameters
        
    # Average errors per category
    return {key: value / len(categories) for key, value in error_counts.items()}



def visualize_avg_errors(model_results, n_models, bar_width): 
  # Sort the model results dictionary by total errors in descending order
  sorted_model_names = sorted(model_results.keys(), key=lambda x: model_results[x]['total_errors'], reverse=True)

  # Now reorganize the data based on the sorted order
  model_results = {model: model_results[model] for model in sorted_model_names}
  # Set up the figure
  fig, ax = plt.subplots(figsize=(10, 6))

  n_models = len(model_results)
  index = np.arange(n_models)
  bar_width = 0.2

  colors = {
      'total_errors': 'grey',  # Grey color
      'instruction_not_followed': 'aqua',  # Light green color
      'overcaring': 'lightgreen',  # Light orange color, using hex code
      'neglecting': 'lightcoral'  # Light yellow color, using hex code
  }
  # Plot each error type
  for i, error_type in enumerate(['total_errors', 'instruction_not_followed', 'overcaring', 'neglecting']):
      values = [model_results[model][error_type] for model in model_results]
      plt.bar(index + i * bar_width, values, bar_width, label=error_type, color=colors[error_type])

  plt.xlabel('Model')
  plt.ylabel('Average Error Scores')
  plt.title('Average Error Scores Per Model')
  plt.xticks(index + 1.5 * bar_width, model_results.keys())
  plt.legend()

  plt.tight_layout()
  # plt.show()




def filter_data_for_conditions(data):
    # Filter columns that match any of the conditions of interest
    conditions = ["deontology", "no prompt", "utilitarianism"]
    columns = [col for col in data.columns if any(cond in col for cond in conditions)]
    return data[columns + ['triage_zone']]  # Assume 'triage_zone' holds the correct labels
