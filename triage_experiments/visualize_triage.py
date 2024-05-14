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

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt=".2f", vmin=0.25)
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
    cm = confusion_matrix(dataset['triage_zone_name'], dataset[category], labels=labels)
    cm = np.round(cm / len(dataset),2)

    np.fill_diagonal(cm, 0)
    print(cm.view())
    row, col = divmod(i, cols)
    heatmap(cm, category, axs[row, col])
  
  # Turn off unused axes
  for j in range(i + 1, rows * cols):
      row, col = divmod(j, cols)
      axs[row, col].axis('off')  # Hide unused sub

  plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area
  plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.6)
  plt.show()

# def misclassification(dataset, categories):
#     # Define the desired order of the categories explicitly
#     category_order = ['healthcare_assistant', 'doctor_assistant', 'no_prompt']
#     # Filter and order categories according to the predefined list
#     categories = [cat for cat in category_order if cat in categories]

#     num_categories = len(categories)
#     cols = int(math.ceil(math.sqrt(num_categories)))  # Determine the number of columns
#     rows = int(math.ceil(num_categories / cols))  # Determine the number of rows

#     # Create figure and axes for the subplots
#     fig, axs = plt.subplots(rows, cols, figsize=(20, 15))

#     # Ensure axs is a 2D array for consistency
#     if num_categories == 1:
#         axs = np.array([[axs]])
#     elif cols == 1 or rows == 1:
#         axs = axs.reshape(rows, cols)

#     # Generate each subplot
#     for i, category in enumerate(categories):
#         print(f"Category: {category}, Index: {i}")
#         cm = confusion_matrix(dataset['triage_zone_name'], dataset[category], labels=labels)
#         cm = np.round(cm / len(dataset), 2)  # Normalize and round the confusion matrix

#         np.fill_diagonal(cm, 0)  # Zero out the diagonal to focus on misclassifications
#         row, col = divmod(i, cols)
#         heatmap = plt.imshow(cm, interpolation='nearest', cmap='viridis', ax=axs[row, col])
#         axs[row, col].set_title(category)
#         plt.colorbar(heatmap, ax=axs[row, col])

#     # Turn off unused axes
#     for j in range(i + 1, rows * cols):
#         row, col = divmod(j, cols)
#         axs[row, col].axis('off')  # Hide unused subplot

#     plt.tight_layout()
#     plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.6)
#     plt.show()
