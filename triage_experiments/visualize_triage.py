import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from triage_zone_mapping import class_to_color, color_to_class
import math


labels = ["MINOR", "EXPECTANT/DECEASED", "DELAYED", "IMMEDIATE"]

# Pivot the DataFrame for the heatmap
def visualize(summary):
  # pivot_df = summary.pivot_table(index=['model', 'prompt_type'], columns='syntax', values='proportion_correct')
  pivot_df = summary.pivot_table(index=['model'], columns='prompt_type', values='proportion_correct')

  plt.figure(figsize=(8, 6))
  sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title('Proportion of Correct Answers by Model and ethics prompt Type')
  # plt.title('Proportion of Correct Answers by Model, Prompt Type, and Syntax')
  # plt.xlabel('Syntax')
  plt.xlabel('Ethics prompt')
  plt.ylabel('Model')

  # plt.ylabel('Model and Prompt Type')

  plt.show()


def heatmap(data, title, ax):
    # Assuming 'data' is the confusion matrix and 'title' is the title for the heatmap
    # 'ax' is the axes object where the heatmap will be drawn
    sns.heatmap(data, annot=True, fmt="d", ax=ax, cmap="Reds", vmin=0, vmax=15, xticklabels=["m", "e", "d", "i"], yticklabels=["m", "e", "d", "i"])
    ax.set_title(title, fontsize = 7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')


def misclassification(dataset, categories):
  fig, axs = plt.subplots(3, 6, figsize=(20, 15))  # Adjust the figure size as needed

  x, y = 0, 0
  print(categories)
  # Create a confusion matrix
  dataset["triage_zone_name"] = [color_to_class[zone] for zone in dataset["triage_zone"]]
  # remove non values
  dataset = dataset.dropna()

  # Make this a 2x6 grid
  # xy = [[0, 0], [0,1], [0,2], [0,3], [0,4], [0,5], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5]]
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

    np.fill_diagonal(cm, 0)
    print(cm.view())
    row, col = divmod(i, cols)
    heatmap(cm, category, axs[row, col])

  plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area
  plt.show()