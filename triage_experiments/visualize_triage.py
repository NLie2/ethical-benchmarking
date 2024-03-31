import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from triage_zone_mapping import class_to_color, color_to_class


labels = ["MINOR", "DELAYED", "IMMEDIATE", "EXPECTANT/DECEASED"]

# Pivot the DataFrame for the heatmap
def visualize(summary):
  pivot_df = summary.pivot_table(index=['model', 'prompt_type'], columns='syntax', values='proportion_correct')

  plt.figure(figsize=(8, 6))
  sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title('Proportion of Correct Answers by Model, Prompt Type, and Syntax')
  plt.xlabel('Syntax')
  plt.ylabel('Model and Prompt Type')

  plt.show()


def heatmap(data, title, ax):
    # Assuming 'data' is the confusion matrix and 'title' is the title for the heatmap
    # 'ax' is the axes object where the heatmap will be drawn
    sns.heatmap(data, annot=True, fmt="d", ax=ax, cmap="Reds", vmin=0, vmax=15, xticklabels=["m", "d", "i", "e"], yticklabels=["m", "d", "i", "e"])
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
  xy = [[0, 0], [0,1], [0,2], [0,3], [0,4], [0,5], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5]]

  for i, category in enumerate(categories):

    print(category)
    print(i)
    cm = confusion_matrix(dataset['triage_zone_name'], dataset[category], labels=labels)

    np.fill_diagonal(cm, 0)
    print(cm.view())
    heatmap(cm, category, axs[xy[i][0], xy[i][1]])


  plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area
  plt.show()