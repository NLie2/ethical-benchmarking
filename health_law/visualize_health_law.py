import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


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