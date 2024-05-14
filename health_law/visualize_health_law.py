import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def visualize(summary):
  model_means = summary.groupby('model')['proportion_correct'].mean().sort_values(ascending=True)

  # Use the calculated means to sort the index of the pivot table
    
  # Sorting is done before pivot to ensure the heatmap uses this order
  summary['model'] = pd.Categorical(summary['model'], categories=model_means.index, ordered=True)  
  pivot_df = summary.pivot_table(index=['model'], columns='prompt_type', values= 'proportion_correct')

  plt.figure(figsize=(8, 6))
  # sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt=".2f") 
  sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt=".2f", vmin=0.5)
  
  plt.title('Proportion of Correct Answers by Model and ethics prompt Type')
  # plt.title('Proportion of Correct Answers by Model, Prompt Type, and Syntax')
  # plt.xlabel('Syntax')
  plt.xlabel('Ethics prompt')
  plt.ylabel('Model')

  # plt.ylabel('Model and Prompt Type')

  plt.show()

