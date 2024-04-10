import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def visualize(summary):
  # pivot_df = summary.pivot_table(index=['model', 'prompt_type'], columns='syntax', values='proportion_correct')
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

def visualize_alt(summary):
    # Assuming 'summary' now includes both 'proportion_correct' and 'std_deviation' for each combination
    # of 'model' and 'prompt_type', as modified in the previous step.
    
    # Create a pivot table for the mean
    pivot_mean = summary.pivot_table(index='model', columns='prompt_type', values='proportion_correct', aggfunc='mean')
    
    # Create a pivot table for the standard deviation
    pivot_std = summary.pivot_table(index='model', columns='prompt_type', values='std_deviation', aggfunc='mean')
    
    # Combine the mean and standard deviation into a single string that will be used for annotations
    pivot_annotation = pivot_mean.round(2).astype(str) + "\n±" + pivot_std.round(2).astype(str)
    
    plt.figure(figsize=(8, 6))
    # Use the pivot_mean for the heatmap values but pivot_annotation for the annotation
    sns.heatmap(pivot_mean, annot=pivot_annotation, fmt='', cmap='coolwarm', vmin=0.5)
    
    plt.title('Proportion of Correct Answers by Model and Ethics Prompt Type')
    plt.xlabel('Ethics Prompt')
    plt.ylabel('Model')
    plt.show()