import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def plot_f1_score_chart(csv_file_path, dataset_type):
    # Read the data from the .csv file
    data = pd.read_csv(csv_file_path)

    # Extract true labels and predicted scores
    true_labels = data['truth'].values
    predicted_scores = data['score'].values

    # Calculate F1 score for different threshold values
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(true_labels, predicted_scores >= threshold) for threshold in thresholds]

    # Find the threshold that maximizes F1 score
    best_threshold = thresholds[np.argmax(f1_scores)]
    max_f1_score = max(f1_scores)

    # Plot the F1 score chart
    plt.plot(thresholds, f1_scores)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score vs. Threshold ({dataset_type})')
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
    plt.axhline(max_f1_score, color='green', linestyle='--', label=f'Max F1 Score: {max_f1_score:.2f}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/F1_Scores/F1_{model_name}_Test.png')

# Replace 'test.csv' and 'train.csv' with the actual paths to your test and train .csv files
model_name = 'LSC_DNN_model_20230722_145012'
test_csv_file_path = f'csv/models/{model_name}/inferences_test_{model_name}.csv'

# Plot F1 score chart for the test dataset
plot_f1_score_chart(test_csv_file_path, 'Test')

