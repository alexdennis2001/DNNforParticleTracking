import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(true_labels, predicted_scores, dataset_name):
    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, predicted_scores)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Plot the ROC curve
    plt.plot(false_positive_rate, true_positive_rate, lw=2, label=f'{dataset_name} (AUC = {roc_auc:.6f})')

# Model name and paths
model_name = 'LineSegmentClassifier_1000_50'
test_data = pd.read_csv(f'csv/models/{model_name}/inferences_test_{model_name}.csv')

# Extract true labels and predicted scores for train and test datasets
test_true_labels = test_data['truth']
test_predicted_scores = test_data['score']

# Plot the ROC curves for both train and test datasets
plt.figure(figsize=(8, 6))
plot_roc_curve(test_true_labels, test_predicted_scores, 'Test')

# Add random ROC curve for reference
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curves - Test ({model_name})')
plt.legend(loc='lower right')

# Save the plot to a file
plt.savefig(f'plots/ROC_Curve/ROC_{model_name}_Test.png')


