import pandas as pd
import matplotlib.pyplot as plt

def plot_score_histogram(predicted_scores, model_name):
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(predicted_scores, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Predicted Scores')
    plt.ylabel('Frequency')
    plt.title(f'Score Histogram - {model_name}')
    plt.grid(True)
    plt.savefig(f'plots/Histogram_Score/{model_name}_Histogram_Score.png')
    plt.show()

# Model name and paths
model_name = 'LSC_DDN_model_5000_500_0.0001_0.5'
train_data = pd.read_csv(f'csv/models/{model_name}/inferences_train_{model_name}.csv')
test_data = pd.read_csv(f'csv/models/{model_name}/inferences_test_{model_name}.csv')

# Extract predicted scores for train and test datasets
train_predicted_scores = train_data['Score']
test_predicted_scores = test_data['Score']

# Plot histograms for both train and test predicted scores
plot_score_histogram(train_predicted_scores, model_name + "_Train")
plot_score_histogram(test_predicted_scores, model_name + "_Test")
