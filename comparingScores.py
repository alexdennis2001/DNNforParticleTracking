import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



# Read the two CSV files into pandas DataFrames
csv_file_gnn = 'csv/models/LineSegment_Classifier_200_50/inferences_test_LineSegment_Classifier_200_50.csv'
csv_file_dnn = 'csv/models/LSC_DNN_model_epoch50/inferences_test_LSC_DNN_model_epoch50.csv'

gnn_df = pd.read_csv(csv_file_gnn)
dnn_df = pd.read_csv(csv_file_dnn)

gnn_df = gnn_df[['LS_idx', 'label', 'score']]
gnn_df['LS_idx'] = np.arange(0, len(gnn_df))
gnn_df.columns = dnn_df.columns
dnn_df['idx'] = np.arange(0, len(dnn_df))


print("Check datasets are same size: ", np.all(gnn_df['idx'] == dnn_df['idx']))

# Threshold value for scores
x = 0.05

# Filter the DataFrames based on the score threshold "x"
gnn_filtered_df = gnn_df[gnn_df['score'] < x]
dnn_filtered_df = dnn_df[dnn_df['score'] < x]


# Get the count of scores that satisfy the threshold condition for both 'gnn_df' and 'dnn_df'
count_gnn_above_threshold = len(gnn_filtered_df)
count_dnn_above_threshold = len(dnn_filtered_df)

# Check if the "idx" values in the filtered DataFrames are present in both DataFrames
idx_in_both = np.isin(gnn_filtered_df['idx'], dnn_filtered_df['idx'])

print("Are idx values present in both DataFrames?", np.sum(idx_in_both))

print('Count of scores above {}:'.format(x))
print(' Total GNN: ', count_gnn_above_threshold)
print(' Total DNN: ', count_dnn_above_threshold)
print(' Sum GNN score: ', np.sum(gnn_df.loc[gnn_df['score'] < x, 'score']))
print(' Sum DNN score: ', np.sum(dnn_df.loc[dnn_df['score'] < x, 'score']))

# Create scatter plot
plt.figure(figsize=(15, 15))

plt.hist2d(dnn_df['score'], gnn_df['score'], bins=[np.linspace(0, 1, 101), np.linspace(0, 1, 101)], norm=mpl.colors.LogNorm())
plt.xlabel('DNN Scores')
plt.ylabel('GNN Scores')
plt.title('DNN vs GNN Scores Scatter Plot')
plt.grid(True)
plt.savefig(f'plots/comparing_Score/GNNvsDNN_Score.png')

