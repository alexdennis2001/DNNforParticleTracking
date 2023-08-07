### Read the data from the JSON file
with open('LineSegmentClassifier/models/results/LSC_DDN_model_5000_500_0.0001.json', 'r') as json_file:
    data = json.load(json_file)

# Extract the model name and loss values from the JSON data
model_name = data["model_name"]
train_loss_values = data["train_loss"]
test_loss_values = data["test_loss"]

# Plot the loss curves for train and test
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, label='Train Loss')
plt.plot(range(1, len(test_loss_values) + 1), test_loss_values, label='Test Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss Curves - Train vs Test ({model_name})')
plt.legend()

# Save the plot to a file
plt.savefig(f'plots/Loss_Curve/Loss_{model_name}_Train_Test.png')