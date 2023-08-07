import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json
from datasets import EdgeDataset, EdgeDataBatch
from torch.utils.data import DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, node_attr, edge_idxs, edge_attr):
        return self.layers(torch.cat((node_attr, edge_attr), dim=1))

# Load the .pt dataset
data_train = torch.load("LineSegmentClassifier/LineSegmentClassifier/LineSegmentClassifier_train.pt")
data_test = torch.load("LineSegmentClassifier/LineSegmentClassifier/LineSegmentClassifier_test.pt")
train_data = EdgeDataset(data_train)  
test_data = EdgeDataset(data_test)

# Parameters
batch_size = 10000
lr = 0.002
n_epochs = 50

train_loader = DataLoader(train_data, batch_size= batch_size, shuffle=True, collate_fn=lambda batch: EdgeDataBatch(batch))
test_loader = DataLoader(test_data, batch_size= batch_size, shuffle=True, collate_fn=lambda batch: EdgeDataBatch(batch))

# print(data_train[0].x.shape[1])
input_size = data_train[0].x.shape[1] + 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model
model = NeuralNetwork(input_size).to(device)

# Train the model   
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr= lr)
 
duration_minutes = 0
# Summary of the trained model
summary = {
    'model_name': f"LSC_DNN_model_{batch_size}_{n_epochs}_{lr}_200",
    'duration_min': duration_minutes,
    "test_loss": [], 
    "train_loss": [],
}

# Measure the script execution time
start_time = time.time()
for epoch in range(n_epochs):
    mean_loss = 0.0
    model.train()  # Set the model back to training mode
    print(f'Starting epoch: {epoch}')

    for data_i, data_batch in enumerate(train_loader):
        data_batch = data_batch.to(device)
        # Concatenate node_attr and edge_attr
        y_pred = model(data_batch.x, data_batch.edge_index, data_batch.edge_attr)  #forward() called
        loss = loss_fn(y_pred, data_batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()

        if data_i % 100:
            print(f"{data_i}: {mean_loss}")

    mean_loss /= len(train_loader)  # Calculate the mean loss per epoch
    print(f'Finished epoch {epoch}, latest loss {mean_loss}')
    summary["train_loss"].append(mean_loss)

    # Evaluate on test data
    test_loss = 0.0
    correct = 0
    total = 0

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data_batch in test_loader:
            data_batch = data_batch.to(device)
            y_true = data_batch.y

            # Calculate the loss
            y_pred = model(data_batch.x, data_batch.edge_index, data_batch.edge_attr)
            loss = loss_fn(y_pred, y_true)
            test_loss += loss.item()

            # Calculate the accuracy
            predicted = torch.round(y_pred)  # Round the outputs to 0 or 1 for binary classification
            correct += (predicted == y_true).sum().item()
            total += y_true.size(0)

    test_loss /= len(test_loader)
    accuracy = correct / total

    summary["test_loss"].append(test_loss)  # Update the test loss in the summary

    print(f'Epoch {epoch} - Test loss: {test_loss}')
    print(f"Accuracy: {accuracy:.6f}")

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"LineSegmentClassifier/models/LSC_DDN_model_{batch_size}_{n_epochs}_{lr}_{epoch}.pt")

# Calculate the duration in minutes
end_time = time.time()
duration_minutes = (end_time - start_time) / 60
print(f"Script execution time: {duration_minutes:.2f} minutes")

# Update the 'duration' value in the summary dictionary
summary['duration_min'] = duration_minutes

print(summary)

# Save the summary in a JSON file
with open(f"LineSegmentClassifier/models/results/LSC_DDN_model_{batch_size}_{n_epochs}_{lr}_200.json", "w") as file:
    json.dump(summary, file, indent=4)
