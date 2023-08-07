import csv
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from datasets import EdgeDataset, EdgeDataBatch  # * Important for wrapping PYG tensors to pytorch tensors


# * Absolute paths
TEST_DATASET_PATH = './LineSegmentClassifier/LineSegmentClassifier/LineSegmentClassifier_test.pt'
TRAIN_DATASET_PATH = './LineSegmentClassifier/LineSegmentClassifier/LineSegmentClassifier_train.pt'
MODEL_PATH = './LineSegmentClassifier/models/LSC_DNN_model_epoch45_20230728_160103.pt' 

# * Define hyperparameters
BATCH_SIZE = 1000
THRESH = 0.55

class NeuralNetwork(nn.Module):
    # * Constructor
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        n_edge_features = 3
        n_node_features = 7
        input_size = 2 * n_node_features + n_edge_features
        n_hidden_layers = 2
        hidden_size = 200

        hidden_layers = []
        for layer_i in range(n_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.ReLU())

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        #print(self)
    
    def forward(self, node_attr, edge_idxs, edge_attr):
        return self.layers(torch.cat((node_attr, edge_attr), dim=1)).unsqueeze(1)

def export_to_csv(inference_data, model_name, dataset_type):
    # * Create a folder to save the plot
    plot_folder = f"./csv/models/{model_name}/"
    os.makedirs(plot_folder, exist_ok=True)

    file_path = f"{plot_folder}inferences_{dataset_type}_{model_name}.csv"
    file_exists = os.path.exists(file_path)
    
    print("...inference csv data file INITIALIZED...")
    if file_exists:
        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['idx', 'truth', 'score'])
            for row in inference_data:
                csvwriter.writerow(row)
    else:
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                csvwriter.writerow(['idx', 'truth', 'score'])
            for row in inference_data:
                csvwriter.writerow(row)

    print("...inference csv data file FINALIZED...")

def predicted_scores_histogram(predicted_scores_list, model_name, dataset_type):

    # * Create a folder to save the plot
    plot_folder = f"./plots/models/{model_name}/"
    os.makedirs(plot_folder, exist_ok=True)

    print("...predicted scores plot INITIALIZED...")
    fig, axes = plt.subplots()
    axes.hist(predicted_scores_list, bins=11)
    axes.set_title(f'Predicted Scores of {dataset_type}_{model_name}_{THRESH}')
    axes.set_xlabel('Scores')
    axes.set_ylabel('Count')
    plt.savefig(f'{plot_folder}predicted_scores_{dataset_type}_{model_name}_{THRESH}.png')
    print("...predicted scores plot FINALIZED...")

def inference(model, dataloader, device, dataset_type, model_name):
    print("...INFERENCE INITIALIZED...")
    total_correct = 0
    total_samples = 0
    predicted_scores_list = []
    labels_list = []
    inference_data = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.edge_attr)
            predicted_scores = outputs.squeeze(1).float() # returns float tensor
            predicted_scores_list_noround = outputs.squeeze(1).cpu().numpy().flatten()
            predicted_scores_list_round = [round(score, 6) for score in predicted_scores_list_noround] 
            predicted_scores_list.extend(predicted_scores_list_round)
            predicted_labels = torch.argmax(outputs, dim=1).squeeze() # returns float tensor
            true_labels = data.y.squeeze() # returns float tensor
            labels_list.extend(true_labels.cpu().numpy())

            # * Calculating True Positives & True Negatives for accuracy rating via THRESH
            TP = torch.sum((data.y == 1).squeeze() & (outputs >= THRESH).squeeze()).item()
            TN = torch.sum((data.y == 0).squeeze() & (outputs <  THRESH).squeeze()).item()

            correct = TP + TN
            accuracy = correct / len(true_labels)
            total_correct += correct
            total_samples += len(true_labels)

            print("Batch No.", batch_idx)
            for i in range(len(predicted_labels)):
                edge_index_str = f"({data.edge_index[i][0]}, {data.edge_index[i][1]})"
                real_ls_isFake = true_labels[i].item()
                score = predicted_scores[i].item()
                inference_data.append([edge_index_str, real_ls_isFake, score])

            print(f"Accuracy for Batch (THRESHOLD = {THRESH}): {accuracy:0.3f}")
            print("--------------------------------------------\n")

    overall_accuracy = total_correct / total_samples

    print(f"Overall Accuracy (THRESHOLD = {THRESH}): {overall_accuracy:0.3f}")
    print("...INFERENCE COMPLETED...\n")

    predicted_scores_histogram(predicted_scores_list, model_name, dataset_type)
    export_to_csv(inference_data, model_name, dataset_type)

    return labels_list, predicted_scores_list

def main():
    # * Get model name
    model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

    # * Load dataset
    test_dataset = torch.load(TEST_DATASET_PATH)
    train_dataset = torch.load(TRAIN_DATASET_PATH)

    # * Create data loader
    test_loader = DataLoader(
        EdgeDataset(test_dataset), batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda batch: EdgeDataBatch(batch)
    )

    train_loader = DataLoader(
        EdgeDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda batch: EdgeDataBatch(batch)
    )

    # * Get if device has a CUDA GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # input_size = train_dataset[0].x.shape[1] + 10

    # * Initialize the neural network in available GPU
    model = NeuralNetwork().to(device)

    # * Load the pre-trained model
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    print("Model loaded for inference.")

    # * Perform inference for test dataset
    print('TEST INFERENCE RUN')
    labels_test, scores_test = inference(model, test_loader, device, 'test', model_name)

    # * Perform inference for train dataset
    print('TRAIN INFERENCE RUN')
    labels_train, scores_train = inference(model, train_loader, device, 'train', model_name)


    

if __name__ == '__main__':
    main()
