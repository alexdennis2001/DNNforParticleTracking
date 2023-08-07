import argparse
import csv
import os
import torch
import questionary
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from datasets import EdgeDataset, EdgeDataBatch  # * Important for wrapping PYG tensors to pytorch tensors
from DeepNeuralNetwork_class import DeepNeuralNetwork # * Class file for model

#Define hyperparameters
THRESH = 0.55

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

def roc_curve_plot(labels_test, scores_test, labels_train, scores_train, model_name):

    # * Calculate ROC curve for test
    fpr_test, tpr_test, thresholds = roc_curve(labels_test, scores_test)
    
    # * ... for train
    fpr_train, tpr_train, thresholds = roc_curve(labels_train, scores_train)

    # Plot ROC curve
    print("...ROC plot INITIALIZED...")
    fig, axes = plt.subplots()
    axes.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='test (area = %0.6f)' % auc(fpr_test, tpr_test))
    axes.plot(fpr_train, tpr_train, color='darkblue', lw=2, label='train (area = %0.6f)' % auc(fpr_train, tpr_train))
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.0])
    axes.set_xlabel('False Positive Rate')
    axes.set_ylabel('True Positive Rate')
    axes.legend(loc="lower right")  # Display the legend at the lower right corner
    axes.set_title(f'ROC Curve of {model_name}')
    plt.savefig(f'./plots/models/{model_name}/roc_curve_{model_name}.png')
    print("...ROC plot FINALIZED...")

def predicted_scores_histogram(predicted_scores_list, labels_list, model_name, dataset_type):

    # * Separate predicted scores for each class based on true_labels_list
    scores_class_0 = np.array(predicted_scores_list)[np.where(np.array(labels_list) == 0)]
    scores_class_1 = np.array(predicted_scores_list)[np.where(np.array(labels_list) == 1)]

    # * Calculate the normalization factors
    num_class_0 = len(scores_class_0)
    num_class_1 = len(scores_class_1)
    norm_factor_0 = 1.0 / num_class_0
    norm_factor_1 = 1.0 / num_class_1

    # * Create a folder to save the plot
    plot_folder = f"./plots/models/{model_name}/"
    os.makedirs(plot_folder, exist_ok=True)

    # * Plot histograms for each class with explicit normalization
    print("...predicted scores plot INITIALIZED...")
    fig, axes = plt.subplots()
    #   for isFake = 0
    axes.hist(scores_class_0, bins=100, color='blue', alpha=0.7, label='isFake=0', density=False, weights=np.full_like(scores_class_0, norm_factor_0))
    #   for isFake = 1
    axes.hist(scores_class_1, bins=100, color='red', alpha=0.7, label='isFake=1', density=False, weights=np.full_like(scores_class_1, norm_factor_1))

    axes.set_title(f'Predicted Scores of {dataset_type}_{model_name}')
    axes.set_xlabel('Scores')
    axes.set_ylabel('Count')
    axes.legend()
    plt.savefig(f'{plot_folder}predicted_scores_{dataset_type}_{model_name}.png')
    print("...predicted scores plot FINALIZED...")

def inference(model, dataloader, device, dataset_type, model_name):
    total_correct = 0
    total_samples = 0
    row_index = 0
    labels_list = []
    inference_data = []
    predicted_scores_list = []

    print("...INFERENCE INITIALIZED...")
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)

            outputs = model(data.x, data.edge_index, data.edge_attr)
            predicted_scores = outputs.squeeze(1).cpu().numpy().flatten()
            predicted_scores_list_round = [round(score, 17) for score in predicted_scores] 
            predicted_scores_list.extend(predicted_scores_list_round)
            true_labels = data.y.int().squeeze() # returns int tensor
            labels_list.extend(true_labels.cpu().numpy())

            # * Calculating True Positives & True Negatives for accuracy rating via THRESH
            TP = torch.sum((data.y == 1).squeeze() & (outputs >= THRESH).squeeze()).item()
            TN = torch.sum((data.y == 0).squeeze() & (outputs <  THRESH).squeeze()).item()

            correct = TP + TN
            total_correct += correct
            total_samples += len(true_labels)

            # * Append the data to the inference_data list for exporting to CSV
            print("Batch No.", batch_idx)
            batch_size = data.y.size(0)  # Get the size of the current batch
            for i in range(batch_size):
                row_index += 1
                if i >= len(labels_list) or i >= len(predicted_scores_list_round):
                # Skip if the index exceeds the available data in the lists
                    break
                truth = labels_list[i].item()
                score = round(predicted_scores_list_round[i].item(), 17)
                inference_data.append([row_index, truth, score])
                print(f"idx: {row_index}\ttruth: {truth}\tscore: {score}")
            print("--------------------------------------------\n")

    overall_accuracy = total_correct / total_samples
    print(f"Overall Accuracy (THRESHOLD = {THRESH}): {overall_accuracy:0.3f}")
    print("...INFERENCE COMPLETED...\n")

    predicted_scores_histogram(predicted_scores_list, labels_list, model_name, dataset_type)
    export_to_csv(inference_data, model_name, dataset_type)

    return labels_list, predicted_scores_list

def get_dataset_path(dataset_type):
    # Define the paths to the dataset folders
    dataset_options = f'./LineSegmentClassifier/datasets/{dataset_type}'

    # Get the list of files in the dataset folder
    dataset_files = os.listdir(dataset_options)

    # Sort the files for better visibility
    dataset_files.sort()

    # Create the list of choices for questionary select question
    dataset_choices = [os.path.join(dataset_options, file) for file in dataset_files]

    # Prompt the user to select dataset path
    dataset_path = questionary.select(
        f'Select the {dataset_type} dataset file:',
        choices=dataset_choices,
        use_shortcuts=True,  # Allow using shortcuts for scrolling
    ).ask()

    return dataset_path

def get_model_path():
    # Define the path to the model folder
    models_folder_options = f'./LineSegmentClassifier/models'

    # Get the list of model files in the folder
    model_files = os.listdir(models_folder_options)

    # Filter .pt files
    model_files = [file for file in model_files if file.endswith('.pt')]

    # Sort the files for better visibility
    model_files.sort()

    # Create the list of choices for questionary select question
    model_choices = [os.path.join(models_folder_options, file) for file in model_files]

    # Prompt the user to select the model file
    model_file = questionary.select(
        'Select the model file:',
        choices=model_choices,
        use_shortcuts=True,  # Allow using shortcuts for scrolling
    ).ask()

    return model_file

def main():

    # TODO consider merging with inferences_GNN...

    BATCH_SIZE = 1000
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='LineSegmentClassifier')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    args = parser.parse_args()

    # Set the batch size from command-line argument
    #BATCH_SIZE = args.batch_size

# * Get model path
    #MODEL_PATH = get_model_path()
    TEST_DATASET_PATH = './LineSegmentClassifier/LineSegmentClassifier/LineSegmentClassifier_test.pt'
    TRAIN_DATASET_PATH = './LineSegmentClassifier/LineSegmentClassifier/LineSegmentClassifier_train.pt'
    MODEL_PATH = './LineSegmentClassifier/models/LSC_DNN_model_20230722_145012.pt' 
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

    # * Initialize the neural network in available GPU
    model = DeepNeuralNetwork().to(device)

    # * Load the pre-trained model
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    print("Model loaded for inference.")

    # * Perform inference
    print('TEST INFERENCE RUN')
    labels_test, scores_test = inference(model, test_loader, device, 'test', model_name)
    print('TRAIN INFERENCE RUN')
    labels_train, scores_train = inference(model, train_loader, device, 'train', model_name)

    # * Plot ROC curve
    roc_curve_plot(labels_test, scores_test, labels_train, scores_train, model_name)

if __name__ == '__main__':
    main()
