import matplotlib.pyplot as plt
import torch
import numpy

TEST_DATASET_PATH = './files/LS_pt/log10_15to25_0to14/test.pt'
TRAIN_DATASET_PATH = './files/LS_pt/log10_15to25_0to14/train.pt'


test_dataset = torch.load(TEST_DATASET_PATH)
train_dataset = torch.load(TRAIN_DATASET_PATH)

def node_features_histograms(data, plot_name, dataset_name):

        name_node_features = [
             'MD_0_x', 
             'MD_0_y', 
             'MD_0_z', 
             'MD_1_x', 
             'MD_1_y', 
             'MD_1_z', 
             'MD_dphichange',]

        for i in range(7):
            fig, axes = plt.subplots()    

            axes.hist(data[:,i], bins=get_equidistant_bins(data[:,i]))
            axes.set_title(f'{plot_name} {name_node_features[i]} {dataset_name}')
            axes.set_xlabel(name_node_features[i])
            axes.set_ylabel('Count')
            plt.savefig(f'./plots/dataset/{dataset_name}/node_features/{plot_name}-{name_node_features[i]}.png')
            print(f"Plot {plot_name}-{name_node_features[i]} for {dataset_name} generated!\n")

def edge_features_histograms(data, plot_name, dataset_name):
        
        edge_features_name = ['LS_pt', 'LS_eta', 'LS_phi']

        for i in range(3):
         
            fig, axes = plt.subplots()

            if i == 0:
                LS_pt = data[:,0]
                #LS_pt[LS_pt > 2000] = 2000
                LS_pt[LS_pt > 2000] = 2000
                axes.hist(LS_pt, bins=get_equidistant_bins(LS_pt))    

            else:
                axes.hist(data[:,i], bins=get_equidistant_bins(data[:,i]))    
                
            axes.set_title(f'{plot_name} {edge_features_name[i]} {dataset_name}')
            axes.set_xlabel(edge_features_name[i])
            axes.set_ylabel('Count')
            plt.savefig(f'./plots/dataset/{dataset_name}/edge_features/{plot_name}-{edge_features_name[i]}.png')
            print(f"Plot {plot_name}-{edge_features_name[i]} for {dataset_name} generated!\n")

def truth_label_histograms(data, plot_name, dataset_name):

    fig, axes = plt.subplots()

    axes.hist(data, bins=[0,1,2])
    axes.set_title(f'{plot_name} LS_isFake {dataset_name}')
    axes.set_xlabel('LS_isFake')
    axes.set_ylabel('Count')
    plt.savefig(f'./plots/dataset/{dataset_name}/truth_label/{plot_name}-LS_isFake.png')
    print(f"Plot {plot_name}-LS_isFake for {dataset_name} generated!\n")

def generate_plots(data, plot_name, dataset_name):
    
    if plot_name == 'node_features':
        node_features_histograms(data, plot_name, dataset_name)

    if plot_name == 'edge_features':
        edge_features_histograms(data, plot_name, dataset_name)

    if plot_name == 'truth_label':
        truth_label_histograms(data, plot_name, dataset_name)
    
def display_tensors(nodes, edges, labels):
    print(f'X (node_features): {nodes.size()}\n')
    print(f'Edge attributes (edge_features): {edges.size()}\n')
    print(f'Y (truth_labels): {labels.size()}\n')

def get_merged_features(dataset_name):
    
    index = 0
    node_features = 0
    edge_features = 0
    truth_label = 0

    if dataset_name == 'test':
        for sample in test_dataset:

            #print(sample)

            if index == 0:
                node_features = sample.x.clone()
                edge_features = sample.edge_attr.clone()
                truth_label = sample.y.clone()
            else:
                node_features = torch.cat((node_features, sample.x), dim=0)
                edge_features = torch.cat((edge_features, sample.edge_attr), dim=0)
                truth_label = torch.cat((truth_label, sample.y), dim=0)

            index = index + 1
    
    if dataset_name == 'train':
        for sample in train_dataset:

                #print(sample)

                if index == 0:
                    node_features = sample.x.clone()
                    edge_features = sample.edge_attr.clone()
                    truth_label = sample.y.clone()
                else:
                    node_features = torch.cat((node_features, sample.x), dim=0)
                    edge_features = torch.cat((edge_features, sample.edge_attr), dim=0)
                    truth_label = torch.cat((truth_label, sample.y), dim=0)

                index = index + 1

    return node_features, edge_features, truth_label
     
def get_equidistant_bins(data):
    bins = numpy.linspace(
         torch.min(data).item(), 
         torch.max(data).item(), 
         11)
    
    return bins
    
def main():
    
    dataset_name = 'train'
    node_features, edge_features, truth_label = get_merged_features(dataset_name)

    display_tensors(node_features, edge_features, truth_label)

    generate_plots(node_features, 'node_features', dataset_name)
    generate_plots(edge_features, 'edge_features', dataset_name) 
    generate_plots(truth_label, 'truth_label', dataset_name)

if __name__ == '__main__':
    main()