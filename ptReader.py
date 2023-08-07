import torch
from torch_geometric.data import Data

def print_data_object_details(data_obj):
    print("Node features (x):", data_obj.x.shape)
    print("Edge indices (edge_index):", data_obj.edge_index.shape)
    print("Edge attributes (edge_attr):", data_obj.edge_attr.shape if data_obj.edge_attr is not None else None)
    print("Labels (y):", data_obj.y.shape if data_obj.y is not None else None)
    print("------")

def print_model_details(model_path):
    try:
        # Load the list of Data objects from the .pt file
        data_list = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Iterate over each Data object and print its details
        for i, data_obj in enumerate(data_list):
            print(f"Data object {i+1} details:")
            print_data_object_details(data_obj)

    except Exception as e:
        print(f"Error while loading the data objects: {e}")

model_path = './LineSegmentClassifier/LineSegmentClassifierGNN/ChangGNN_MDnodes_LSedges_test.pt'
print_model_details(model_path)
