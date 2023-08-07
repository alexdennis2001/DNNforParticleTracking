import os
import json
import argparse

import uproot
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from utils import SimpleProgress

def transform(feature, transf):
    if transf == None:
        return feature
    elif transf == "rescale":
        return (feature - feature.min())/(feature.max() - feature.min())
    elif transf == "log":
        return torch.log(feature)
    elif transf == "log2":
        return torch.log2(feature)
    elif transf == "log10":
        return torch.log10(feature)
    else:
        raise ValueError(f"transformation '{transf}' not supported")

def ingress(set_name, config, save=True):

    start, stop = config["ingress"][f"{set_name}_entry_range"]
    transforms = config["ingress"].get("transforms", {})
    tree = uproot.open(f"{config['ingress']['input_file']}:{config['ingress']['ttree_name']}")
    print(f"Loaded input file, reading {start} to {stop} for {set_name} set")
    data_list = []
    for batch in SimpleProgress(tree.iterate(step_size=1, filter_name=config["ingress"].get("branch_filter", None), entry_start=start, entry_stop=stop)):

        batch = batch[0,:] # only one event per batch

        # Get truth labels
        truth = torch.tensor(~(batch[config["ingress"]["truth_label"]].to_numpy().astype(bool)), dtype=torch.float)
        truth_mask = truth.to(torch.bool)

        # Get indices of nodes connected by each edge
        edge_idxs = torch.tensor([batch[n].to_list() for n in config["ingress"]["edge_indices"]], dtype=torch.long)

        # Get edge features
        edge_attr = []
        for branch_name in config["ingress"]["edge_features"]:
            feature = torch.tensor(batch[branch_name].to_list(), dtype=torch.float)
            feature[torch.isinf(feature)] = feature[~torch.isinf(feature)].max()
            feature = transform(feature, transforms.get(branch_name, None))
            edge_attr.append(feature)

        edge_attr = torch.transpose(torch.stack(edge_attr), 0, 1)

        # Get node features
        node_attr = []
        for branch_name in config["ingress"]["node_features"]:
            feature = torch.tensor(batch[branch_name].to_list(), dtype=torch.float)
            feature = transform(feature, transforms.get(branch_name, None))
            node_attr.append(feature)

        node_attr = torch.transpose(torch.stack(node_attr), 0, 1)

        if config["ingress"].get("undirected", False):
            edge_idxs_bi, edge_attr_bi = to_undirected(edge_idxs, edge_attr)
            _, truth_bi = to_undirected(edge_idxs, truth)
            data = Data(x=node_attr, y=truth_bi, edge_index=edge_idxs_bi, edge_attr=edge_attr_bi)
        else:
            data = Data(x=node_attr, y=truth, edge_index=edge_idxs, edge_attr=edge_attr)

        data_list.append(data)

    if save:
        outdir = f"{config['base_dir']}/{config['name']}"
        os.makedirs(outdir, exist_ok=True)
        outfile = f"{outdir}/{config['name']}_{set_name}.pt"
        torch.save(data_list, outfile)
        print(f"Wrote {outfile}")

    return data_list


if __name__ == "__main__":
    with open("configs/LS_DNN.json") as f:
        config = json.load(f)

    ingress("test", config)
    # ingress("train", config)
