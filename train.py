from datasets import EdgeDataset, EdgeDataBatch
from torch.utils.data import DataLoader

# PyG graph-level GNN inputs --> edge-level DNN inputs
train_loader = DataLoader(
    EdgeDataset(train_loader), batch_size=config.train.train_batch_size, shuffle=True,
    collate_fn=lambda batch: EdgeDataBatch(batch)
)
test_loader = DataLoader(
    EdgeDataset(test_loader), batch_size=config.train.test_batch_size, shuffle=True,
    collate_fn=lambda batch: EdgeDataBatch(batch)
)

for event_i, data in enumerate(train_loader):
    # Log start
    data = data.to(device)
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_attr)
    y, output = data.y, output.squeeze(1)

