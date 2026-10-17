import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.transforms import NormalizeFeatures, RemoveIsolatedNodes


def load_dataset(name, root="data"):
    """
    Load dataset with specified name
    name: Dataset name ('Cora', 'PubMed', 'CiteSeer', 'CoauthorCS')
    root: Root directory for data storage
    """
    if name.lower() in ['cora', 'pubmed', 'citeseer']:
        dataset = Planetoid(root=f"{root}/{name}", name=name, transform=NormalizeFeatures())
        data = dataset[0]
        if name.lower() == 'citeseer':
            data = RemoveIsolatedNodes()(data)
        data.num_classes = dataset.num_classes

    elif name.lower() == 'ba_shapes':

        dataset = torch.load('./data/ba_shapes/processed/dataBA_shapes.pt')
        data = dataset[0]
        data.num_classes = 4
        data = random_split(data)


    elif name.lower() == 'cs':
        dataset = Coauthor(root=f"{root}/CoauthorCS", name='cs', transform=NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_split(data)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return data


def random_split(data, ratio=[0.6, 0.2, 0.2]):
    """
    Randomly split dataset into train/validation/test sets according to given ratios
    data: Input graph data
    ratio: Split ratios for train/validation/test [train_ratio, val_ratio, test_ratio]
    """

    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)

    train_idx = np.random.choice(data.num_nodes, int(data.num_nodes * ratio[0]), replace=False)
    residue = np.array(list(set(range(data.num_nodes)) - set(train_idx)))
    val_idx = np.random.choice(residue, int(data.num_nodes * ratio[1]), replace=False)
    test_idx = np.array(list(set(residue) - set(val_idx)))

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(val_idx).intersection(set(test_idx))) == 0
    assert len(set(train_idx).intersection(set(test_idx))) == 0

    return data
