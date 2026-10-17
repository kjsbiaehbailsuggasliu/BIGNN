import torch.nn as nn
from torch.nn.functional import dropout
from torch_geometric.nn import GCNConv, GraphConv
from BilinearGCN import BiGCNConv
from torch.nn import functional as F

from torch_geometric.transforms import NormalizeFeatures, Compose
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR, ReduceLROnPlateau

from torch.nn import Linear
import torch
from torch_geometric.data import Data
import random
import numpy as np
import torch

import os
import time  # Import time module for timing operations
import psutil  # Import psutil module for memory usage monitoring

from get_dataset import load_dataset


def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_memory_usage():
    """
    Get current process memory usage (MB)
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB


set_seed(42)


class BiGCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, num_nodes=None, gate=None):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            BiGCNConv(in_channels, hidden_channels, num_nodes=num_nodes, gate=gate)
            # GraphConv(in_channels, hidden_channels)
        )
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(
                BiGCNConv(hidden_channels, hidden_channels, num_nodes=num_nodes, gate=gate)
            )
        # Output layer (no activation at the end)
        # self.layers.append(
        #     BiGCNConv(hidden_channels, hidden_channels, num_nodes=num_nodes,
        #               lambda_struct=lambda_struct, gate=gate)
        # )

        # self.classifier = BiGCNConv(hidden_channels, out_channels, num_nodes=num_nodes, gate=gate)
        # self.classifier = GraphConv(hidden_channels, out_channels)
        self.classifier_linear = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_masks=None):
        """
        Forward pass through the network
        :param x: Node features
        :param edge_index: Graph edge indices
        :param edge_masks: None or List[Tensor] with length = num_layers, each Tensor shape = [E]
        """
        for i, conv in enumerate(self.layers):
            if edge_masks is not None:
                x = conv(
                    x,
                    edge_index,
                    edge_mask=edge_masks
                )
            else:
                x = conv(x, edge_index)

            x = F.dropout(x, p=0.6, training=self.training)

        x = self.classifier_linear(x)
        return F.log_softmax(x, dim=-1)


def main(data_name):
    # Set multiple random seeds
    seeds = [42]  # Can be adjusted based on requirements   42, 123, 456, 789, 999

    # Storage for results from all seeds
    all_results = {
        'val_accs': [],
        'test_accs': [],
        'best_test_accs': [],
        'epoch_times': [],  # Store epoch times for each seed
        'memory_usages': [],  # Store memory usage for each seed
        'all_epoch_times': [],  # Store time for each epoch
        'all_epoch_memories': []  # Store memory usage for each epoch
    }
    data_name = data_name  # 'cora', 'pubmed', 'citeseer' 'ba_shapes''cs'
    path = f"./checkpoints/{data_name}"
    os.makedirs(f"{path}", exist_ok=True)

    for seed in seeds:
        print(f"\n=== Training with seed {seed} ===")
        set_seed(seed)  # Set current seed

        # Load dataset
        data = load_dataset(data_name)

        # Initialize model
        model = BiGCNNet(
            in_channels=data.num_node_features,
            hidden_channels=146,  # 146
            out_channels=data.num_classes,
            num_layers=1,  # 3
            num_nodes=data.num_nodes,
            gate=None
        )

        checkpoint_path = f"{path}/best_bignn_{data_name}_seed{seed}_layer{model.num_layers}.pt"

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # lr=0.001
        # scheduler = MultiStepLR(optimizer=optimizer, milestones=[50, 100, 200, 300, 500, 800], gamma=0.5)
        # scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.9)
        # scheduler = CosineAnnealingLR(optimizer, T_max=100)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=30, factor=0.5)
        # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=30)

        loss_fn = torch.nn.NLLLoss()

        # Training function
        def train():
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            # scheduler.step()
            return loss.item()

        # Test function
        @torch.no_grad()
        def test():
            model.eval()
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

            accs = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                correct = pred[mask].eq(data.y[mask]).sum().item()
                accs.append(correct / mask.sum().item())
            return accs

        best_val_acc = 0
        patience = 100
        counter = 0

        # Record initial memory usage
        initial_memory = get_memory_usage()

        # Record training start time
        start_time = time.time()

        # Storage for epoch times and memory usage for current seed
        epoch_times = []
        epoch_memories = []

        # Training loop
        for epoch in range(1, 2001):
            epoch_start_time = time.time()  # Record start time for each epoch
            current_memory_before = get_memory_usage()  # Record memory usage before epoch

            loss = train()
            train_acc, val_acc, test_acc = test()

            epoch_end_time = time.time()  # Record end time for each epoch
            current_memory_after = get_memory_usage()  # Record memory after epoch

            epoch_time = epoch_end_time - epoch_start_time  # Calculate epoch duration
            epoch_memory_avg = (current_memory_before + current_memory_after) / 2  # Calculate average memory during epoch

            epoch_times.append(epoch_time)  # Store epoch time
            epoch_memories.append(epoch_memory_avg)  # Store epoch memory usage

            scheduler.step(val_acc)  # ReduceLROnPlateau

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                counter = 0

                # Save model with seed in filename
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_test_acc': best_test_acc,
                    'seed': seed
                }, checkpoint_path)
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % 50 == 0 or epoch == 1:  # Reduce printing frequency
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | "
                      f"Epoch Time: {epoch_time:.4f}s | Memory: {current_memory_after:.4f}MB")

        # Record total training end time
        end_time = time.time()
        total_training_time = end_time - start_time

        # Calculate average epoch time and memory usage
        avg_epoch_time = np.mean(epoch_times)
        avg_epoch_memory = np.mean(epoch_memories)

        # Record memory usage at end of training
        final_memory = get_memory_usage()
        memory_used = final_memory - initial_memory

        print(f"Seed {seed} - Best Val Acc: {best_val_acc:.4f} | Test Acc at best val: {best_test_acc:.4f}")
        print(f"Total training time for seed {seed}: {total_training_time:.4f} seconds")
        print(f"Average epoch time for seed {seed}: {avg_epoch_time:.6f} seconds")
        print(f"Average epoch memory for seed {seed}: {avg_epoch_memory:.4f} MB")
        print(f"Memory used during training for seed {seed}: {memory_used:.4f} MB")

        # Store final results for current seed
        all_results['val_accs'].append(best_val_acc)
        all_results['test_accs'].append(best_test_acc)
        all_results['best_test_accs'].append(best_test_acc)
        all_results['epoch_times'].append(avg_epoch_time)  # Add average epoch time to results list
        all_results['memory_usages'].append(memory_used)  # Add memory usage to results list
        all_results['all_epoch_times'].append(epoch_times)  # Add all epoch times to results list
        all_results['all_epoch_memories'].append(epoch_memories)  # Add all epoch memory usages to results list

    # Calculate and output statistical results
    print("\n=== Final Results Summary ===")
    print(f"Number of runs: {len(seeds)}")
    print(f"Validation Accuracy - Mean: {np.mean(all_results['val_accs']):.4f} ± {np.std(all_results['val_accs']):.4f}")
    print(f"Test Accuracy - Mean: {np.mean(all_results['test_accs']):.4f} ± {np.std(all_results['test_accs']):.4f}")
    print(
        f"Best Test Accuracy - Mean: {np.mean(all_results['best_test_accs']):.4f} ± {np.std(all_results['best_test_accs']):.4f}")

    # Combine all epoch times and memory usage across runs, then calculate mean and std
    all_epoch_times_combined = []
    all_epoch_memories_combined = []

    for times, memories in zip(all_results['all_epoch_times'], all_results['all_epoch_memories']):
        all_epoch_times_combined.extend(times)
        all_epoch_memories_combined.extend(memories)

    # Output mean and std of average epoch training time
    avg_epoch_time_mean = np.mean(all_epoch_times_combined)
    std_epoch_time = np.std(all_epoch_times_combined)
    print(f"All Epoch Training Time - Mean: {avg_epoch_time_mean:.6f} ± {std_epoch_time:.6f} seconds")

    # Output mean and std of average epoch memory usage
    avg_memory_usage = np.mean(all_epoch_memories_combined)
    std_memory_usage = np.std(all_epoch_memories_combined)
    print(f"All Epoch Memory Usage - Mean: {avg_memory_usage:.4f} ± {std_memory_usage:.4f} MB")

    # Output detailed results for each seed
    print("\nDetailed results per seed:")
    for i, seed in enumerate(seeds):
        print(f"Seed {seed:3d}: Val Acc: {all_results['val_accs'][i]:.4f}, "
              f"Test Acc: {all_results['test_accs'][i]:.4f}, "
              f"Avg Epoch Time: {all_results['epoch_times'][i]:.6f}s, "
              f"Mem: {all_results['memory_usages'][i]:.4f}MB")


if __name__ == "__main__":
    data_name_list = ['cora', 'pubmed', 'citeseer', 'ba_shapes', 'cs']  # Fixed list syntax error
    for data_name in data_name_list:
        main(data_name)

    main('ba_shapes')
