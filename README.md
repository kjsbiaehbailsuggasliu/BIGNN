# Bilinear Graph Convolutional Network (BiGCN)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch 2.2.2](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.6.1-green.svg)](https://pytorch-geometric.readthedocs.io/)

## 📖 项目概述

**BiGCN** 是一个创新的图神经网络架构，通过双线性交互和动态边权重机制增强图表示学习能力。该模型具有以下核心特点：

- **🔄 双线性特征变换**：通过门控双线性层实现节点特征间的交互学习
- **⚡ 动态边权重**：根据节点特征动态调整图中边的重要性
- **🔍 对称交互矩阵分解**：提供模型可解释性，支持特征级交互分析
- **📊 多数据集支持**：包括 Cora、PubMed、CiteSeer、CoauthorCS 和 BA-Shapes 等标准图数据集

## 📁 文件结构

```bash
.
├── BilinearGCN.py           # 核心模型定义（双线性GCN层）
├── environment.yml          # Conda环境配置文件
├── get_dataset.py           # 数据集加载和预处理工具
├── interpretation.py        # 模型解释和可视化工具
├── train_main.py            # 模型训练和评估主程序
├── checkpoints/             # 模型检查点保存目录
│   ├── cora/               # Cora数据集模型
│   ├── pubmed/             # PubMed数据集模型
│   └── ...                 # 其他数据集模型
└── README.md               # 项目说明文档
```

## ⚙️ 环境配置

### 方法一：使用 Conda 环境（推荐）

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate GNN_Base
```

### 方法二：手动安装主要依赖

```bash
# 创建基础环境
conda create -n GNN_Base python=3.9
conda activate GNN_Base

# 安装 PyTorch（CUDA 11.8）
conda install pytorch=2.2.2 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装 PyTorch Geometric
conda install pyg -c pyg

# 安装其他依赖
pip install numpy pandas matplotlib scikit-learn seaborn tqdm wandb
```

## 🚀 快速开始

### 1. 数据准备

```python
from get_dataset import load_dataset

# 加载数据集（支持多个标准数据集）
data = load_dataset('cora')  # 可选：'cora', 'pubmed', 'citeseer', 'ba_shapes', 'cs'
print(f"数据集: {data.num_nodes}个节点, {data.num_edges}条边, {data.num_classes}个类别")
```

### 2. 模型训练

```bash
# 运行训练脚本（自动训练所有支持的数据集）
python train_main.py
```

### 3. 模型解释

```bash
# 运行解释分析脚本
python interpretation.py
```

**注意**：运行解释前需要：
1. 训练模型并保存检查点
2. 修改 `interpretation.py` 中的配置：
   ```python
   CKPT_PATH = "checkpoints/ba_shapes/best_bignn_ba_shapes_seed42_layer1.pt"
   NODE_ID = 0  # 要解释的节点ID
   ```
## 📄 许可证

本项目采用 [​​Apache License 2.0​](LICENSE) 开源许可证。

## 🤝 贡献

