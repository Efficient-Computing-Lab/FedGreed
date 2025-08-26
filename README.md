# FedGreed: A Byzantine-Robust Loss-Based Aggregation Method for Federated Learning

## 📝 Overview

This repository contains the official implementation of our paper:

**Authors**: Emmanouil Kritharakis, Dusan Jakovetic, Antonios Makris, and Konstantinos Tserpes   
**Conference**: 3rd IEEE International Conference on Federated Learning Technologies and Applications @ FLTA (2025)  
**Paper**: [FedGreed: A Byzantine-Robust Loss-Based Aggregation Method for Federated Learning](https://arxiv.org/abs/2508.18060) <br>
**Citation**:
```bibtex
@misc{kritharakis2025fedgreedbyzantinerobustlossbasedaggregation,
      title={FedGreed: A Byzantine-Robust Loss-Based Aggregation Method for Federated Learning}, 
      author={Emmanouil Kritharakis and Antonios Makris and Dusan Jakovetic and Konstantinos Tserpes},
      year={2025},
      eprint={2508.18060},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.18060}, 
}
```

### 🔍 Abstract
> Federated Learning (FL) enables collaborative model training across multiple clients while preserving data privacy by keeping local datasets on-device. In this work, we address FL settings where
clients may behave adversarially, exhibiting Byzantine attacks, while the central server is trusted and equipped with a reference dataset. We propose FedGreed, a resilient aggregation strategy for 
federated learning that does not require any assumptions about the fraction of adversarial participants. FedGreed orders clients’ local model updates based on their loss metrics evaluated against a
trusted dataset on the server and greedily selects a subset of clients whose models exhibit the minimal evaluation loss. Unlike many existing approaches, our method is designed to operate reliably 
under heterogeneous (non-IID) data distributions, which are prevalent in real-world deployments. FedGreed exhibits convergence guarantees and bounded optimality gaps under strong adversarial behavior.
Experimental evaluations on MNIST, FMNIST, and CIFAR-10 demonstrate that our method significantly outperforms standard and robust federated learning baselines, such as Mean, Trimmed Mean, Median, 
Krum, and Multi-Krum, in the majority of adversarial scenarios considered, including label flipping and Gaussian noise injection attacks. All experiments were conducted using the Flower federated 
learning framework.

## 🗂️ Project Structure

```bash
.
├── src/                # Core Flower code
│   ├── strategies/     # Flower server strategies (FedGreed (ours), Mean, Trimmed Mean, Median, Krum, Multi Krum) 
├── configs/            # Configuration files for running and customizing experiments
├── data/               # Preprocessed partitioned data for FL clients
├── scripts/            # Data partitioning and simulation scripts
├── outputs/            # Timestamped outputs: logs, metrics, and best global model checkpoints per experiment
├── pyproject.toml      # Python project configuration and dependencies
└── README.md           # Project documentation and usage instructions
```

> **Note:** The `data` and `outputs` directories are created automatically upon executing the `partition_dataset` and `run_simulation` scripts, respectively.

## 🚀 Quick Start

### Prerequisites
Ensure you have [Poetry](https://python-poetry.org/docs/) installed and Python 3.12+ before proceeding.

### 1. Install Dependencies
Navigate to the root of the repository and install dependencies using Poetry:

```sh
poetry install
```

### 2. Preprocess and Distribute Data
Once dependencies are installed, preprocess the dataset and distribute it to the appropriate clients using the following command:

```sh
poetry run partition-dataset [OPTIONS]
```

#### Available Arguments
- `dataset_name` (Required): Name of the dataset.
- `--num_clients` (Optional): Number of federated learning (FL) clients.
- `--type` (Optional): Partitioning type, either `homogeneous` or `heterogeneous`. Default is `homogeneous`.
- `--alpha` (Optional): Alpha parameter for the Dirichlet distribution.

> **Note 1:** To reproduce the data partitioning used in the paper for the CIFAR-10, MNIST, and Fashion-MNIST datasets, execute the following commands:
> ```sh 
> poetry run partition-dataset CIFAR10 --num_clients=10 --type=heterogeneous --alpha=0.1
> poetry run partition-dataset MNIST --num_clients=10 --type=heterogeneous --alpha=0.1
> poetry run partition-dataset FMNIST --num_clients=10 --type=heterogeneous --alpha=0.1
>```
> After, completing the steps for high skewed data distribution, repeat the same for mild skewed data distribution with `alpha=1.0`

> **Note 2:** At this stage, you may proceed with Steps 3 and 4 to configure your own simulation, or alternatively, execute the automated script used for the experiments presented in the paper by running
the following command: 
> ```sh 
> poetry run sh run_experiments.sh 
> ```
> The results of each experiment will be stored in timestamped directories within the `outputs` folder.


### 3. Set Configuration File
Set the configuration YAML file as an environment variable:

```sh
export config_file_name=config
```
The configs directory contains predefined YAML configuration files designed for simulating various attacks, such as `config_data_attack` for data specific attacks 
and `config_model_attack` for model specific attacks.
To apply a specific configuration, simply update the corresponding environment variable with the desired YAML file name.

### 4. Run the Simulation
Execute the simulation script using Poetry:

```sh
poetry run simulation
```

