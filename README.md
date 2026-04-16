# Deep Learning Project Baseline

Welcome to the Deep Learning Project Baseline! This repository serves as an intuitive and straightforward template designed specifically to help beginners set up modular, customizable, and reproducible deep learning pipelines using PyTorch.

By following this template, you'll learn how to organize your code efficiently, handle hyperparameters properly, and track your experiments.

## 📂 Project Structure

```text
DL-project-baseline/
├── data/
│   ├── data.mat       # Your input data subset (or custom dataset generation)
│   └── labels.mat     # Ground truth labels
├── results/           # Automatically generated directory containing saved model states & logs
├── code/              # Main directory containing execution scripts
│   ├── main.py        # The primary script used to train and test your model
│   └── src/           # The source code supporting the project
│       ├── dataloading.py  # Data ingestion and Dataset preparation logic
│       ├── loss_fn.py      # Custom loss functions (e.g., RMSELoss)
│       ├── model.py        # Neural network architectures (e.g., MLP)
│       ├── trainer.py      # The training loop abstracting boilerplates
│       ├── callbacks.py    # Standard tracking events (EarlyStopping, Checkpointing)
│       ├── utils.py        # Reproducibility utilities, logging, and data helpers
│       └── visualization.py# Scripts to plot training curves
```

## 🧠 Core Modules Explanation

### 1. The `main.py` entry point
The `main.py` is what you will execute to start an experiment. It acts as the orchestrator by:
- Defining the configuration parameters (hyperparameters, network sizes) visually at the bottom of the script.
- Importing components from the `src` directory.
- Loading the dataset via `dataloading.py` and preparing the `DataLoader`.
- Invoking the training process using the `Trainer`, and then picking the best hyperparameters to execute testing.

### 3. `src/dataloading.py` - Loading Data
Handles loading the `.mat` data formats and creating the PyTorch custom `Dataset`. 

### 4. `src/model.py` - Network Architectures
Houses standard neural networks. Currently, it supports a modular `MLP` feature extractor capable of adjusting to any hidden layer depth based on the lists you provide in `main.py`.

### 5. `src/trainer.py` - Boilerplate Abstraction
Responsible for keeping `main.py` clean by absorbing the typical PyTorch `forward`, `loss calculation`, `backward` passes, and the metric tracking. 

## 🚀 How to Use

To use this template, first ensure you have the required packages standard to PyTorch deep learning (`torch`, `pandas`, `scipy`, `scikit-learn`, `numpy`).

### 1. Customize and Train
Open the `main.py` file and navigate to the bottom inside the `if __name__ == '__main__':` block. 
Here, you'll intuitively see the `hyperparams` and `model_params` data structures. You can manually adjust elements directly, such as increasing the number of `epochs` or adding another layer like `[128, 64, 32]` into the `units` argument.

Once your structure is configured, run the following command in the terminal:
```bash
cd code
python main.py
```

## 📊 Outputs & Tracking

Whenever you execute a run, the progress automatically records into your `results/` folder! You will find:
- **Weights:** Safely checkpointed model state `Model_{fold}.pt` saved when validation improves.
- **Excel Logs:** Convenient `.xlsx` tracking sheets recording validation and training loss along your hyperparameter choices for easy comparatives!

Have fun experimenting and building!
