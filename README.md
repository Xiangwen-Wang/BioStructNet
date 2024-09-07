# BioStructNet
BioStructNet is a structure-based deep learning model designed to enhance the prediction of enzyme-substrate interactions, particularly focusing on biocatalysis. It includes transfer learning approaches for small, function-based datasets. The parameters are validated through molecular docking and MD simulations. 
Here's a detailed `README.md` template for your GitHub repository:

---

## Overview

BioStructNet is a deep learning framework developed to predict enzyme-substrate interactions by leveraging protein and ligand structural data. It employs Graph Neural Networks (GNNs) and Transformer-based modules to process protein structures and ligand molecules. This repository contains code for training and transfer learning.

BioStructNet is especially useful for predicting biocatalytic activities (e.g., Kcat values) and enhancing function-based prediction with small dataset, such as in plastic degradation by enzymes like *Candida antarctica* lipase B (CalB).

## Features

- **GNN-based Protein and Ligand Encoders**: Encodes protein structures using contact maps and ligand molecules in SMILES format.
- **BCN Interaction Module**: Captures local dependencies between protein-ligand pairs using Bilinear Co-Attention Networks (BCN).
- **Transformer-based Interaction Module**: Handles long-range dependencies for protein-ligand interactions using multi-head attention.
- **Transfer Learning**: Fine-tune pre-trained models on small function-specific datasets to enhance prediction accuracy with three fine-tuning methods, block, free, and LoRa.
- **Molecular Docking Validation**: Integrates docking simulations to compare learned interaction maps with physical conformations of protein-ligand complexes.

---

## Repository Contents

- `src/source/BCN`and `src/source/Trans`: Contains all core scripts for model training and evaluation with BCN interaction module or Transformer-based interaction module.
  - `feature.py`: Graph Neural Network encoder for processing protein and ligand structures.
  - `module.py`: Implements the traing model for local protein-ligand dependencies.
  - `main.py`: training set.
  - `configs.py`: hyperparameter configuration.
  - `submit.sh`: submit jobs on HPC.

- `src/bootstrapping`: Contains all core scripts for transfer learning model training and evaluation.    
  - `transfer_model.py`: Code for applying different fine-tuning methods to the model.

- `data_preprocessing`: Script to clean and preprocess enzyme-ligand datasets (e.g., BRENDA and SABIO-RK datasets).

- `data/`: Contains links or instructions to download the datasets used, including enzyme datasets (Kcat, CalB).
- `model/`: Contains pretrained Kcat BCN and Trans models.
- `results/`: Stores output results, including validation metrics, plots, and docking figures.
  
---

## Getting Started

### Prerequisites

Ensure that you have the following installed:

- Python 3.10+
- `environment.yml`

You may also need molecular docking software (e.g., AutoDock) and molecular dynamics tools like AMBER for the docking validation.

### data preprocessing

Download PDB file from PDBBANK.
Prepare the pdb files of the database by RosettaCM (https://docs.rosettacommons.org/docs/latest/application_documentation/structure_prediction/RosettaCM)
or Alphafold2 (https://alphafold.ebi.ac.uk/).
Some PDB files have been saved in 'zip' in `data/` folder

### Model Training and predicting

**Training**:
   Run the following command to train the BioStructNet model:
   ```
   sbatch submit.sh
   ```
   You can specify model parameters (e.g., learning rate) in the `config.py` file.
   
**Predicting**:
   ```
   python predict.py
   ```

---

## Results

Key result figures include:

- **RMSE** and **R²** for regression tasks on datasets Kcat.
- **AUC**, **Accuracy**, and **TPR-FPR** for CalB classification task after agumentation.
- Docking and MD simulation validation to compare physical and computational results.

---

## References

If you use BioStructNet in your research, please cite the following:

- Wang, X., Zhou, J., Quinn, D., Moody, T., Huang, M. "Enhancing Function-Based Biocatalysis Prediction through a Structure-Based Deep Learning Approach."




