# Multi-Objective Site Selection for Coal to Nuclear Power Plant Transition
The dataset and codes used in the comparison of US Coal Power Plant (CPP) sites for Nuclear Power Plant (NPP) siting. 

## Paper

Erdem, O. F., Radaideh, M. I. (2024). Multi-Objective Site Selection for Coal to Nuclear Power Plant Transition. In 2024 International Congress on Advances in Nuclear Power Plants (ICAPP), Las Vegas, NV. https://www.ans.org/pubs/proceedings/article-56308/


## 🛠️ Environment Installation

To set up the environment for this project, follow these steps:


```bash
# 1. Create a new conda environment with the .yml file
conda create  -f C2N_env_environment.yml

# 2. Activate the environment
conda activate C2N_env

# 3. Install Jupyter and Papermill for notebook execution
pip install jupyter papermill
```

## How to generate the results

The dataset has been acquired from: https://www.sciencedirect.com/science/article/pii/S2352484724002993

- Step 1: The 5th sheet of the Excel dataset has been used for generating the dataset with unprocessed objective values. This dataset is located in the "Processing" folder, named as "coaldata.xlsx". Serial and parallel run scripts are used for generating the recorder.txt file from this dataset. 

- Step 2: The "PCA" and "CCA" subdirectories are used for creating the results with dimensionality reduction. 

- Step 3: The Jupyter notebook in the "Postprocessing" directory is used for analyzing the data processing results, for identifying the best CPP sites, and visualizing the results to create the figures presented in the paper.