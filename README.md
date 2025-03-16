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

- Step 1: The "Relevant Attribute Values" sheet of the "Complete_Coal_Plant_Analysis_umich.xlsx" dataset has been used for generating the dataset with unprocessed objective values "coaldata.xlsx". This dataset is located in the "Coal_to_Nuclear/Processing" directory. Run the data processing script with either serial and parallel processing script:

```bash
nohup python coal_single_parallel_run.py &
```

Or:

```bash
nohup python coal_single_parallel_run.py &
```

These scripts create the "recorder.txt" file by using the dataset.

- Step 2: The "Coal_to_Nuclear/Processing/PCA" subdirectory is used for creating the results with PCA dimensionality reduction method. 

```bash
nohup python coaltonuclear_PCA.py.py &
```

- Step 3: The "Coal_to_Nuclear/Processing/CCA" subdirectory is used for creating the results with CCA dimensionality reduction method. 

```bash
nohup python coaltonuclear_CCA.py.py &
```

- Step 4: The Jupyter notebook "post_processor.ipynb" in the "Coal_to_Nuclear/Postprocessing" directory is used for analyzing the data processing results in the "recorder.txt", for identifying the best CPP sites, and visualizing the results to create the figures presented in the paper.

```bash
nohup papermill create_diagonal_validation_plot.ipynb create_diagonal_validation_plot_out.ipynb &
```
