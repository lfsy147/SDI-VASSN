# Spatial domain identification based on variational autoencoder and single sample network (SDI-VASSN).

![image](https://github.com/lfsy147/SDI-VASSN/tree/main/SDI-VASSN/Utilities/figure.png)

# Installation

## Install stMVC


#### 1. Grab source code of stMVC

```
git clone https://github.com/lfsy147/SDI-VASSN.git

cd SDI-VASSN
```

#### 2. Install SDI-VASSN in the virtual environment by conda 

* Firstly, install conda: https://docs.anaconda.com/anaconda/install/index.html

* Then, automatically install all used packages (described by "used_package.txt") for stMVC in a few mins.

```
conda create -n SDI-VASSN python=3.6.12 pip

source activate

conda activate stMVC

pip install -r used_package.txt
```

# Quick start

## Input

download:https://drive.google.com/file/d/16MCJbCnuYqvWCxgfv-3vQJZHZzniTrpI/view?usp=drive_link

```
unzip stMVC_test_data.zip
```
## Run
### Step 1. Preprocess raw data
```
python Preprcessing_stMVC.py --basePath ./stMVC_test_data/DLPFC_151673/ 
```
### Step 2 Use CSN in MATLAB
```
Processing_csn
```
### Step 3. Manual cell segmentation (for IDC dataset)
```
python Image_cell_segmentation.py --basePath ./stMVC_test_data/IDC/ --jsonFile tissue_hires_image.json
```
### Step 4. Run VAE and stMVC model
```
python VAE train
python stMVC_model.py --basePath ./stMVC_test_data/DLPFC_151673/ --fusion_type Attention
```
## Output

## Output file will be saved for further analysis:

* GAT_2-view_model.pth: a saved model for reproducing results.

* GAT_2-view_robust_representation.csv: robust representations for latter clustering, visualization, and data denoising.
