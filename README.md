# RobustCell

# Installation

To use our attack-defense framework, the users need to install the following packages:

Install [Deeprobust](https://github.com/DSE-MSU/DeepRobust?tab=readme-ov-file) by

```
pip install deeprobust
pip install scipy==1.7.3 #(to avoid one bug in PGD)
```

Due to the bugs exisiting in Deeprobust, we highly recommend you to utilize the codes from our repo (known as the folder **deeprobust**).

Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) by

```
conda install pyg -c pyg
```

In addition to train a neural-network-based classifier, the users are encouraged to install [pytorch_lightning](https://lightning.ai/docs/pytorch/stable/) by

```
pip install lightning
```

After finishing these steps, please refer **README_setup.md** to install robustcell.


To run our evaluations based on [scGPT](https://github.com/bowang-lab/scGPT), the users need to install scGPT by

```
pip install scgpt "flash-attn<1.0.5"  # optional, recommended
# As of 2023.09, pip install may not run with new versions of the google orbax package, if you encounter related issues, please use the following command instead:
# pip install scgpt "flash-attn<1.0.5" "orbax<0.1.8"
pip install wandb
```

To run our evaluations based on [Geneformer](https://huggingface.co/ctheodoris/Geneformer/tree/main), the users need to install Geneformer by

```
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .
```

We also offer a conda environment to install the packages needed for running attack and defense methods, which is known as **robustcell.yml**. This environment can be installed by

```
conda env create -f robustcell.yml
conda activate deeprobust_gcn
```

# Tutorial

We offer a simple tutorial for running one attack method shown in the folder **Tutorial**. More cases can be found in the documents from Deeprobust and Adversarial_training.

# Framework

Please check the files under folder **robustcell** for details. We offer attacks for both cells and graphs, as well as defense methods for both cells and graphs :)

# Experiments

We offer examples of our experiments, shown in the folder **Experiments**, which contains notebooks for computation-oriented attacks, biology-oriented attacks, defense based on marker gene selection, defense based on adversarial training, graph-based attacks, and defense based on graph defenses.

# Acknowledgement

We appreciate the great codes offered by Deeprobust and [Adversarial_training](https://github.com/MehrshadSD/robustness-interpretability)!

# Citation

```
@article{liu2024robustcell,
  title={RobustCell: A Model Attack-Defense Framework for Robust Transcriptomic Data Analysis},
  author={Liu, Tianyu and Xiao, Yijia and Luo, Xiao and Zhao, Hongyu},
  journal={bioRxiv},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```