
import scanpy as sc
from deeprobust.image.attack.pgd import PGD
from deeprobust.image.config import attack_params
from deeprobust.image.utils import download_model
import torch
import deeprobust.image.netmodels.resnet as resnet
from torchvision import transforms,datasets
from deeprobust.image.attack.fgsm import FGSM
import numpy as np

class scRobustCell(object):

    def __init__(self, adata, device='cpu', eps = 1, seed=2023):
        self.adata = adata # adata is the output of the model you plan to benchmark.
        self.pvalue = 0.005
        self.device = device
        self.eps = eps
        self.seed = seed


    def scFGSM(self, input_data, label, model):
        adversary = FGSM(model, self.device)
        Adv_img = adversary.generate(input_data, label, epsilon = self.eps)

        return Adv_img

    def scPGD(self, input_data, label, model):
        adversary = PGD(model, self.device)
        Adv_img = adversary.generate(input_data, label, epsilon = self.eps)

        return Adv_img

    # single sample deepfool
    def scDeepFool(input_data, model):

        pert_sample = []
        for i in range(input_data.shape[0]):
            r_tot, loop_i, label, k_i, pert_image = deepfool(input_data[i:i+1,:], model)
            pert_sample.append(pert_image.view(-1).numpy())

        X_test_pert = torch.FloatTensor(np.array(pert_sample)).to(self.device)
        return X_test_pert

    def scRandomAttack(input_data, eplison, seed):
        np.random.seed(seed)
        rand_data = torch.FloatTensor(np.asarray(np.random.rand(input_data.shape[0], input_data.shape[1])))
        return input_data + eplison * rand_data

    def scMaxGene(adata, gene=None, scale=None):
        if gene == None:
            gene = np.random.choice(adata.var_names)
        
        if scale == None:
            adata[:,gene].X = max(adata.X)
        else:
            adata[:,gene].X = adata[:,gene].X * scale
        
        return torch.FloatTensor(adata.X)






