
import scanpy as sc
from deeprobust.image.attack.pgd import PGD
from deeprobust.image.config import attack_params
from deeprobust.image.utils import download_model
import torch
import deeprobust.image.netmodels.resnet as resnet
from torchvision import transforms,datasets
from deeprobust.image.attack.fgsm import FGSM
import numpy as np
import pickle 
import torch.nn as nn
import scanpy as sc
import torch
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, DICE, Random, MinMax
from deeprobust.graph.global_attack import DICE

def deepfool(input_data, model, num_classes=10, overshoot=0.02, max_iter=10):
    image = input_data 
    net = model

    f_image = net.forward(image).data.numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.detach().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = torch.tensor(pert_image[None, :],requires_grad=True)
    
    fs = net.forward(x[0])
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.numpy().copy()

        for k in range(1, num_classes):
            
            #x.zero_grad()
            
            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = torch.tensor(pert_image, requires_grad=True)
        fs = net.forward(x[0])
        k_i = np.argmax(fs.data.numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image


def single_attack(X_tr, y_tr, model, attack, eps=10, device = 'cuda', seed=2023, num_classes=10):
    if attack == 'PGD':
        adversary = PGD(model, device)
        Adv_img = adversary.generate(X_tr, y_tr, epsilon = eps)
        return Adv_img.detach().to('cpu')
    elif attack == 'FGSM':
        adversary = FGSM(model, device)
        Adv_img = adversary.generate(X_tr, y_tr, epsilon = eps)
        return Adv_img.detach().to('cpu')
    elif attack == 'Random':
        np.random.seed(seed)
        rand_data = torch.FloatTensor(np.asarray(np.random.rand(X_tr.shape[0], X_tr.shape[1])))
        return X_tr + eps * rand_data
    elif attack == 'Deepfool':
        X_train = X_tr.to('cpu')
        copyed_model = pickle.loads(pickle.dumps(model))
        # These lines are used to remove the batch norm layers.
        copyed_model.shared_layers[1] = nn.Identity()
        copyed_model.shared_layers[5] = nn.Identity()
        
        pert_sample = []
        # This may take long time.
        for i in range(X_train.shape[0]):
            r_tot, loop_i, label, k_i, pert_image = deepfool(X_tr[i:i+1,:], copyed_model.to('cpu'), num_classes=num_classes)
            pert_sample.append(pert_image.view(-1).numpy())
        X_test_pert = torch.FloatTensor(np.array(pert_sample))
        return X_test_pert
    else:
        print("Please include correct attack methods")
        raise Exception()

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
    def scDeepFool(self, input_data, model, num_classes=10, overshoot=0.02, max_iter=10):

        pert_sample = []
        for i in range(input_data.shape[0]):
            r_tot, loop_i, label, k_i, pert_image = deepfool(input_data[i:i+1,:], model, num_classes, overshoot, max_iter)
            pert_sample.append(pert_image.view(-1).numpy())

        X_test_pert = torch.FloatTensor(np.array(pert_sample)).to(self.device)
        return X_test_pert

    def scRandomAttack(self, input_data):
        np.random.seed(self.seed)
        rand_data = torch.FloatTensor(np.asarray(np.random.rand(input_data.shape[0], input_data.shape[1])))
        return input_data + self.eps * rand_data
    
    def scEnsemble

    def scMaxGene(self, gene=None, scale=None):
        adata = self.adata
        if gene == None:
            gene = np.random.choice(adata.var_names)
        
        if scale == None:
            adata[:,gene].X = max(adata.X)
        else:
            adata[:,gene].X = adata[:,gene].X * scale
        
        return torch.FloatTensor(adata.X).to(self.device)



class graphRobustCell(object):

    def __init__(self, adata, device='cpu', num_neig=10, eps = 1, seed=2023):
        self.adata = adata # adata is the output of the model you plan to benchmark.
        sc.pp.neighbors(adata, use_rep = 'spatial', n_neighbors=num_neig)
        self.adj = (adata.obsp['distances']>0)*1 # an example of creating adj matrix
        self.num_neig = num_neig
        self.device = device
        self.eps = eps
        self.seed = seed


    def fit_GCN(self, input_data, adj, labels, idx_train):
        if adj == None:
            adj = self.adj
        device = self.device 
        features = input_data.to(device)
        adj = adj.to(device)

        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
                        with_relu=False, device=self.device)
        surrogate = surrogate.to(device)
        surrogate.fit(features, adj, labels, idx_train = idx_train, train_iters=1000)
        return surrogate

    def metaattack(self, surrogate, input_data, adj, labels, idx_train):
        if adj == None:
            adj = self.adj
        features = input_data
        model = Metattack(model=surrogate.to(self.device), nnodes=adj.shape[0], feature_shape=features.shape, device=device)
        model = model.to(self.device)
        perturbations = int(0.05 * (adj.sum() // 2))
        model.attack(features, adj, labels, idx_train, idx_test, perturbations, ll_constraint=False)
        modified_adj = model.modified_adj

        return modified_adj

    def Minmaxattack(self, surrogate, input_data, adj, labels, idx_train, n_perturbations = 10):
        if adj == None:
            adj = self.adj
        features = input_data
        model = MinMax(model=surrogate, nnodes=adj.shape[0], loss_type='CE', device=self.device).to(self.device)
        model.attack(features, adj.toarray(), labels, idx_train, n_perturbations=n_perturbations)
        modified_adj = model.modified_adj

        return modified_adj

    def Randomattack(self, adj, n_perturbations = 10):
        if adj == None:
            adj = self.adj
        model = Random()
        model.attack(adj, n_perturbations=n_perturbations)
        modified_adj = model.modified_adj

        return modified_adj


    def DICEattack(self, adj, labels, idx_train, n_perturbations = 10):
        model = DICE()
        model.attack(adj, labels, n_perturbations=10)
        modified_adj = model.modified_adj
        return modified_adj

    # We recommend running evaluations on cpu
    def evaluate(self, surrogate, input_data, modified_adj, labels, idx_train, return_dict=False):
        features = input_data.to('cpu')
        surrogate = surrogate.to('cpu')
        _,predicted = torch.max(torch.softmax(surrogate.predict(features, modified_adj), dim=1), 1)
        results = sklearn.metrics.classification_report(labels[train_obs], predicted.numpy()[train_obs], return_dict=return_dict)

        print(results)

        return results
