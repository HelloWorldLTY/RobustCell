
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

    def __init__(self, adata):
        self.adata = adata # adata is the output of the model you plan to benchmark.
        self.pvalue = 0.005


    def scFGSM(input_data, label, model, device, eps):
        adversary = FGSM(model, device)
        Adv_img = adversary.generate(input_data, label, epsilon = eps)

        return Adv_img

    def scPGD(input_data, label, model, device, eps):
        adversary = PGD(model, device)
        Adv_img = adversary.generate(input_data, label, epsilon = eps)

        return Adv_img


    def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=10):

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

    # single sample deepfool
    def scDeepFool(input_data, model, device='cpu'):

        pert_sample = []
        for i in range(input_data.shape[0]):
            r_tot, loop_i, label, k_i, pert_image = deepfool(input_data[i:i+1,:], model)
            pert_sample.append(pert_image.view(-1).numpy())

        X_test_pert = torch.FloatTensor(np.array(pert_sample))
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






class RobustCell(object):

    def __init__(self, adata):
        self.adata = adata # adata is the output of the model you plan to benchmark.
        self.pvalue = 0.005

    def evaluation_bec(self, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scGPT'):
        results = eval_scib_metrics(self.adata,batch_key,label_key, emb_name)
        return results
    

    def evaluation_cta_gfp(self, pred_label, true_label):
        results = classification_report(pred_label, true_label, digits=4)
        return results
    
    def evaluation_perturb_pred(self, pred_model, true_result): #assume the outputs are both in AnnData format. Rows are cells while columns are genes.
        cor_total = calculate_correlation_metric(pred_model.X.T, true_result.X.T)
        return {"correlation":cor_total / len(pred_model.X.T)}
    
    def evaluation_perturb_pred_gearsofficial(self, gears_model, pred_model ):
        from gears.inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis
        test_res = evaluate(gears_model.dataloader['test_loader'], pred_model)
        test_metrics, test_pert_res = compute_metrics(test_res)
        return test_metrics
    
    def evaluation_imputation_scrna(self, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scGPT'):
        results = eval_scib_metrics_onlybio(self.adata,batch_key,label_key, emb_name)
        return results
    
    def evaluation_imputation_spatial(self, adata_sp):
        adata_imp_new = self.adata[:, adata_sp.var_names]
        cor_list = []
        pval_list = []
        for item in adata_sp.var_names:
            adata1 = adata_sp[:,item]
            adata2 = adata_imp_new[:,item]
            cor, pval = scipy.stats.pearsonr(np.array(adata1.X.todense().T)[0], np.array(adata2.X.T)[0]) # for this step, please check the data form
            cor_list.append(cor)
            pval_list.append(pval)

        adata_imp_new.var['cor'] = cor_list 
        adata_imp_new.var['pval'] = pval_list

        mean_cor = np.mean(adata_imp_new.var['cor'].values)

        avg_sig = np.sum(adata_imp_new.var['pval'].values<self.pvalue)/len(adata_imp_new.var['pval'].values)
        return {"mean_cor":mean_cor, "avg_sign":avg_sig} 
    
    def evaluation_simulation(self, batch_key = 'batch',label_key = 'celltype', isbatch = True, emb_name = 'X_scGPT'):

        if isbatch:
            results = eval_scib_metrics(self.adata,batch_key,label_key, emb_name)
            return results 
        else:
            results = eval_scib_metrics_onlybio(self.adata,batch_key,label_key, emb_name)
            return results             
