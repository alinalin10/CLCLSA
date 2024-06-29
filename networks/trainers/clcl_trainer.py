import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pprint

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils.data_utils import one_hot_tensor, prepare_trte_data, get_mask
from networks.models.clcl import CLUECL3
from datetime import datetime
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_mask_wrapper(n_views, data_len, missing_rate):
    success = False
    while not success:
        try:
            mask = get_mask(n_views, data_len, missing_rate)
            success = True
        except:
            success = False
    
    return mask

class CLCLSA_Trainer(object):

    def __init__(self, params):
        self.params = params
        self.device = self.params['device']
        self.__init_dataset__()
        self.model = CLUECL3(self.dim_list, self.params['hidden_dim'], self.num_class, self.params['dropout'], self.params['prediction'])
        # print(self.dim_list)
        # self.dim_list is [2000, 2000, 548] 
        # which is 3 views

        # print(self.num_class)
        # 2 num_class

        # error len(classifier_dims) classifier_dims is self.dim_list[i], the first parameter, object of type int has no len()
        # error solution: turn [2000, 2000, 548] -> [[240], [76], [216], [47], [64], [6]] format instead of flat list
        # SOLVED
        self.dim_list = [[x] for x in self.dim_list]
        # print(self.dim_list)

        # new
        self.preds2, self.gts2, self.us2 = [], [], []



        # self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.num_class) for i in range(self.views)])
        self.Classifiers = nn.ModuleList([Classifier(self.dim_list[i], self.num_class) for i in range(len(self.dim_list))])


    def __init_dataset__(self):
        self.data_tr_list, self.data_test_list, self.trte_idx, self.labels_trte = prepare_trte_data(self.params['data_folder'], True)
        self.labels_tr_tensor = torch.LongTensor(self.labels_trte[self.trte_idx["tr"]])
        num_class = len(np.unique(self.labels_trte))
        self.onehot_labels_tr_tensor = one_hot_tensor(self.labels_tr_tensor, num_class)
        self.labels_tr_tensor = self.labels_tr_tensor.cpu()
        self.onehot_labels_tr_tensor = self.onehot_labels_tr_tensor.cpu()
        dim_list = [x.shape[1] for x in self.data_tr_list]
        self.dim_list = dim_list
        self.num_class = num_class
        print("[x] number of num_class = ", self.num_class)

        if self.params['missing_rate'] > 0.:
            mask = get_mask(3, self.data_tr_list[0].shape[0], self.params['missing_rate'])
            mask = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(self.device)
            x1_train = self.data_tr_list[0] * torch.unsqueeze(mask[:, 0], 1)
            x2_train = self.data_tr_list[1] * torch.unsqueeze(mask[:, 1], 1)
            x3_train = self.data_tr_list[2] * torch.unsqueeze(mask[:, 2], 1)
            self.mask_train = mask
            self.data_tr_list = [x1_train, x2_train, x3_train]
            #mask = get_mask(3, self.data_test_list[0].shape[0], self.params['missing_rate'])
            mask = get_mask_wrapper(3, self.data_test_list[0].shape[0], self.params['missing_rate'])
            mask = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(self.device)
            x1_test = self.data_test_list[0] * torch.unsqueeze(mask[:, 0], 1)
            x2_test = self.data_test_list[1] * torch.unsqueeze(mask[:, 1], 1)
            x3_test = self.data_test_list[2] * torch.unsqueeze(mask[:, 2], 1)
            self.mask_test = mask
            self.data_test_list = [x1_test, x2_test, x3_test]
    
    # new DS_Combin from TMC
    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))

                # figure out error
                # error: CLCLSA_Trainer object has no attribute classes
                # SOLVED change self.classes to self.num_class
                u[v] = self.num_class/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.num_class, 1), b[1].view(-1, 1, self.num_class))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # u_a represents uncertainty
            #print(u_a)

            # calculate new S
            S_a = self.num_class / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a, u_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a, u_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a, u_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a, u_a


    def train(self):

        exp_name = os.path.join(self.params['exp'], f"{self.params['data_folder']}_{datetime.utcnow().strftime('%B_%d_%Y_%Hh%Mm%Ss')}")
        os.makedirs(exp_name, exist_ok=True)
        with open(os.path.join(exp_name, 'config.json'), 'w') as fp:
            json.dump(self.params, fp, indent=4)

        self.model.cpu()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'], weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.params['step_size'], gamma=0.2)
        global_acc = 0.
        best_eval = []
        print("\nTraining...")
        for epoch in tqdm(range(self.params['num_epoch']+ 1)):
            print_loss = True if epoch % self.params['test_inverval'] == 0 else False
            self.train_epoch(print_loss)

            self.scheduler.step()
            if epoch % self.params['test_inverval'] == 0:
                te_prob, alpha_a, u_a = self.test_epoch()
                if not np.any(np.isnan(te_prob)):
                    print("\nTest: Epoch {:d}".format(epoch))
                    if self.num_class == 2:
                        # uncertainty
                        
                        acc = accuracy_score(self.labels_trte[self.trte_idx["te"]], te_prob.argmax(1))
                        f1 = f1_score(self.labels_trte[self.trte_idx["te"]], te_prob.argmax(1))
                        auc = roc_auc_score(self.labels_trte[self.trte_idx["te"]], te_prob[:, 1])
                        print(f"Test ACC: {acc:.5f}, F1: {f1:.5f}, AUC: {auc:.5f}, Uncertainty:")
                        self.us2.extend(u_a)
                        self.gts2.extend(self.labels_trte[self.trte_idx["te"]])
                        self.preds2.extend(te_prob.argmax(1))
                        print(len(self.us2))
                        print(len(self.gts2))
                        print(len(self.preds2))

                        # print(self.gts2)
                        

                        

                        #preds2 = [list(arr) for arr in self.preds2]

                        #preds2 = [tensor.item() for tensor in self.preds2]
                        #print(preds2)

                        #gts2 = [tensor.item() for tensor in self.gts2]
                        #print(gts2)

                        # ValueError: can only convert an array of size 1 to a Python scalar
                        

                        # print(alpha_a)
                        if acc > global_acc:
                            global_acc = acc
                            best_eval = [acc, f1, auc]
                            self.save_checkpoint(exp_name)
                    else:
                        acc = accuracy_score(self.labels_trte[self.trte_idx["te"]], te_prob.argmax(1))
                        f1w = f1_score(self.labels_trte[self.trte_idx["te"]], te_prob.argmax(1), average='weighted')
                        f1m = f1_score(self.labels_trte[self.trte_idx["te"]], te_prob.argmax(1), average='macro')
                        print(f"Test ACC: {acc:.5f}, F1 weighted : {f1w:.5f}, F1 macro: {f1m:.5f}")
                        if acc > global_acc:
                            global_acc = acc
                            best_eval = [acc, f1w, f1m]
                            self.save_checkpoint(exp_name)
        
        
        return best_eval, exp_name

    def testing(self):
        return self.preds2, self.gts2, self.us2

    def train_epoch(self, print=False):
        self.model.train()
        self.optimizer.zero_grad()
        if self.params['missing_rate'] > 0:
            loss, _, loss_dict = self.model.train_missing_cg(self.data_tr_list, self.mask_train, self.labels_tr_tensor, self.device,
                aux_loss=self.params['lambda_al']>0, lambda_al=self.params['lambda_al'],
                cross_omics_loss=self.params['lambda_co']>0, lambda_col=self.params['lambda_co'],
                constrastive_loss=self.params['lambda_cl']>0, lambda_cl=self.params['lambda_cl'])
        else:
            loss, _, loss_dict = self.model(self.data_tr_list, self.labels_tr_tensor, 
                                            aux_loss=self.params['lambda_al']>0, lambda_al=self.params['lambda_al'])
        
        if print:
            pprint.pprint(loss_dict)
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            if self.params['missing_rate'] > 0:
                logit = self.model.infer_on_missing(self.data_test_list, self.mask_test, self.device)
            else:
                logit = self.model.infer(self.data_test_list)
            prob = F.softmax(logit, dim=1).data.cpu().numpy()
        
        # new: setting alpha to evidence
        evidence = self.infer_evidence(self.data_test_list)
        alpha = dict()
        for v_num in range(len(self.data_test_list)):
            alpha[v_num] = evidence[v_num] + 1
        
        # using DS_Combin to obtain uncertainty and alpha_a
        alpha_a, u_a = self.DS_Combin(alpha)
        
        # print(len(u_a))
        # u_a is len = 153
        # TODO figure out what this represents

        return prob, alpha_a, u_a

    def save_checkpoint(self, checkpoint_path, filename="checkpoint.pt"):
        os.makedirs(checkpoint_path, exist_ok=True)
        filename = os.path.join(checkpoint_path, filename)
        torch.save(self.model.state_dict(), filename)
    
    # new: returns evidence of every view of data, test data = self.data_test_list
    def infer_evidence(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """

        evidence = dict()

        # TODO figure out views: maybe use data_utils.py get_mask
        views = len(self.dim_list)

        for v_num in range(views):
            evidence[v_num] = self.Classifiers[v_num](input[v_num])
        return evidence 
    
    def plot(self):
        us2 = [tensor.item() for tensor in self.us2]
        preds2 = self.preds2
        gts2 = self.gts2
        df_stage2 = pd.DataFrame({"uncertainty": us2, "pred": preds2, "gt": gts2})
        labels2 = []
        print("working") 
        for g in gts2:
            if g:
                labels2.append("AD")
            else:
                labels2.append("Normal Control")
        df_stage2["label"] = labels2

        fig, ax = plt.subplots()
        sns.histplot(df_stage2, x="uncertainty", hue="label", element="step", ax=ax)
        ax.set_title("Uncertainty histogram for 3 Views")
        ax.set_xlim(0,1)
        plt.show()
        

    
class Classifier(nn.Module):
    def __init__(self, classifier_dims, num_class):
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i+1]))
        self.fc.append(nn.Linear(classifier_dims[self.num_layers-1], num_class))
        self.fc.append(nn.Softplus())


    def forward(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h