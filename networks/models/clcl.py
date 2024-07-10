import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from networks.models.common_layers import LinearLayer, Prediction
from networks.models.losses import contrastive_Loss, edl_digamma_loss, relu_evidence

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from networks.models.common_layers import LinearLayer, Prediction
from networks.models.losses import contrastive_Loss

class CLUECL3_3(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout, prediction_dicts):
        super().__init__()
        assert(len(in_dim)) == 3
        self.in_dim = in_dim
        self.views = 3  # views = 3, in_dim = [1000, 503, 1000]
        self.classes = num_class
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.att = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])  # fc [in_dim, in_dim]
        self.emb = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])  # fc [in_dim, hidden]
        self.aux_clf = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])  # fc [hidden, num_class]
        self.aux_conf = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])  # fc [hidden, 1]

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))  # [views*hidden, num_class]
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

        # inserted code
        
        def KL(alpha, c):
            beta = torch.ones((1, c)).cpu()
            S_alpha = torch.sum(alpha, dim=1, keepdim=True)
            S_beta = torch.sum(beta, dim=1, keepdim=True)
            lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
            lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
            dg0 = torch.digamma(S_alpha)
            dg1 = torch.digamma(alpha)
            kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
            return kl

 
        def ce_loss(p, alpha, c, global_step, annealing_step):
            S = torch.sum(alpha, dim=1, keepdim=True)
            E = alpha - 1
            label = F.one_hot(p, num_classes=c)
            A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

            annealing_coef = min(1, global_step / annealing_step)

            alp = E * (1 - label) + 1
            B = annealing_coef * KL(alp, c)

            return (A + B)
        
        # TODO
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none') 

        # CLUE
        self.a2b = Prediction([hidden_dim[-1]]+prediction_dicts[0])
        self.a2c = Prediction([hidden_dim[-1]]+prediction_dicts[0])
        self.b2a = Prediction([hidden_dim[-1]]+prediction_dicts[1])
        self.b2c = Prediction([hidden_dim[-1]]+prediction_dicts[1])
        self.c2a = Prediction([hidden_dim[-1]]+prediction_dicts[2])
        self.c2b = Prediction([hidden_dim[-1]]+prediction_dicts[2])

        # head for instance-level CL, TODO

    def forward(self, data_list, label=None, infer=False, aux_loss=False, lambda_al=0.05):
        att_score, feat_emb, aux_logit, aux_confidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            att_score[view] = torch.sigmoid(self.att[view](data_list[view]))
            feat_emb[view] = data_list[view] * att_score[view]
            feat_emb[view] = F.dropout(F.relu(self.emb[view](feat_emb[view])), self.dropout, training=self.training)
            aux_logit[view] = self.aux_clf[view](feat_emb[view])
            aux_confidence[view] = self.aux_conf[view](feat_emb[view])
            feat_emb[view] = feat_emb[view] * aux_confidence[view]

        MMfeature = torch.cat([i for i in feat_emb.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        loss_dict = {}
        # 1. Loss between classifier and gt
        MMLoss = torch.mean(self.criterion(MMlogit, label))
        loss_dict["clf"] = round(MMLoss.item(), 4)

        if aux_loss:
            aux_losses = []
            for view in range(self.views):
                # 2. loss of attention scores, l0 norm
                MMLoss = MMLoss+torch.mean(att_score[view]) # L0 for attention scores
                pred = F.softmax(aux_logit[view], dim=1)
                p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
                # 3. confidence loss between calculated confidence and max classes in the aux classifiers
                # 4. loss for aux classifier
                confidence_loss = torch.mean(F.mse_loss(aux_confidence[view].view(-1), p_target)+self.criterion(aux_logit[view], label))
                MMLoss = MMLoss+ lambda_al * confidence_loss
                aux_losses.append(round(lambda_al * confidence_loss.item(), 4))

            loss_dict["aux"] = aux_losses
        return MMLoss, MMlogit, loss_dict


    def train_missing_cg(self, data_list, mask, label=None, device=None, 
                         aux_loss=False, lambda_al=0.05,
                         cross_omics_loss=False, lambda_col=0.05,
                         constrastive_loss=False, lambda_cl=0.05):
        # select the complete samples
        x1_train, x2_train, x3_train = data_list
        flag = torch.LongTensor([1, 1, 1]).to(device)
        flag = (mask == flag).int()
        flag = ((flag[:, 1] + flag[:, 0] + flag[:, 2]) == 3)
        train_view1, train_view2, train_view3 = x1_train[flag], x2_train[flag], x3_train[flag]
        data_list = [train_view1, train_view2, train_view3]
        label = label[flag]

        # perform feature embedding
        att_score, feat_emb, aux_logit, aux_confidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            att_score[view] = torch.sigmoid(self.att[view](data_list[view]))
            feat_emb[view] = data_list[view] * att_score[view]
            feat_emb[view] = F.dropout(F.relu(self.emb[view](feat_emb[view])),self.dropout, training=self.training)
            aux_logit[view] = self.aux_clf[view](feat_emb[view])
            aux_confidence[view] = self.aux_conf[view](feat_emb[view])
            feat_emb[view] = feat_emb[view] * aux_confidence[view]

        MMfeature = torch.cat([i for i in feat_emb.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)

        loss_dict = {}
        # 1. Loss between classifier and gt
        MMLoss = torch.mean(self.criterion(MMlogit, label))
        loss_dict["clf"] = round(MMLoss.item(), 4)
        if aux_loss:
            aux_losses = []
            for view in range(self.views):
                # 2. loss of attention scores, l0 norm
                MMLoss = MMLoss+torch.mean(att_score[view]) # L0 for attention scores
                pred = F.softmax(aux_logit[view], dim=1)
                p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
                # 3. confidence loss between calculated confidence and max classes in the aux classifiers
                # 4. loss for aux classifier
                confidence_loss = torch.mean(F.mse_loss(aux_confidence[view].view(-1), p_target)+self.criterion(aux_logit[view], label))
                MMLoss = MMLoss+ lambda_al * confidence_loss
                aux_losses.append(round(lambda_al * confidence_loss.item(), 4))
            loss_dict['aux_clf'] = aux_losses

        if cross_omics_loss:
             # cross-omics unified embedding
            a2b, _ = self.a2b(feat_emb[0])
            a2c, _ = self.a2c(feat_emb[0])
            b2a, _ = self.b2a(feat_emb[1])
            b2c, _ = self.b2c(feat_emb[1])
            c2a, _ = self.c2a(feat_emb[2])
            c2b, _ = self.c2b(feat_emb[2])
            pre1 = F.mse_loss(a2b, feat_emb[1])
            pre2 = F.mse_loss(b2a, feat_emb[0])
            pre3 = F.mse_loss(a2c, feat_emb[2])
            pre4 = F.mse_loss(c2a, feat_emb[0])
            pre5 = F.mse_loss(b2c, feat_emb[2])
            pre6 = F.mse_loss(c2b, feat_emb[1])

            loss_co = lambda_col * (pre1 + pre2 + pre3 + pre4 + pre5 + pre6)
            MMLoss = MMLoss +  loss_co
            loss_dict['col'] = round(loss_co.item(), 4)

        if constrastive_loss:
            loss_cil_ab = contrastive_Loss(feat_emb[0], feat_emb[1], lambda_cl)
            loss_cil_ac = contrastive_Loss(feat_emb[0], feat_emb[2], lambda_cl)
            loss_cil_bc = contrastive_Loss(feat_emb[1], feat_emb[2], lambda_cl)
            # loss_cil_ab = InfoNceDist()
            loss_cil = lambda_cl * (loss_cil_ab + loss_cil_ac + loss_cil_bc)
            MMLoss = MMLoss + loss_cil
            loss_dict['cil'] = round(loss_cil.item(), 4)

        return MMLoss, MMlogit, loss_dict

    def encode(self, data_x, f_att, f_emb, f_aux_conf):
        att_score = torch.sigmoid(f_att(data_x))
        feat_emb = data_x * att_score
        feat_emb = F.dropout(F.relu(f_emb(feat_emb)), self.dropout, training=False)
        aux_confidence = f_aux_conf(feat_emb)
        feat_emb = feat_emb*aux_confidence
        return feat_emb

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


    def infer_on_missing(self, data_list, mask, device):
        # make sure eval all modules before
        x1_train, x2_train, x3_train = data_list
        a_idx_eval = mask[:, 0] == 1
        b_idx_eval = mask[:, 1] == 1
        c_idx_eval = mask[:, 2] == 1
        a_missing_idx_eval = mask[:, 0] == 0
        b_missing_idx_eval = mask[:, 1] == 0
        c_missing_idx_eval = mask[:, 2] == 0

        # latent_code_x_eval, store information
        latent_code_a_eval = torch.zeros(x1_train.shape[0], self.hidden_dim[-1]).to(device)
        latent_code_b_eval = torch.zeros(x2_train.shape[0], self.hidden_dim[-1]).to(device)
        latent_code_c_eval = torch.zeros(x3_train.shape[0], self.hidden_dim[-1]).to(device)

        # predict on each omics without missing
        a_latent_eval = self.encode(x1_train[a_idx_eval], self.att[0], self.emb[0], self.aux_conf[0])
        b_latent_eval = self.encode(x2_train[b_idx_eval], self.att[1], self.emb[1], self.aux_conf[1])
        c_latent_eval = self.encode(x3_train[c_idx_eval], self.att[2], self.emb[2], self.aux_conf[2])

        if a_missing_idx_eval.sum() != 0:
            ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
            ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
            ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

            ano_bonlyhas = self.encode(x2_train[ano_bonlyhas_idx], self.att[1], self.emb[1], self.aux_conf[1])
            ano_bonlyhas, _ = self.b2a(ano_bonlyhas)
                           
            ano_conlyhas = self.encode(x3_train[ano_conlyhas_idx], self.att[2], self.emb[2], self.aux_conf[2])
            ano_conlyhas, _ = self.c2a(ano_conlyhas)

            ano_bcbothhas_1 = self.encode(x2_train[ano_bcbothhas_idx], self.att[1], self.emb[1], self.aux_conf[1])
            ano_bcbothhas_2 = self.encode(x3_train[ano_bcbothhas_idx], self.att[2], self.emb[2], self.aux_conf[2])
            ano_bcbothhas = (self.b2a(ano_bcbothhas_1)[0] + self.c2a(ano_bcbothhas_2)[0]) / 2.0

            latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
            latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
            latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas
        
        if b_missing_idx_eval.sum() != 0:
            bno_aonlyhas_idx = b_missing_idx_eval * a_idx_eval * ~c_idx_eval
            bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval
            bno_acbothhas_idx = b_missing_idx_eval * a_idx_eval * c_idx_eval

            bno_aonlyhas = self.encode(x1_train[bno_aonlyhas_idx], self.att[0], self.emb[0], self.aux_conf[0])
            bno_aonlyhas, _ = self.a2b(bno_aonlyhas)

            bno_conlyhas = self.encode(x3_train[bno_conlyhas_idx], self.att[2], self.emb[2], self.aux_conf[2])
            bno_conlyhas, _ = self.c2b(bno_conlyhas)

            bno_acbothhas_1 = self.encode(x1_train[bno_acbothhas_idx], self.att[0], self.emb[0], self.aux_conf[0])
            bno_acbothhas_2 = self.encode(x3_train[bno_acbothhas_idx], self.att[2], self.emb[2], self.aux_conf[2])
            bno_acbothhas = (self.a2b(bno_acbothhas_1)[0] + self.c2b(bno_acbothhas_2)[0]) / 2.0

            latent_code_b_eval[bno_aonlyhas_idx] = bno_aonlyhas
            latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas
            latent_code_b_eval[bno_acbothhas_idx] = bno_acbothhas

        if c_missing_idx_eval.sum() != 0:
            cno_aonlyhas_idx = c_missing_idx_eval * a_idx_eval * ~b_idx_eval
            cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval
            cno_abbothhas_idx = c_missing_idx_eval * a_idx_eval * b_idx_eval

            cno_aonlyhas = self.encode(x1_train[cno_aonlyhas_idx], self.att[0], self.emb[0], self.aux_conf[0])
            cno_aonlyhas, _ = self.a2c(cno_aonlyhas)

            cno_bonlyhas = self.encode(x2_train[cno_bonlyhas_idx], self.att[1], self.emb[1], self.aux_conf[1])
            cno_bonlyhas, _ = self.b2c(cno_bonlyhas)

            cno_abbothhas_1 = self.encode(x1_train[cno_abbothhas_idx], self.att[0], self.emb[0], self.aux_conf[0])
            cno_abbothhas_2 = self.encode(x2_train[cno_abbothhas_idx], self.att[1], self.emb[1], self.aux_conf[1])
            cno_abbothhas = (self.a2c(cno_abbothhas_1)[0] + self.b2c(cno_abbothhas_2)[0]) / 2.0

            latent_code_c_eval[cno_aonlyhas_idx] = cno_aonlyhas
            latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas
            latent_code_c_eval[cno_abbothhas_idx] = cno_abbothhas

        latent_code_a_eval[a_idx_eval] = a_latent_eval
        latent_code_b_eval[b_idx_eval] = b_latent_eval
        latent_code_c_eval[c_idx_eval] = c_latent_eval

        latent_fusion_train = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval], dim=1)
        MMlogit = self.MMClasifier(latent_fusion_train)
        return MMlogit









class CLUECL3_2(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout, prediction_dicts):
        super().__init__()
        # assert(len(in_dim)) == 3
        assert(len(in_dim)) == 2
        self.in_dim = in_dim
        # self.views = 3   views = 3, in_dim = [1000, 503, 1000]
        self.views = 2
        self.classes = num_class
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.att = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])  # fc [in_dim, in_dim]
        self.emb = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])  # fc [in_dim, hidden]
        self.aux_clf = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])  # fc [hidden, num_class]
        self.aux_conf = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])  # fc [hidden, 1]

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))  # [views*hidden, num_class]
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

        
        # TODO change loss function
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none') 

        # CLUE
        self.a2b = Prediction([hidden_dim[-1]]+prediction_dicts[0])
        self.a2c = Prediction([hidden_dim[-1]]+prediction_dicts[0])
        self.b2a = Prediction([hidden_dim[-1]]+prediction_dicts[1])
        self.b2c = Prediction([hidden_dim[-1]]+prediction_dicts[1])
        self.c2a = Prediction([hidden_dim[-1]]+prediction_dicts[2])
        self.c2b = Prediction([hidden_dim[-1]]+prediction_dicts[2])

        # head for instance-level CL, TODO

    def forward(self, data_list, label=None, infer=False, aux_loss=False, lambda_al=0.05):
        att_score, feat_emb, aux_logit, aux_confidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            att_score[view] = torch.sigmoid(self.att[view](data_list[view]))
            feat_emb[view] = data_list[view] * att_score[view]
            feat_emb[view] = F.dropout(F.relu(self.emb[view](feat_emb[view])), self.dropout, training=self.training)
            aux_logit[view] = self.aux_clf[view](feat_emb[view])
            aux_confidence[view] = self.aux_conf[view](feat_emb[view])
            feat_emb[view] = feat_emb[view] * aux_confidence[view]

        MMfeature = torch.cat([i for i in feat_emb.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        loss_dict = {}
        # 1. Loss between classifier and gt
        MMLoss = torch.mean(self.criterion(MMlogit, label))
        loss_dict["clf"] = round(MMLoss.item(), 4)

        if aux_loss:
            aux_losses = []
            for view in range(self.views):
                # 2. loss of attention scores, l0 norm
                MMLoss = MMLoss+torch.mean(att_score[view]) # L0 for attention scores
                pred = F.softmax(aux_logit[view], dim=1)
                p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
                # 3. confidence loss between calculated confidence and max classes in the aux classifiers
                # 4. loss for aux classifier
                confidence_loss = torch.mean(F.mse_loss(aux_confidence[view].view(-1), p_target)+self.criterion(aux_logit[view], label))
                MMLoss = MMLoss+ lambda_al * confidence_loss
                aux_losses.append(round(lambda_al * confidence_loss.item(), 4))

            loss_dict["aux"] = aux_losses
        return MMLoss, MMlogit, loss_dict


    def train_missing_cg(self, data_list, mask, label=None, device=None, 
                         aux_loss=False, lambda_al=0.05,
                         cross_omics_loss=False, lambda_col=0.05,
                         constrastive_loss=False, lambda_cl=0.05):
        # select the complete samples
        # x1_train, x2_train, x3_train = data_list
        x1_train, x2_train = data_list
        flag = torch.LongTensor([1, 1, 1]).to(device)
        flag = (mask == flag).int()
        flag = ((flag[:, 1] + flag[:, 0] + flag[:, 2]) == 3)
        # train_view1, train_view2, train_view3 = x1_train[flag], x2_train[flag], x3_train[flag]
        train_view1, train_view2 = x1_train[flag], x2_train[flag]
        # data_list = [train_view1, train_view2, train_view3]
        data_list = [train_view1, train_view2]
        label = label[flag]

        # perform feature embedding
        att_score, feat_emb, aux_logit, aux_confidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            att_score[view] = torch.sigmoid(self.att[view](data_list[view]))
            feat_emb[view] = data_list[view] * att_score[view]
            feat_emb[view] = F.dropout(F.relu(self.emb[view](feat_emb[view])),self.dropout, training=self.training)
            aux_logit[view] = self.aux_clf[view](feat_emb[view])
            aux_confidence[view] = self.aux_conf[view](feat_emb[view])
            feat_emb[view] = feat_emb[view] * aux_confidence[view]

        MMfeature = torch.cat([i for i in feat_emb.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)

        loss_dict = {}
        # 1. Loss between classifier and gt
        MMLoss = torch.mean(self.criterion(MMlogit, label))
        loss_dict["clf"] = round(MMLoss.item(), 4)
        if aux_loss:
            aux_losses = []
            for view in range(self.views):
                # 2. loss of attention scores, l0 norm
                MMLoss = MMLoss+torch.mean(att_score[view]) # L0 for attention scores
                pred = F.softmax(aux_logit[view], dim=1)
                p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
                # 3. confidence loss between calculated confidence and max classes in the aux classifiers
                # 4. loss for aux classifier
                confidence_loss = torch.mean(F.mse_loss(aux_confidence[view].view(-1), p_target)+self.criterion(aux_logit[view], label))
                MMLoss = MMLoss+ lambda_al * confidence_loss
                aux_losses.append(round(lambda_al * confidence_loss.item(), 4))
            loss_dict['aux_clf'] = aux_losses

        if cross_omics_loss:
             # cross-omics unified embedding
            a2b, _ = self.a2b(feat_emb[0])
            a2c, _ = self.a2c(feat_emb[0])
            b2a, _ = self.b2a(feat_emb[1])
            b2c, _ = self.b2c(feat_emb[1])
            # c2a, _ = self.c2a(feat_emb[2])
            # c2b, _ = self.c2b(feat_emb[2])
            pre1 = F.mse_loss(a2b, feat_emb[1])
            pre2 = F.mse_loss(b2a, feat_emb[0])
            # pre3 = F.mse_loss(a2c, feat_emb[2])
            # pre4 = F.mse_loss(c2a, feat_emb[0])
            # pre5 = F.mse_loss(b2c, feat_emb[2])
            # pre6 = F.mse_loss(c2b, feat_emb[1])

            # loss_co = lambda_col * (pre1 + pre2 + pre3 + pre4 + pre5 + pre6)
            loss_co = lambda_col * (pre1 + pre2)
            MMLoss = MMLoss +  loss_co
            loss_dict['col'] = round(loss_co.item(), 4)

        if constrastive_loss:
            loss_cil_ab = contrastive_Loss(feat_emb[0], feat_emb[1], lambda_cl)
            # loss_cil_ac = contrastive_Loss(feat_emb[0], feat_emb[2], lambda_cl)
            # loss_cil_bc = contrastive_Loss(feat_emb[1], feat_emb[2], lambda_cl)
            # loss_cil_ab = InfoNceDist()
            # loss_cil = lambda_cl * (loss_cil_ab + loss_cil_ac + loss_cil_bc)
            loss_cil = lambda_cl * (loss_cil_ab)
            MMLoss = MMLoss + loss_cil
            loss_dict['cil'] = round(loss_cil.item(), 4)

        return MMLoss, MMlogit, loss_dict

    def encode(self, data_x, f_att, f_emb, f_aux_conf):
        att_score = torch.sigmoid(f_att(data_x))
        feat_emb = data_x * att_score
        feat_emb = F.dropout(F.relu(f_emb(feat_emb)), self.dropout, training=False)
        aux_confidence = f_aux_conf(feat_emb)
        feat_emb = feat_emb*aux_confidence
        return feat_emb

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


    def infer_on_missing(self, data_list, mask, device):
        # make sure eval all modules before
        # x1_train, x2_train, x3_train = data_list
        x1_train, x2_train = data_list
        a_idx_eval = mask[:, 0] == 1
        b_idx_eval = mask[:, 1] == 1
        c_idx_eval = mask[:, 2] == 1
        a_missing_idx_eval = mask[:, 0] == 0
        b_missing_idx_eval = mask[:, 1] == 0
        c_missing_idx_eval = mask[:, 2] == 0

        # latent_code_x_eval, store information
        latent_code_a_eval = torch.zeros(x1_train.shape[0], self.hidden_dim[-1]).to(device)
        latent_code_b_eval = torch.zeros(x2_train.shape[0], self.hidden_dim[-1]).to(device)
        # latent_code_c_eval = torch.zeros(x3_train.shape[0], self.hidden_dim[-1]).to(device)

        # predict on each omics without missing
        a_latent_eval = self.encode(x1_train[a_idx_eval], self.att[0], self.emb[0], self.aux_conf[0])
        b_latent_eval = self.encode(x2_train[b_idx_eval], self.att[1], self.emb[1], self.aux_conf[1])
        # c_latent_eval = self.encode(x3_train[c_idx_eval], self.att[2], self.emb[2], self.aux_conf[2])

        if a_missing_idx_eval.sum() != 0:
            ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
            ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
            ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

            ano_bonlyhas = self.encode(x2_train[ano_bonlyhas_idx], self.att[1], self.emb[1], self.aux_conf[1])
            ano_bonlyhas, _ = self.b2a(ano_bonlyhas)
                           
            # ano_conlyhas = self.encode(x3_train[ano_conlyhas_idx], self.att[2], self.emb[2], self.aux_conf[2])
            # ano_conlyhas, _ = self.c2a(ano_conlyhas) 

            ano_bcbothhas_1 = self.encode(x2_train[ano_bcbothhas_idx], self.att[1], self.emb[1], self.aux_conf[1])
            # ano_bcbothhas_2 = self.encode(x3_train[ano_bcbothhas_idx], self.att[2], self.emb[2], self.aux_conf[2])
            # ano_bcbothhas = (self.b2a(ano_bcbothhas_1)[0] + self.c2a(ano_bcbothhas_2)[0]) / 2.0

            latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
            # latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
            # latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas
        
        if b_missing_idx_eval.sum() != 0:
            bno_aonlyhas_idx = b_missing_idx_eval * a_idx_eval * ~c_idx_eval
            bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval
            bno_acbothhas_idx = b_missing_idx_eval * a_idx_eval * c_idx_eval

            bno_aonlyhas = self.encode(x1_train[bno_aonlyhas_idx], self.att[0], self.emb[0], self.aux_conf[0])
            bno_aonlyhas, _ = self.a2b(bno_aonlyhas)

            # bno_conlyhas = self.encode(x3_train[bno_conlyhas_idx], self.att[2], self.emb[2], self.aux_conf[2])
            # bno_conlyhas, _ = self.c2b(bno_conlyhas)

            bno_acbothhas_1 = self.encode(x1_train[bno_acbothhas_idx], self.att[0], self.emb[0], self.aux_conf[0])
            # bno_acbothhas_2 = self.encode(x3_train[bno_acbothhas_idx], self.att[2], self.emb[2], self.aux_conf[2])
            # bno_acbothhas = (self.a2b(bno_acbothhas_1)[0] + self.c2b(bno_acbothhas_2)[0]) / 2.0

            latent_code_b_eval[bno_aonlyhas_idx] = bno_aonlyhas
            # latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas
            # latent_code_b_eval[bno_acbothhas_idx] = bno_acbothhas

        if c_missing_idx_eval.sum() != 0:
            cno_aonlyhas_idx = c_missing_idx_eval * a_idx_eval * ~b_idx_eval
            cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval
            cno_abbothhas_idx = c_missing_idx_eval * a_idx_eval * b_idx_eval

            cno_aonlyhas = self.encode(x1_train[cno_aonlyhas_idx], self.att[0], self.emb[0], self.aux_conf[0])
            cno_aonlyhas, _ = self.a2c(cno_aonlyhas)

            cno_bonlyhas = self.encode(x2_train[cno_bonlyhas_idx], self.att[1], self.emb[1], self.aux_conf[1])
            cno_bonlyhas, _ = self.b2c(cno_bonlyhas)

            cno_abbothhas_1 = self.encode(x1_train[cno_abbothhas_idx], self.att[0], self.emb[0], self.aux_conf[0])
            cno_abbothhas_2 = self.encode(x2_train[cno_abbothhas_idx], self.att[1], self.emb[1], self.aux_conf[1])
            cno_abbothhas = (self.a2c(cno_abbothhas_1)[0] + self.b2c(cno_abbothhas_2)[0]) / 2.0

            # latent_code_c_eval[cno_aonlyhas_idx] = cno_aonlyhas
            # latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas
            # latent_code_c_eval[cno_abbothhas_idx] = cno_abbothhas

        latent_code_a_eval[a_idx_eval] = a_latent_eval
        latent_code_b_eval[b_idx_eval] = b_latent_eval
        # latent_code_c_eval[c_idx_eval] = c_latent_eval

        # latent_fusion_train = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval], dim=1)
        latent_fusion_train = torch.cat([latent_code_a_eval, latent_code_b_eval], dim=1)
        MMlogit = self.MMClasifier(latent_fusion_train)
        return MMlogit
    

# model for single view
# NOTE this is also used for multiview

class Classifier(nn.Module):
    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i+1]))
        self.fc.append(nn.Linear(classifier_dims[self.num_layers-1], classes))
        self.fc.append(nn.Softplus())



    def forward(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h


"""
# trying new MLP layers:
class Classifier(nn.Module):
    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i+1]))
            self.fc.append(nn.ReLU())
            self.fc.append(nn.BatchNorm1d(classifier_dims[i+1]))
            self.fc.append(nn.Dropout(0.5))
        
        self.fc.append(nn.Linear(classifier_dims[self.num_layers-1], classes))
        self.fc.append(nn.Softmax(dim=1)) 



    def forward(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h
"""


class SingleViewData(Dataset):

    def __init__(self, root, train=True, cv=0):
        """
        :param root: data name and path
        :param train: load training set or test set
        """
        super(SingleViewData, self).__init__()
        train = "tr" if train else "te"
        # self.X = pd.read_csv(f"{root}/view1_{train}_x{cv}.csv").to_numpy()
        # y = np.loadtxt(f"{root}/{train}_y_{cv}.txt")

        # fix file being read
        self.X = pd.read_csv(f"{root}/1_{train}.csv").to_numpy()
        y = np.loadtxt(f"{root}/labels_{train}.csv")

        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        self.y = y

    def __getitem__(self, index):
        data = self.X[index].astype(np.float32)
        target = self.y[index]
        return data, target

    def __len__(self):
        return self.X.shape[0]

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes).to(labels.device)
    return y[labels]

class EV_SV(nn.Module):

    def __init__(self, classes, classifier_dims, use_uncertainty=True, loss="digamma"):
        super(EV_SV, self).__init__()
        self.num_classes = classes
        self.classifier = Classifier(classifier_dims, self.num_classes)
        
        # types of loss functions used digamma
        if use_uncertainty:
            if loss == "digamma":
                criterion = edl_digamma_loss
            else:
                raise ValueError("[!] not a supported loss")
            """elif loss == "log":
                criterion = edl_log_loss
            elif loss == "mse":
                criterion = edl_mse_loss"""
            
        else:
            criterion = nn.CrossEntropyLoss()
        
        self.criterion = criterion
        self.num_classes = classes
    
    def forward(self, X, y, epoch):
        outputs = self.classifier(X)
        y = one_hot_embedding(y, self.num_classes)
        loss = self.criterion(outputs, y, epoch, num_classes=self.num_classes, annealing_step=50)

        evidence = relu_evidence(outputs)
        alpha = evidence + 1
        u = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)

        return outputs, loss, u
