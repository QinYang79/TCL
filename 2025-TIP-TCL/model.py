import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_ 
from lib.VSEModel import VSEModel
from lib.SGRAF import SGRAF
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# label is an identity matrix with size K
def ce_loss(label, alpha, K,  lambda_=0.0001):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    B = lambda_ * KL(alp, K)
    return (A + B)

def KL(alpha, K):
    beta = torch.ones((1, K)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def MAE_loss(label, alpha, K, annealing_coef=0.0001):
    A = F.l1_loss(alpha / alpha.sum(dim=-1, keepdim=True), label)
    alp = (alpha - 1) * (1 - label) + 1
    B = annealing_coef * KL(alp, K)
    return (A + B)


def log_loss(label, alpha, K, annealing_coef=0.0001):
    A = torch.sum(label * (torch.log(alpha.sum(dim=-1, keepdim=True)) - torch.log(alpha)), dim=-1)
    alp = (alpha - 1) * (1 - label) + 1
    B = annealing_coef * KL(alp, K)
    return (A + B)


def ContrastiveLoss(scores, margin=0.2,max_violation=True):
    # compute image-sentence score matrix
    # scores = cosine_similarity_matrix(img_embs, cap_embs)
    diagonal = scores.diag().view(scores.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
    return cost_s.sum() + cost_im.sum()


class TCL(object):
    def __init__(self, opt, word2idx=None):
        self.opt = opt
        # based models
        self.models = dict()
        if opt.method in ['SAF','SGR','SAF_bert','SGR_bert','SAF_glove','SGR_glove']:
            self.models[0] = SGRAF(opt,word2idx)
            if self.opt.dual:
                self.models[1] = SGRAF(opt,word2idx)
        elif 'VSE' in opt.method:
            self.models[0] = VSEModel(opt, word2idx)
            if self.opt.dual:
                self.models[1] = VSEModel(opt, word2idx)
        else:
            raise ValueError('Invalid method of {}.'.format(opt.method))
    
        self.models_num = len(self.models)
        self.optimizer_dict = dict()
        self.params_dict = dict()
        self.step = 0
        self.logger = None

        for i in range(self.models_num):
            self.params_dict[i] = self.models[i].params
            if 'bert' in opt.method:
                decay_factor = 1e-4
                all_text_params = list(self.models[i].txt_enc.parameters())
                bert_params = list(self.models[i].txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                if 'VSE' in opt.method:
                    self.optimizer_dict[i] = {
                    'warmup': torch.optim.AdamW([
                        {'params': text_params_no_bert, 'lr': opt.learning_rate},
                        {'params': self.models[i].img_enc.parameters(), 'lr': opt.learning_rate},
                    ], lr=opt.learning_rate, weight_decay=decay_factor),
                    'train': torch.optim.AdamW([
                        {'params': text_params_no_bert, 'lr': opt.learning_rate},
                        {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                        {'params': self.models[i].img_enc.parameters(), 'lr': opt.learning_rate},
                    ], lr=opt.learning_rate, weight_decay=decay_factor)
                    }
                else:
                    self.optimizer_dict[i] = {
                    'warmup': torch.optim.AdamW([
                        {'params': text_params_no_bert, 'lr': opt.learning_rate},
                        {'params': self.models[i].img_enc.parameters(), 'lr': opt.learning_rate},
                        {'params': self.models[i].sim_enc.parameters(), 'lr': opt.learning_rate},
                    ], lr=opt.learning_rate, weight_decay=decay_factor),
                    'train': torch.optim.AdamW([
                        {'params': text_params_no_bert, 'lr': opt.learning_rate},
                        {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                        {'params': self.models[i].img_enc.parameters(), 'lr': opt.learning_rate},
                        {'params': self.models[i].sim_enc.parameters(), 'lr': opt.learning_rate},
                    ], lr=opt.learning_rate, weight_decay=decay_factor)
                    } 
            else:
                self.optimizer_dict[i] = torch.optim.AdamW(self.params_dict[i], lr=opt.learning_rate)

    def train_start(self):
        """switch to train mode
        """
        for i in range(self.models_num):
            self.models[i].train_start()
    def val_start(self):
        """switch to evaluate mode
        """
        for i in range(self.models_num):
            self.models[i].val_start()

    def state_dict(self):
        state_dict = [[self.models[i].state_dict()] for i in
                      range(self.models_num)]
        return state_dict

    def load_state_dict(self, state_dict):
        for i in range(self.models_num):
            self.models[i].load_state_dict(state_dict[i][0])

    def reset_grad(self):
        for i in range(self.models_num):
            if 'bert' in self.opt.method:
                self.optimizer_dict[i]['warmup'].zero_grad()
                self.optimizer_dict[i]['train'].zero_grad()
            else:
                self.optimizer_dict[i].zero_grad()

    def optimizer_step(self, epoch):
        lr = self.opt.learning_rate
        for i in range(self.models_num):
            if self.opt.grad_clip > 0:
                clip_grad_norm_(self.params_dict[i], self.opt.grad_clip)
            if 'bert' in self.opt.method:
                if epoch == 0:
                    self.optimizer_dict[i]['warmup'].step()
                else:
                    self.optimizer_dict[i]['train'].step()
                    lr = self.optimizer_dict[i]['train'].param_groups[0]['lr'] 
            else:
                self.optimizer_dict[i].step()
                lr = self.optimizer_dict[i].param_groups[0]['lr']
        self.logger.update('lr', lr)

    def adjust_bert(self, epoch):
        if 'bert' in self.opt.method:
            if epoch == 0:
                for i in range(self.models_num):
                    self.models[i].txt_enc.freeze_bert()
            else:
                for i in range(self.models_num):
                    self.models[i].txt_enc.unfreeze_bert()

    def forward_emb(self, images, captions, caption_lengths=None, image_lengths=None):
        img_embs = dict()
        cap_embs = dict()
        for i in range(self.models_num):
            if self.opt.method in ['SAF','SGR', 'SAF_bert','SGR_bert','SAF_glove','SGR_glove']:   
                img_emb, cap_emb =self.models[i].forward_emb( images, captions, caption_lengths)
            else:
                img_emb, cap_emb =self.models[i].forward_emb( images, captions, caption_lengths, image_lengths)

            img_embs[i] = img_emb
            cap_embs[i] = cap_emb
        return img_embs, cap_embs

    # l1/l2-norm
    def forward_decy(self, evidences, T=False):
        if T:
            e1 = evidences[0].t()
            e2 = evidences[1].t().detach()
        else:
            e1 = evidences[0].detach()
            e2 = evidences[1] 
        loss =  F.l1_loss(e1 / (e1 + 1).sum(1, keepdim=True),e2 / (e2 + 1).sum(1, keepdim=True)).mean()
        return loss

    def forward(self, images, captions, image_lengths=None, caption_lengths=None):
        return self.forward_emb(images, captions, caption_lengths, image_lengths)

    def train_emb(self, images, captions, image_lengths=None, caption_lengths=None, epoch=None):
        self.step += 1
        self.reset_grad() 
        if 'bert' in self.opt.method:
            self.adjust_bert(epoch)
            lambda1 =  min([1, 0.001 * epoch])
        else:
            lambda1 =  min([1, 0.005 * epoch])

        self.logger.update('step', self.step)
        self.logger.update('lambda1', lambda1)
        img_embs, cap_embs = self.forward_emb(images, captions, caption_lengths, image_lengths)
        K = captions.size(0)

        if not self.opt.dual:
            m_v = False
            if epoch > 0:
                m_v = True
            # train our VSE with MH loss    
            sims = self.models[0].forward_sim(img_embs[0], cap_embs[0], caption_lengths)
            loss = ContrastiveLoss(sims,max_violation=m_v)
            loss.backward()
            self.optimizer_step(epoch)
            self.logger.update('loss', loss.item(), 5)
        else:
            ground_truth = torch.eye(K).cuda()
            i2t_alpha_dict = dict()
            t2i_alpha_dict = dict()
            for i in range(self.models_num): 
                evidences = self.models[i].forward_sim(img_embs[i], cap_embs[i], caption_lengths, mode='not sim')[1]
                i2t_alpha_dict[i] = evidences + 1
                t2i_alpha_dict[i] = evidences.t() + 1
            loss_i2t = ce_loss(ground_truth, i2t_alpha_dict[0], K, lambda_=lambda1).mean()
            loss_t2i = ce_loss(ground_truth, t2i_alpha_dict[1], K, lambda_=lambda1).mean()

            loss = loss_i2t + loss_t2i 
            
            loss.backward()
            self.optimizer_step(epoch)
            if epoch > 0:
                loss_discrepancy_min_ = 0
                for i in range(self.opt.dis_k):
                    self.reset_grad()
                    evidences_dict = dict()
                    img_embs, cap_embs = self.forward_emb(images, captions, caption_lengths, image_lengths)
                    for i in range(self.models_num): 
                        evidences = self.models[i].forward_sim(img_embs[i], cap_embs[i], caption_lengths, mode='not sim')[1]
                        evidences_dict[i] = evidences
                    loss_discrepancy_min = self.forward_decy(evidences_dict)
                    loss_discrepancy_min += self.forward_decy(evidences_dict, T=True)
                    loss_discrepancy_min.backward()
                    self.optimizer_step(epoch)
                    loss_discrepancy_min_ += loss_discrepancy_min.item()
                self.logger.update('loss_c', loss_discrepancy_min_, K)
            self.logger.update('loss_i2t', loss_i2t.item(), K)
            self.logger.update('loss_t2i', loss_t2i.item(), K)
            self.logger.update('loss', loss.item(), K)
