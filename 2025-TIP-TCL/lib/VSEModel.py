# coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from transformers import BertModel
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchtext

def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)

        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features, pool_weights

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe

class TwoLayerMLP(nn.Module):
    def __init__(self, num_features, hid_dim, out_dim, return_hidden=False):
        super().__init__()
        self.return_hidden = return_hidden
        self.model = nn.Sequential(
            nn.Linear(num_features, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if not self.return_hidden:
            return self.model(x)
        else:
            hid_feat = self.model[:2](x)
            results = self.model[2:](hid_feat)
            return hid_feat, results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B*N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x



def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        # print(k,length)
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)

# From https://github.com/woodfrog/vse_infty
# VSE_infty
class EncoderImageAggr(nn.Module):
    def __init__(self, opt, type='infty'):
        super(EncoderImageAggr, self).__init__()
        self.type=type
        self.embed_size = opt.embed_size  
        self.img_dim = opt.img_dim
        self.no_imgnorm = opt.no_imgnorm
        self.fc = nn.Linear(self.img_dim,  self.embed_size)
        self.mlp = MLP(self.img_dim,  self.embed_size // 2,  self.embed_size, 2)
        self.gpool = GPO(32, 32)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)
        features = self.mlp(images) + features

        if self.type=='infty':
            features, pool_weights = self.gpool(features, image_lengths)
        else:
            features = maxk_pool1d_var(features, 1, features.size(1), image_lengths) # max-pool

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

# From https://github.com/woodfrog/vse_infty
# VSE_infty Language Model with BERT
class EncoderText_BERT(nn.Module):
    def __init__(self, opt,type='infty'):
        super(EncoderText_BERT, self).__init__()
        self.type =type
        self.embed_size = opt.embed_size
        self.no_txtnorm = opt.no_txtnorm
        # download bert-base-uncased from huggingface
        self.bert = BertModel.from_pretrained('./bert-base-uncased')
        self.linear = nn.Linear(768, self.embed_size)
        self.gpool = GPO(32, 32)

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
        cap_len = lengths

        cap_emb = self.linear(bert_emb)

        if self.type == 'infty':
            pooled_features, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))
        else:
            pooled_features = maxk_pool1d_var(cap_emb, 1, cap_emb.size(1), cap_len) # max-pool

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)

        return pooled_features

# From https://github.com/woodfrog/vse_infty
# VSE_infty Language Model with GRU
class EncoderText_GRU(nn.Module):
    def __init__(self, opt, word2idx=None, type='infty'):
        super(EncoderText_GRU, self).__init__()
        self.embed_size = opt.embed_size
        self.no_txtnorm = opt.no_txtnorm
        self.type=type
        # word embedding
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)

        self.gpool = GPO(32, 32)
        # caption embedding
        self.rnn = nn.GRU(opt.word_dim, opt.embed_size, opt.num_layers, batch_first=True, bidirectional=opt.use_bi_gru)
        self.init_weights()

    def init_weights(self): 
        self.embed.weight.data.uniform_(-0.1, 0.1)
         
    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)

        self.rnn.flatten_parameters()
        packed = pack_padded_sequence(x_emb, lengths.cpu(), batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        if self.type == 'infty':
            pooled_features, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))
        else:
            pooled_features = maxk_pool1d_var(cap_emb, 1, cap_emb.size(1), cap_len) # avg-pool

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)

        return pooled_features

# RNN GloVe Based Language Model
class GloveEmb(nn.Module):
    def __init__(
            self,
            num_embeddings,
            glove_dim,
            glove_path,
            add_rand_embed=False,
            rand_dim=None,
            **kwargs
    ):
        super(GloveEmb, self).__init__()

        self.num_embeddings = num_embeddings
        self.add_rand_embed = add_rand_embed
        self.glove_dim = glove_dim
        self.final_word_emb = glove_dim

        # word embedding
        self.glove = nn.Embedding(num_embeddings, glove_dim)
        glove = nn.Parameter(torch.load(glove_path))
        self.glove.weight = glove
        self.glove.requires_grad = True

        if add_rand_embed:
            self.embed = nn.Embedding(num_embeddings, rand_dim)
            self.final_word_emb = glove_dim + rand_dim

    def get_word_embed_size(self, ):
        return self.final_word_emb

    def forward(self, x):
        '''
            x: (batch, nb_words, nb_characters [tokens])
        '''
        emb = self.glove(x)
        if self.add_rand_embed:
            emb2 = self.embed(x)
            emb = torch.cat([emb, emb2], dim=2)
        return emb


class EncoderText_glove(nn.Module):
    def __init__(self, opt, type='infty'):
        super(EncoderText_glove, self).__init__()
        self.embed_size = opt.embed_size
        self.no_txtnorm = opt.no_txtnorm
        self.type = type
        self.gpool = GPO(32, 32)
        # word embedding
        self.embed = GloveEmb(
            opt.vocab_size,
            glove_dim=opt.word_dim,
            # https://github.com/CrossmodalGroup/NAAF/tree/main/vocab
            glove_path=f'/home/qinyang/projects/TCL/vocab/glove_840B_{opt.data_name}.json.pkl',
            add_rand_embed=False,
            rand_dim=opt.word_dim,
        )
        self.rnn = nn.GRU(self.embed.final_word_emb, opt.embed_size, opt.num_layers, batch_first=True,
                          bidirectional=opt.use_bi_gru)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)
        self.rnn.flatten_parameters()
        packed = pack_padded_sequence(x_emb, lengths.cpu(), batch_first=True)
        # Forward propagate RNN
        out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2
        # pooled_features, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))
        if self.type == 'infty':
            pooled_features, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))
        else:
            pooled_features = maxk_pool1d_var(cap_emb, 1, cap_emb.size(1), cap_len) # avg-pool

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)
        return pooled_features

class VSEModel(nn.Module):
    """
        The standard VSE model
    """
    def __init__(self, opt, word2idx=None):
        # Build Models
        super(VSEModel, self).__init__()

        self.img_enc = EncoderImageAggr(opt)
        if opt.method == 'VSE_infty':
            self.txt_enc = EncoderText_GRU(opt)
        elif opt.method == 'VSE_infty_glove':
            self.txt_enc = EncoderText_glove(opt)
        elif opt.method == 'VSE_infty_bert' :
            self.txt_enc = EncoderText_BERT(opt)
        elif opt.method == 'VSE_glove':
            self.img_enc = EncoderImageAggr(opt,type='VSE') 
            self.txt_enc = EncoderText_glove(opt,type='VSE')
        elif opt.method == 'VSE':
            self.img_enc = EncoderImageAggr(opt,type='VSE')
            self.txt_enc = EncoderText_GRU(opt,type='VSE')
        elif opt.method == 'VSE_bert':
            self.img_enc = EncoderImageAggr(opt,type='VSE')
            self.txt_enc = EncoderText_BERT(opt,type='VSE')
        else:
            raise ValueError('Invalid method of {}'.format(opt.method))
        self.method = opt.method
        self.tau = opt.tau
        self.active = opt.active
        self.opt=opt

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, image_lengths=None):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            image_lengths = image_lengths.cuda()
            lengths = torch.Tensor(lengths).cuda()

        img_emb = self.img_enc(images, image_lengths)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb

    def forward_sim(self, img_embs, cap_embs, cap_lens, mode='sim'):
        sims = img_embs@cap_embs.t()
        if mode == 'sim':
           return sims
        else:
            if self.active=='Exp':
                evidences = (sims/self.tau).exp()
            elif self.active=='ReLU':
                evidences = (sims/self.tau).relu()
            else:
                raise ValueError('Invalid active of {}'.format(self.active))
            return [sims, evidences]

  