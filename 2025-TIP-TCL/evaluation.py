import logging
import sys
import time
import torch
from collections import OrderedDict
from utils import cosine_similarity_matrix
import numpy as np
from torch.autograd import Variable

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)

def encode_data_VSE(model, data_loader):
    """Encode all images and captions loadable by `data_loader`
    """
    model.val_start()
    img_embs = None
    cap_embs = None

    for i, data_i in enumerate(data_loader):
        if 'CLIP' == model.opt.method:
            images, captions, img_ids, ids = data_i
            with torch.no_grad():
                img_emb, cap_emb = model.forward_emb(images, captions)
        else:    
            images, img_lengths, captions, cap_lengths, ids = data_i
            with torch.no_grad():
                img_emb, cap_emb = model.forward_emb(images, captions, cap_lengths, img_lengths)

        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
        # cache embeddings
        img_embs[ids, :] = img_emb.data.cpu()
        cap_embs[ids, :] = cap_emb.data.cpu()

        del images, captions

    return img_embs, cap_embs


def encode_data_SGRAF(model, data_loader):
    """Encode all images and captions loadable by `data_loader`
    """
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    max_n_word = 0
    # mr = data_loader.dataset.mr
    # data_loader.dataset.mr = 0
    for i, (images, _, captions, lengths, ids) in enumerate(data_loader):
        max_n_word =  max(max_n_word, int(max(lengths).item())) 

    # data_loader.dataset.mr = mr      
    ids_ = []
    for i, (images, _, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        ids_ += ids
        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb = model.forward_emb(images, captions, lengths)
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids, :, :] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :int(max(lengths).item()), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = lengths[j]

        del images, captions

    return img_embs, cap_embs, torch.stack(cap_lens).long()

def shard_attn_scores_SGRAF(model, img_embs, cap_embs, cap_lens, opt, shard_size=100, mode="sim"):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                if mode == "sim":
                    sim = model.forward_sim(im, ca, l, mode)
                else:
                    _, _, sim = model.forward_sim(im, ca, l, mode)  # Calculate evidence for retrieval

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims

def shard_xattn_BCAN(model, images, img_means, captions, caplens, cap_means, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).float().cuda()
            im_m = Variable(torch.from_numpy(img_means[im_start:im_end]), volatile=True).float().cuda()
            s_m = Variable(torch.from_numpy(cap_means[cap_start:cap_end]), volatile=True).float().cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).float().cuda()
            l = caplens[cap_start:cap_end]
            with torch.no_grad():
                sim = model.forward_sim([im, im_m], [s, s_m], l)
                # sim = model.xattn_score_t2i2(im, s, l)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def t2i(npts, sims, per_captions=1, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(per_captions * npts)
    top1 = np.zeros(per_captions * npts)
    top5 = np.zeros((per_captions * npts, 5), dtype=int)

    # --> (per_captions * N(caption), N(image))
    sims = sims.T
    retreivaled_index = []
    for index in range(npts):
        for i in range(per_captions):
            inds = np.argsort(sims[per_captions * index + i])[::-1]
            retreivaled_index.append(inds)
            ranks[per_captions * index + i] = np.where(inds == index)[0][0]
            top1[per_captions * index + i] = inds[0]
            top5[per_captions * index + i] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


def i2t(npts, sims, per_captions=1, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5), dtype=int)
    retreivaled_index = []
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        retreivaled_index.append(inds)
        # Score
        rank = 1e20
        for i in range(per_captions * index, per_captions * index + per_captions, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        top5[index] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


def validate_test(val_loader, model, split_name, fold=False):
    model.val_start()
    logger.info(f"=====>Split_name: {split_name},  {'' if fold else 'no'} five fold")
    per_captions = 5
    if not fold:
        sims_sum = 0
        npts = 0
        simss = []
        with torch.no_grad():
            for i in range(model.models_num):
                if 'VSE' in model.opt.method:
                    img_embs, cap_embs = encode_data_VSE(model.models[i], val_loader)
                elif model.opt.method in ['SAF', 'SGR','SAF_bert','SGR_bert','SAF_glove','SGR_glove']:  
                    img_embs, cap_embs, cap_lens = encode_data_SGRAF(model.models[i], val_loader)

                # clear duplicate 5*images and keep 1*images
                img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])

                # record computation time of validation
                start = time.time()
                if model.opt.method in ['SAF','SGR','SAF_bert','SGR_bert','SAF_glove','SGR_glove']:
                    sims = shard_attn_scores_SGRAF(model.models[i], img_embs, cap_embs, cap_lens, model.opt, shard_size=1000)
                else:
                    sims = cosine_similarity_matrix(img_embs, cap_embs)
                end = time.time()
                logger.info(f"calculate similarity time: {end - start}")

                # caption retrieval
                (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims, per_captions, return_ranks=False)
                logger.info("Average i2t Recall: %.2f" % ((r1 + r5 + r10) / 3))
                logger.info("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr, meanr))
                # image retrieval
                (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims, per_captions, return_ranks=False)
                logger.info("Average t2i Recall: %.2f" % ((r1i + r5i + r10i) / 3))
                logger.info("Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1i, r5i, r10i, medri, meanr))
                currscore = r1 + r5 + r10 + r1i + r5i + r10i
                logger.info(f"rsum:{currscore}") 
                sims_sum += sims

        if model.models_num == 2:
            # caption retrieval
            (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_sum, per_captions, return_ranks=False)
            logger.info("Average i2t Recall: %.2f" % ((r1 + r5 + r10) / 3))
            logger.info("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr, meanr))
            # image retrieval
            (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_sum, per_captions, return_ranks=False)
            logger.info("Average t2i Recall: %.2f" % ((r1i + r5i + r10i) / 3))
            logger.info("Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1i, r5i, r10i, medri, meanr))
            
            currscore = r1 + r5 + r10 + r1i + r5i + r10i
            logger.info(f"rsum:{currscore}")
        
    else:
        results = []
        with torch.no_grad():
            img_embs_dict, cap_embs_dict, cap_lens_dict = dict(), dict(), dict()
            for i in range(model.models_num):
                if 'VSE' in model.opt.method:
                    img_embs, cap_embs = encode_data_VSE(model.models[i], val_loader)
                elif model.opt.method in ['SAF', 'SGR','SAF_bert','SGR_bert','SAF_glove','SGR_glove']:  
                    img_embs, cap_embs, cap_lens = encode_data_SGRAF(model.models[i], val_loader)
                    cap_lens_dict[i] = cap_lens
                img_embs_dict[i] = img_embs
                cap_embs_dict[i] = cap_embs
        all_r = []
        for i in range(5):
            logger.info(f"==========================>fold :{i}")
            sims_sum = 0
            npts = 0
            r_ = []
            r1, r5, r10, medr, meanr, r1i, r5i, r10i, medri, meanr = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            for j in range(model.models_num):
                img_embs_shard = img_embs_dict[j][i * 5000:(i + 1) * 5000:per_captions]
                cap_embs_shard = cap_embs_dict[j][i * 5000:(i + 1) * 5000]

                if model.opt.method in ['SAF','SGR','SAF_bert','SGR_bert','SAF_glove','SGR_glove']:
                    cap_lens_shard =  cap_lens_dict[j][i * 5000:(i + 1) * 5000]
                    sims = shard_attn_scores_SGRAF(model.models[j], img_embs_shard, cap_embs_shard, cap_lens_shard, model.opt, shard_size=1000)
                else:
                    sims = cosine_similarity_matrix(img_embs_shard, cap_embs_shard)

                # sims = np.exp(cosine_similarity_matrix(img_embs_shard, cap_embs_shard) / model.opt.tau)
                npts = img_embs_shard.shape[0]
                (r1, r5, r10, medr, meanr) = i2t(npts, sims, per_captions, return_ranks=False)
                logger.info("Average i2t Recall: %.2f" % ((r1 + r5 + r10) / 3))
                logger.info("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr, meanr))
                # image retrieval
                (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims, per_captions, return_ranks=False)
                logger.info("Average t2i Recall: %.2f" % ((r1i + r5i + r10i) / 3))
                logger.info("Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1i, r5i, r10i, medri, meanr))
                currscore = r1 + r5 + r10 + r1i + r5i + r10i
                logger.info(f"rsum:{currscore}")

                sims_sum += sims
                r_.append([r1, r5, r10, r1i, r5i, r10i])
            all_r.append(r_)
            if model.models_num == 2:
                sims_ensemble = sims_sum 
                (r1, r5, r10, medr, meanr) = i2t(npts, sims_ensemble, per_captions, return_ranks=False)
                logger.info("Average i2t Recall: %.2f" % ((r1 + r5 + r10) / 3))
                logger.info("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr, meanr))
                # image retrieval
                (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims_ensemble, per_captions, return_ranks=False)
                logger.info("Average t2i Recall: %.2f" % ((r1i + r5i + r10i) / 3))
                logger.info("Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1i, r5i, r10i, medri, meanr))

                rsum = r1 + r5 + r10 + r1i + r5i + r10i
                ar = (r1 + r5 + r10) / 3
                ari = (r1i + r5i + r10i) / 3
                logger.info('Current rsum is {}'.format(rsum))
                results.append([r1, r5, r10, medr, meanr, r1i, r5i, r10i, medri, meanr, ar, ari, rsum])
            else:
                rsum = r1 + r5 + r10 + r1i + r5i + r10i
                ar = (r1 + r5 + r10) / 3
                ari = (r1i + r5i + r10i) / 3
                results.append([r1, r5, r10, medr, meanr, r1i, r5i, r10i, medri, meanr, ar, ari, rsum])

        logger.info(f"---------------- five fold -------------------")
        if model.models_num == 2:
            r_ = np.array(all_r).mean(axis=0)
            r_i2t = r_[0].flatten()
            r_t2i = r_[1].flatten()
            print(tuple(r_.flatten()))
            print(tuple(r_i2t[0:3].flatten()))
            logger.info('The results of i2t model:')
            logger.info("rsum: %.1f" % (np.sum(r_i2t)))
            logger.info("Average i2t Recall: %.1f" % (np.mean(r_i2t[0:3])))
            logger.info("Image to text: %.1f %.1f %.1f" %
                        tuple(r_i2t[0:3].flatten()))
            logger.info("Average t2i Recall: %.1f" % (np.mean(r_i2t[3:6])))
            logger.info("Text to image: %.1f %.1f %.1f" %
                        tuple(r_i2t[3:6].flatten()))
            # ------------
            logger.info('\nThe results of t2i model:')
            logger.info("rsum: %.1f" % (np.sum(r_t2i)))
            logger.info("Average i2t Recall: %.1f" % (np.mean(r_t2i[0:3])))
            logger.info("Image to text: %.1f %.1f %.1f" %
                        tuple(r_t2i[0:3]))
            logger.info("Average t2i Recall: %.1f" % (np.mean(r_t2i[3:6])))
            logger.info("Text to image: %.1f %.1f %.1f" %
                        tuple(r_t2i[3:6]))
        # ------------
        logger.info('Mean similarities of each model:')
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        logger.info("rsum: %.1f" % (mean_metrics[12]))
        logger.info("Average i2t Recall: %.1f" % mean_metrics[10])
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[:5])
        logger.info("Average t2i Recall: %.1f" % mean_metrics[11])
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[5:10])

