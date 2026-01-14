# coding=utf-8
import logging
import os
import random
import time

import numpy as np
import torch
from transformers import BertTokenizer

import opts
import tensorboard_logger as tb_logger
import data.data as data
from utils import save_config, cosine_similarity_matrix
from evaluation import AverageMeter, LogCollector, encode_data_SGRAF, encode_data_VSE, i2t, shard_attn_scores_SGRAF, t2i
from model import TCL
from vocab import deserialize_vocab, deserialize_vocab_glove
import warnings

warnings.filterwarnings("ignore")

def adjust_learning_rate(model, epoch, lr_schedules):
    logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for i in range(model.models_num):
            if 'bert' in model.opt.method:
                for param_group in model.optimizer_dict[i]['train'].param_groups:
                    old_lr = param_group['lr']
                    new_lr = old_lr * 0.1
                    param_group['lr'] = new_lr
                    logger.info('new lr {}'.format(new_lr))
            else:
                for param_group in model.optimizer_dict[i].param_groups:
                    old_lr = param_group['lr']
                    new_lr = old_lr * 0.1
                    param_group['lr'] = new_lr
                    logger.info('new lr {}'.format(new_lr))

def init_logging(log_file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s %(message)s')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='', ckpt=True):
    logger = logging.getLogger(__name__)
    tries = 15

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            if ckpt:
                torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

def train(opt, train_loader, model, epoch, val_loader, best_rsum=0):
    # average meters to record the training statistics
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    num_loader_iter = len(train_loader.dataset) // train_loader.batch_size + 1
    end = time.time()
    logger.info("=======>Epoch: {0}".format(epoch))
    for i, train_data in enumerate(train_loader):
        model.train_start()
        data_time.update(time.time() - end)
        model.logger = train_logger

        # Update the model
        if 'CLIP' in opt.method:
            images, captions, img_ids, ids = train_data
            if len(ids) == 1:
                break
            else:
                model.train_emb(images, captions, epoch=epoch)
        else:    
            images, img_lengths, captions, cap_lengths, ids = train_data
            if len(ids) == 1:
                break
            else:
                model.train_emb(images, captions, img_lengths, cap_lengths, epoch=epoch)

        batch_time.update(time.time() - end)
        end = time.time()
        if model.step % opt.log_step == 0:
            logger.info( 
                'Epoch: [{0}][{1}/{2}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) \t{e_log}'.format(epoch, i, num_loader_iter,
                                                                                  batch_time=batch_time,
                                                                                  data_time=data_time,
                                                                                  e_log=str(model.logger)))
        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.step)
        tb_logger.log_value('step', i, step=model.step)
        tb_logger.log_value('batch_time', batch_time.val, step=model.step)
        tb_logger.log_value('data_time', data_time.val, step=model.step)
        model.logger.tb_log(tb_logger, step=model.step)


def validate(val_loader, model, mode='dev'):
    model.val_start()
    sims_sum = 0
    logger.info(f"=====>Mode: {mode}")
    npts = 0
    per_captions = 5
    with torch.no_grad():
        for i in range(model.models_num):
            if model.opt.method == 'BCAN':
                img_embs, img_means, cap_embs, cap_lens, cap_means = encode_data_BCAN(model.models[i], val_loader)
            elif model.opt.method == 'CLIP' or 'VSE' in model.opt.method:
                img_embs, cap_embs = encode_data_VSE(model.models[i], val_loader)
            elif model.opt.method in ['SAF', 'SGR','SAF_bert','SGR_bert','SAF_glove','SGR_glove']:  
                img_embs, cap_embs, cap_lens = encode_data_SGRAF(model.models[i], val_loader)

            # clear duplicate 5*images and keep 1*images
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])

            # record computation time of validation
            start = time.time()
            if model.opt.method in ['SAF','SGR','SAF_bert','SGR_bert','SAF_glove','SGR_glove']:
                sims = shard_attn_scores_SGRAF(model.models[i], img_embs, cap_embs, cap_lens, opt, shard_size=1000)
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
            # np.save(f"./Tc_SAF_{i}.npy",sims)
            sims_sum += sims

    if model.models_num == 2: 
        # sims_sum = smis1+smis2
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
    
    return currscore

if __name__ == '__main__':
    parser = opts.get_argument_parser()
    opt = parser.parse_args()
    # Set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    
    # Set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.random.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    # Make dir
    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.isdir(opt.logger_path):
        os.makedirs(opt.logger_path)
    # Save config
    save_config(opt, os.path.join(opt.logger_path, "config.json"))
    # logger initialization
    tb_logger.configure(opt.logger_path, flush_secs=5)
    logger = init_logging(opt.logger_path + '/log.txt')
    logger.info(f"===>PID:{os.getpid()}, GPU:[{opt.gpu}]")
    logger.info(opt)
    
    # Load Vocabulary
    if 'bert' in opt.method:
        tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
        train_loader, val_loader, test_loader = data.get_loaders(opt.data_name, tokenizer, opt.batch_size, opt.workers, opt)
        model = TCL(opt) 
    else:
        v_path = os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)
        if 'glove' in opt.method:
            v_path = os.path.join(opt.vocab_path, '%s.json' % opt.data_name)
            vocab = deserialize_vocab_glove(v_path)
        else:
            vocab = deserialize_vocab(v_path)

        # vocab = deserialize_vocab(v_path)
        opt.vocab_size = len(vocab)
        # Get data loaders
        train_loader, val_loader, test_loader = data.get_loaders(opt.data_name, vocab, opt.batch_size, opt.workers, opt)
        model = TCL(opt, vocab.word2idx)

    # Load checkpoint
    start_epoch = 0
    best_rsum = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # step is used to show logs as the continuation of another training
            model.step = checkpoint['step']

            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch-1, best_rsum))
            validate(val_loader, model, 'dev')
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))
    #####
    # Train the Model
    logger.info("Logger path\t" + opt.logger_path)
    logger.info("Save path\t" + opt.model_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    lr_schedules = [opt.lr_update]
    # validate(test_loader, model, 'test')
    for epoch in range(start_epoch, opt.num_epochs):
        adjust_learning_rate(model, epoch, lr_schedules)
        train(opt, train_loader, model, epoch, val_loader, best_rsum)
        # # evaluate on validation set
        rsum = validate(val_loader, model, 'dev')
        validate(test_loader, model, 'test')
        # # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch+1,
            'model': model.state_dict(),
            'step': model.step,
            'best_rsum': best_rsum,
            'opt': opt,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_path + '/',ckpt=False)

    logger.info(f"best_rsum:{best_rsum}")