import logging
import os
import torch
from transformers import BertTokenizer
import data.data_clip as data_clip
import data.data as data
from evaluation import LogCollector, validate_test

from model import TCL
from vocab import deserialize_vocab, deserialize_vocab_glove
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    from datetime import datetime
    start_time = datetime.now()
    print(start_time)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    mode_path = "/home/qinyang/projects/TCL/runs/coco_precomp_SGR_glove_dis_k3_tau0.05_decyl1_activeExp_lr0.0002_1/checkpoint/model_best.pth.tar"
    checkpoint = torch.load(mode_path)
    opt = checkpoint['opt']
    # Uncomment the following two lines and change them to your own path
    # opt.data_path = "/home/qinyang/projects/data/cross_modal_data/data/data"
    # opt.vocab_path = "/home/qinyang/projects/data/cross_modal_data/data/vocab"
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logger.info(opt)
    logger.info(f"Load model: {mode_path}")
    logger.info(f"Best epoch: {checkpoint['epoch']}")
    logger.info(f"Best dev rsum: {checkpoint['best_rsum']}")
    # Load Vocabulary
    if 'bert' in opt.method:
        vocab_or_tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
        train_loader, val_loader, test_loader = data.get_loaders(opt.data_name, vocab_or_tokenizer, opt.batch_size, opt.workers, opt)
        model = TCL(opt)
    else:
        v_path = os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)
        if 'glove' in opt.method:
            v_path = os.path.join(opt.vocab_path, '%s.json' % opt.data_name)
            vocab_or_tokenizer = deserialize_vocab_glove(v_path)
        else:
            vocab_or_tokenizer = deserialize_vocab(v_path)
        opt.vocab_size = len(vocab_or_tokenizer)
    
        model = TCL(opt, vocab_or_tokenizer.word2idx)
    # model.make_data_parallel()
    model.load_state_dict(checkpoint['model'])

    # Get data loader
    if 'coco' in opt.data_name:
        test_loader = data.get_test_loader('testall', opt.data_name, vocab_or_tokenizer, opt.batch_size, opt.workers,
                                           opt)
        validate_test(test_loader, model, 'testall', fold=True)
        validate_test(test_loader, model, 'testall', fold=False)
    else:
        test_loader = data.get_test_loader('test', opt.data_name, vocab_or_tokenizer, 128, opt.workers,
                                           opt)
        validate_test(test_loader, model, 'test', fold=False)


    end_time = datetime.now()
    print('cost_time', int((end_time - start_time).seconds))    

    # from thop import profile
    # train_logger = LogCollector()
    # model.logger=train_logger
    # images, image_lengths, captions, caption_lengths, ids = iter(test_loader).next()
    # p = 0
    # flops, params = profile(model.models[0]['img_enc'], (images.cuda(), image_lengths.cuda()), verbose=False)
    # p += flops
    # flops, params = profile(model.models[0]['txt_enc'], (captions.cuda(), caption_lengths.cuda()), verbose=False)
    # p += flops
    # print("FLOPs=", str(p / 1e9) + '{}'.format("G"))
    # print("FLOPs=", str(p*2 / 1e9) + '{}'.format("G"))
    # 创建输入网络的tensor
    # from thop import profile
    # images, image_lengths, captions, caption_lengths, ids = iter(test_loader).next()
    # flops, params = profile(model.models[0]['img_enc'], (images, image_lengths), verbose=False)
    # 分析parameters
    # from nni.compression.pytorch.utils.counter import count_flops_params
    # flops, params, results = count_flops_params(model.models[0]['img_enc'], (images, image_lengths))
    # flops, params, results = count_flops_params(model.models[0]['txt_enc'], (captions, caption_lengths))
    # flops, params, results = count_flops_params(model, (images, captions, image_lengths, caption_lengths,1))

