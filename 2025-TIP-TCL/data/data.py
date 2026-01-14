"""COCO dataset loader"""
import torch
import torch.utils.data as data
import os
import numpy as np
import random
import nltk
import h5py
import logging

logger = logging.getLogger(__name__)


class PrecompDataset_gru(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, caption_enhance=True, img_enhance=True, method='VSE'):
        self.vocab = vocab
        self.caption_enhance = caption_enhance
        self.img_enhance = img_enhance
        self.data_split = data_split
        loc = data_path + '/'
        self.ag = 0.2
        self.method = method
        # Captions
        self.captions = []
        with open(loc + '%s_caps.txt' % data_split, 'r', encoding="utf-8") as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)
            
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def process_caption(self, caption):
        enhance = self.caption_enhance if self.data_split == 'train' else False
        if not enhance:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            caption = list()
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            target = torch.Tensor(caption)
            return target
        else:
            # Convert caption (string) to word ids.
            tokens = ['<start>', ]
            tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
            tokens.append('<end>')
            deleted_idx = []
            for i, token in enumerate(tokens):
                prob = random.random()
                if prob < self.ag:
                    prob /= self.ag
                    # 50% randomly change token to mask token
                    if prob < 0.5:
                        try:
                            tokens[i] = self.vocab.word2idx['<mask>']
                        except:
                            tokens[i] = self.vocab.word2idx['<unk>']
                    # 10% randomly change token to random token
                    elif prob < 0.6:
                        tokens[i] = random.randrange(len(self.vocab))
                    # 40% randomly remove the token
                    else:
                        tokens[i] = self.vocab(token)
                        deleted_idx.append(i)
                else:
                    tokens[i] = self.vocab(token)
            if len(deleted_idx) != 0:
                tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]
            target = torch.Tensor(tokens)
            return target

    def process_image(self, image):
        enhance = self.img_enhance if self.data_split == 'train' else False
        if enhance:  # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            tmp = image[np.where(rand_list > 0.20)]
            while tmp.size(1) <= 1:
                rand_list = np.random.rand(num_features)
                tmp = image[np.where(rand_list > 0.20)]
            return tmp
        else:
            return image

    def process_image_1(self, image):
        enhance = self.img_enhance if self.data_split == 'train' else False
        if enhance:  # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image[np.where(rand_list < 0.20)] = 1e-8
            return image
        else:
            return image


    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        # Convert caption (string) to word ids.
        target = self.process_caption(caption)
        if 'VSE' in self.method: 
            image = self.process_image(image)
        else:
            image = self.process_image_1(image)
        return image, target, index, img_id

    def __len__(self):
        return self.length


class PrecompDataset_bert(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, tokenizer, caption_enhance=True, img_enhance=True,method='VSE'):
        self.tokenizer = tokenizer
        self.caption_enhance = caption_enhance
        self.img_enhance = img_enhance
        self.data_split = data_split
        self.method=method
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc + '%s_caps.txt' % data_split, 'r', encoding="utf-8") as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        # self.images = np.load(loc + '%s_ims.npy' % data_split)
        
        img_path = loc+'%s_ims.h5py' % data_split
        if os.path.exists(img_path):
            self.images = h5py.File(img_path, 'r')['img']
        else:
            self.images = np.load(loc+'%s_ims.npy' % data_split)
            f = h5py.File(img_path, 'w')
            f.create_dataset('img', data=self.images)
            f.close()
            
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def process_caption(self, caption):
        enhance = self.caption_enhance if self.data_split == 'train' else False
        output_tokens = []
        deleted_idx = []
        tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
        for i, token in enumerate(tokens):
            sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()
            if prob < 0.20 and enhance:
                prob /= 0.20
                # 50% randomly change token to mask token
                if prob < 0.5:
                    for sub_token in sub_tokens:
                        output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif prob < 0.6:
                    for sub_token in sub_tokens:
                        output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                        # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)
                        deleted_idx.append(len(output_tokens) - 1)
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)

        if len(deleted_idx) != 0:
            output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]
        output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
        target = self.tokenizer.convert_tokens_to_ids(output_tokens)
        target = torch.Tensor(target)
        return target

    def process_image(self, image):
        enhance = self.img_enhance if self.data_split == 'train' else False
        if enhance:  # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            tmp = image[np.where(rand_list > 0.20)]
            while tmp.size(1) <= 1:
                rand_list = np.random.rand(num_features)
                tmp = image[np.where(rand_list > 0.20)]
            return tmp
        else:
            return image

    def process_image_1(self, image):
        enhance = self.img_enhance if self.data_split == 'train' else False
        if enhance:  # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image[np.where(rand_list < 0.20)] = 1e-8
            return image
        else:
            return image

    def __getitem__(self, index):
        img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        target = self.process_caption(caption)
        if 'VSE' in self.method: 
            image = self.process_image(image)
        else:
            image = self.process_image_1(image)
        return image, target, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    img_lengths = [len(image) for image in images]
    all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
    for i, image in enumerate(images):
        end = img_lengths[i]
        all_images[i, :end] = image[:end]
    img_lengths = torch.Tensor(img_lengths).long()
    # Merget captions
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    lengths = torch.Tensor(lengths).long()
    return all_images, img_lengths, targets, lengths, ids


def get_loader(data_path, data_split, vocab_or_tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        drop_last = True
    else:
        drop_last = False
    if 'bert' in opt.method:
        data_set = PrecompDataset_bert(data_path, data_split, vocab_or_tokenizer,
                                       opt.caption_enhance, opt.img_enhance,opt.method)
    else:
        data_set = PrecompDataset_gru(data_path, data_split, vocab_or_tokenizer,
                                      opt.caption_enhance, opt.img_enhance,opt.method)
    data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers,
                                              drop_last=drop_last)

    return data_loader


def get_loaders(data_name, vocab_or_tokenizer, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_loader(dpath, 'train', vocab_or_tokenizer, opt,
                              batch_size, True, workers, train=True)
    val_loader = get_loader(dpath, 'dev', vocab_or_tokenizer, opt,
                            batch_size, False, workers, train=False)
    test_loader = get_loader(dpath, 'test', vocab_or_tokenizer, opt,
                             batch_size, False, workers, train=False)
    return train_loader, val_loader, test_loader


def get_test_loader(split_name, data_name, vocab_or_tokenizer, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    # get the test_loader
    test_loader = get_loader(dpath, split_name, vocab_or_tokenizer, opt,
                             batch_size, False, workers, train=False)
    return test_loader
