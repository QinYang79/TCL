import argparse
import random

def get_argument_parser():
    run = 'f30k_u_gru_detach' #已跑
    parser = argparse.ArgumentParser() 
    
    parser.add_argument('--active', default="Exp",
                        help='ReLU|Exp')
    parser.add_argument('--method', default='uniform',
                        help='VSE|VSE_bert|VSE_infty|VSE_infty_glove|VSE_infty_bert|SAF|SGR|NAAF|CLIP')

    parser.add_argument('--tau', default=0.05, type=float,
                        help='the temperature of evidence extractor')
    parser.add_argument('--dis_k', default=3, type=int,
                        help='Default is 1')
    parser.add_argument('--gpu', default='0',
                        help='Which gpu to use.')

    parser.add_argument('--dual', action='store_false',
                        help='If dual is True, train our TCL, else train model with Max of Hinge loss. Default is true')
    
    parser.add_argument('--img_enhance', action='store_false',
                        help='Default is True')
    parser.add_argument('--caption_enhance', action='store_false',
                        help='Default is True')
    parser.add_argument('--use_bi_gru', action='store_false',
                        help='Default is True')
    parser.add_argument('--logger_path', default=f'./runsx/{run}/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_path', default=f'./runsx/{run}/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')

    parser.add_argument('--data_path', default='/home_bak/hupeng/data/data',
                        help='path to datasets')
    parser.add_argument('--vocab_path', default='/home_bak/hupeng/data/vocab',
                        help='Path to saved vocabulary json files.')

    parser.add_argument('--num_epochs', default=20, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=10, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0005, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to logger.info and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--resume', default='',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument(
        "--seed", default=random.randint(0, 1024), type=int, help="Random seed."
    )

    return parser
