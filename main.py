import os
import configparser
import argparse
import datetime


import torch

from utils.exp import Experiment
from utils.tools import fix_random_seed, set_logger, logger, STATISTICAL_MODEL_LIST


def parse_args():
    parser = argparse.ArgumentParser(description="Suo Yang Time Series Imputation.")
    parser.add_argument('-dataset_dir', type=str, default='dataset/humidity', help='')
    parser.add_argument('-dataset_file', type=str, default='humidity_20per_missing.csv',
                        help='')
    parser.add_argument('-checkpoints_dir', type=str, default='checkpoints', help='')
    parser.add_argument('-log_dir', type=str, default='log', help='')

    parser.add_argument('-train', type=bool, default=True, help='')
    parser.add_argument('-model', type=str, default='BRITS', help='')
    parser.add_argument('-target', type=list[str], default=['humidity_missing'], help='')
    parser.add_argument('-seq_len', type=int, default=48, help='')
    parser.add_argument('-n_samples', type=int, default=100, help='Generative model only')
    parser.add_argument('-missing_type', type=str, default='cr', help='missing type: [cr, cbr, br, mix]')
    parser.add_argument('-missing_rate', type=float, default=0.1, help='')
    parser.add_argument('-train_ratio', type=float, default=0.7, help='')
    parser.add_argument('-vali_ratio', type=float, default=0.1, help='')
    parser.add_argument('-random_seed', type=int, default=202221543, help='')
    parser.add_argument('-gpu', type=bool, default=True, help='')
    parser.add_argument('-multi_gpu', type=bool, default=False, help='')
    parser.add_argument('-gpu_id', type=str, default="0", 
                        help='gpu_id should be 0,1,2,3... when use multiple gpu')
    parser.add_argument('-use_amp', type=bool, default=False, help='')

    parser.add_argument('-batch_size', type=int, default=32, help='')
    parser.add_argument('-lr', type=float, default=1e-3, help='')
    parser.add_argument('-epochs', type=int, default=200, help='')
    parser.add_argument('-patience', type=int, default=10, help='')
    parser.add_argument('-num_workers', type=int, default=0, help='')

    # SAITS
    parser.add_argument('-diagonal_attention_mask', type=bool, default=True, help='')

    # SAITS Transformer
    parser.add_argument('-n_layers', type=int, default=2, help='')
    parser.add_argument('-d_model', type=int, default=256, help='')
    parser.add_argument('-d_inner', type=int, default=128, help='')
    parser.add_argument('-n_heads', type=int, default=4, help='')
    parser.add_argument('-d_k', type=int, default=64, help='')
    parser.add_argument('-d_v', type=int, default=64, help='')
    parser.add_argument('-d_ffn', type=int, default=128, help='')
    parser.add_argument('-dropout', type=float, default=0.1, help='')
    parser.add_argument('-attn_dropout', type=float, default=0.1, help='')
    parser.add_argument('-ORT_weight', type=float, default=1, help='')
    parser.add_argument('-MIT_weight', type=float, default=1, help='')

    # USGAN
    parser.add_argument('-USGAN_lambda_mse', type=int, default=1, help='')
    parser.add_argument('-USGAN_hint_rate', type=float, default=0.7, help='')
    parser.add_argument('-USGAN_G_steps', type=int, default=1, help=''),
    parser.add_argument('-USGAN_D_steps', type=int, default=1, help='')
    parser.add_argument('-USGAN_rnn_hidden_size', type=int, default=512, help='')
    parser.add_argument('-USGAN_dropout', type=float, default=0.0, help='')

    # BRITS
    parser.add_argument('-rnn_hidden_size', type=int, default=512, help='')

    # LOCF
    parser.add_argument('-LOCF_first_step_imputation', type=str, default='backward', help='')

    # CSDI
    parser.add_argument('-CSDI_timeemb', type=int, default=128, help='')
    parser.add_argument('-CSDI_featureemb', type=int, default=16, help='')
    parser.add_argument('-CSDI_is_unconditional', type=bool, default=False, help='')
    parser.add_argument('-CSDI_target_strategy', type=str, default='random', help='')
    parser.add_argument('-CSDI_channels', type=int, default=64, help='')
    parser.add_argument('-CSDI_num_steps', type=int, default=50, help='')
    parser.add_argument('-CSDI_diffusion_embedding_dim', type=int, default=128, help='')
    parser.add_argument('-CSDI_nheads', type=int, default=8, help='')
    parser.add_argument('-CSDI_is_linear', type=bool, default=False, help='')
    parser.add_argument('-CSDI_layers', type=int, default=4, help='')
    parser.add_argument('-CSDI_schedule', type=str, default='quad', help='[quad, linear]')
    parser.add_argument('-CSDI_beta_start', type=float, default=0.0001, help='')
    parser.add_argument('-CSDI_beta_end', type=float, default=0.5, help='')

    args = parser.parse_args()
    args.features = len(args.target)
    args.gpu = args.gpu and torch.cuda.is_available()
    time_now = datetime.datetime.now().__format__("%Y%m%d_T%H%M%S")
    args.log_path = os.path.join(args.log_dir, args.model, time_now)
    args.checkpoints_path = os.path.join(args.checkpoints_dir, args.model, time_now)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.checkpoints_path, exist_ok=True)
    # save all configuration.
    config = configparser.ConfigParser(vars(args))
    with open(os.path.join(args.log_path, 'config.ini'), 'w') as f:
        config.write(f)
    return args

def main() :
    args = parse_args()
    fix_random_seed(args.random_seed)
    set_logger(args.log_path)
    exp = Experiment(args)
    logger.info(f'model params:{exp.params()}')

    if args.train:
        # statistical model do not need trian parameter.
        if args.model in STATISTICAL_MODEL_LIST:
            exp.impute()
        else:
            exp.train()
    else:
        args.checkpoints_path = 'checkpoints/CSDI/20241104_T101036/checkpoint.pth'
        exp.load_model(args.checkpoints_path)
        exp.impute()

if __name__ == '__main__' :
    main()