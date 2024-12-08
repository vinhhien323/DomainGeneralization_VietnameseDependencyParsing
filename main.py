import argparse
import torch
from model import Dependency_Parsing
import logging
from datetime import datetime

if __name__ == "__main__":
    config = argparse.ArgumentParser(description='Configuration for Biaffine Dependency Parsing')
    # Train set arguments
    config.add_argument('--train_dir', action='store', default='', type=str)
    config.add_argument('--train_use_domain', action='store_true')
    config.add_argument('--train_use_folder', action='store_true')

    # Dev set arguments
    config.add_argument('--dev_dir', action='store', default='', type=str)
    config.add_argument('--dev_use_domain', action='store_true')
    config.add_argument('--dev_use_folder', action='store_true')

    # Test set arguments
    config.add_argument('--test_dir', action='store', default='', type=str)
    config.add_argument('--test_use_domain', action='store_true')
    config.add_argument('--test_use_folder', action='store_true')

    # Embedding arguments
    config.add_argument('--embedding_type', action='store', default='roberta', type=str)
    config.add_argument('--embedding_name', action='store', default='', type=str)
    config.add_argument('--embedding_max_len', action='store', default=256, type=int)

    # Model parameters
    config.add_argument('--arc_mlp', action='store', default=500, type=int)
    config.add_argument('--label_mlp', action='store', default=100, type=int)
    config.add_argument('--drop_out', action='store', default=0.33, type=float)
    config.add_argument('--batch_size', action='store', default=32, type=int)
    config.add_argument('--learning_rate', action='store', default=5e-5, type=float)
    config.add_argument('--lr_rate', action='store', default=10, type=float)
    config.add_argument('--n_epochs', action='store', default=5, type=int)

    # GRL parameters
    config.add_argument('--use_grl', action='store_true')
    config.add_argument('--grl_theta', action='store', default=1e-5, type=float)
    config.add_argument('--grl_loss_rate', action='store', default=0.001, type=float)
    config.add_argument('--eval_with_grl', action='store_true')

    # Device arguments
    config.add_argument('--device', action='store', default='cpu', type=str)

    # Train arguments
    config.add_argument('--mode', choices=['train', 'evaluate', 'test'], required=True, help= 'Model mode')
    config.add_argument('--model_name', action='store', default='', type=str)

    args = config.parse_args()

    log_file = args.model_name + str(datetime.now()).replace(' ', '-') + '.log'
    logging.basicConfig(filename=log_file, level=logging.INFO)
    args_list = [f'{i}: {j}' for i, j in vars(args).items()]
    print('Model arguments:')
    for arg in args_list:
        print(arg)
    torch.set_default_device(args.device)

    # Call dependency models

    parser = Dependency_Parsing(args)
    parser.Train(n_epochs=args.n_epochs)
