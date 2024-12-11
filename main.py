import argparse
import torch
from model import Dependency_Parsing
from dataset import Dataset
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

    # Evaluate arguments
    config.add_argument('--eval_dir', action='store', default='', type=str)
    config.add_argument('--eval_use_domain', action='store_true')
    config.add_argument('--eval_use_folder', action='store_true')
    config.add_argument('--eval_require_preprocessing', action='store_false')

    # Predict arguments
    config.add_argument('--predict_dir', action='store', default='', type=str)
    config.add_argument('--predict_require_preprocessing', action='store_false')

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
    config.add_argument('--mode', choices=['train', 'evaluate', 'predict'], required=True, help='Model mode')
    config.add_argument('--model_name', action='store', required=True, type=str)
    config.add_argument('--save_dir', action='store', default='./', type=str)

    args = config.parse_args()

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger = logging.getLogger('')
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(args.model_name + '.log'), logging.StreamHandler()])
    args_list = [f'{i}: {j}' for i, j in vars(args).items()]
    logger.info('Model arguments:')
    for arg in args_list:
        logger.info(arg)
    logger.info('---------------------------------------------------------------------------------------------------')

    # Call the biaffine model
    torch.set_default_device(args.device)
    parser = Dependency_Parsing(args)
    logger.info(parser)
    logger.info('---------------------------------------------------------------------------------------------------')

    if args.mode == 'train':
        parser.Train(n_epochs=args.n_epochs, logger=logger)

    if args.mode == 'evaluate':
        eval_dataset = Dataset(directory=args.eval_dir, use_folder=args.eval_use_folder,
                               use_domain=args.eval_use_domain)
        parser.Eval(dataset=eval_dataset, require_preprocessing=args.eval_require_preprocessing, logger=logger)

    if args.mode == 'predict':
        predict_dataset = Dataset(directory=args.predict_dir)
        predicted_heads, predicted_labels = parser.Eval(dataset=predict_dataset, require_preprocessing=args.predict_require_preprocessing,
                                                        logger=logger, mode=args.mode)
        inv_label_list = {parser.label_vocab[key]: key for key in parser.label_vocab.keys()}
        idx = 0
        data = open(args.predict_dir, 'r', encoding='utf-8').readlines()
        predicted_data = []
        for line in data:
            split_line = line.split('\t')
            if len(split_line) != 10:
                predicted_data.append(line)
            else:
                split_line[6] = str(int(predicted_heads[idx]))
                split_line[7] = inv_label_list[int(predicted_labels[idx])]
                idx += 1
                predicted_data.append('\t'.join(split_line))
        out_dir = args.save_dir + '/' + args.predict_dir.strip('/').strip('\\').split('/')[-1].split('\\')[-1] + '.predict.conllu'
        out_file = open(out_dir, 'w', encoding='utf-8')
        for line in predicted_data:
            out_file.write(line)
        out_file.close()
        logger.info(f'Predicted file is saved to {out_dir}.')