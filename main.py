import argparse
import torch
from model import Biaffine, Dependency_Parsing


if __name__ == "__main__":
    config = argparse.ArgumentParser(description='Configuration for Biaffine Dependency Parsing')
    # Train set arguments
    config.add_argument('--train_dir', action='store', default='', type=str)
    config.add_argument('--train_use_domain', action='store', default=False, type=bool)
    config.add_argument('--train_use_folder', action='store', default=False, type=bool)

    # Dev set arguments
    config.add_argument('--dev_dir', action='store', default='', type=str)
    config.add_argument('--dev_use_domain', action='store', default=False, type=bool)
    config.add_argument('--dev_use_folder', action='store', default=False, type=bool)

    # Test set arguments
    config.add_argument('--test_dir', action='store', default='', type=str)
    config.add_argument('--test_use_domain', action='store', default=False, type=bool)
    config.add_argument('--test_use_folder', action='store', default=False, type=bool)

    # Embedding arguments
    config.add_argument('--embedding_type', action='store', default='roberta', type=str)
    config.add_argument('--embedding_name', action='store', default='', type=str)

    # Model parameters
    config.add_argument('--arc_mlp', action='store', default=500, type=int)
    config.add_argument('--label_mlp', action='store', default=100, type=int)
    config.add_argument('--drop_out', action='store', default=0.33, type=float)
    config.add_argument('--batch_size', action='store', default=32, type=int)
    config.add_argument('--learning_rate', action='store', default=5e-5, type=float)
    config.add_argument('--lr_rate', action='store', default=10, type=float)
    config.add_argument('--n_epochs', action='store', default=5, type=int)

    # Device arguments
    config.add_argument('--device', action='store', default='cpu', type=str)

    args = config.parse_args()
    torch.set_default_device(args.device)

    # Call dependency models
    parser = Dependency_Parsing(args)
    parser.Train(n_epochs=args.n_epochs)