import argparse
from recbole.quick_start import run_recbole

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DGCF on a single domain dataset")
    parser.add_argument('--dataset', '-d', type=str, default='AmazonElec_AmazonCloth_commonUser_10-core',
                        help='dataset name (stored in ./dataset/)')
    parser.add_argument('--epochs', '-e', type=int, default=400,
                        help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=4096,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--gpu_id', '-g', type=int, default=0,
                        help='gpu id')
    
    # Model-specific parameters (Base GCN)
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='embedding dimension size (must be divisible by n_factors)')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of GCN layers')
    parser.add_argument('--reg_weight', type=float, default=1e-05,
                        help='L2 regularization weight')

    # DGCF-specific hyperparameters
    parser.add_argument('--n_factors', type=int, default=4,
                        help='number of latent factors/intents (embedding_size must be divisible by this)')
    parser.add_argument('--n_iterations', type=int, default=2,
                        help='number of routing iterations')
    parser.add_argument('--cor_weight', type=float, default=0.01,
                        help='weight of correlation regularization loss')

    args = parser.parse_args()

    # Validate that embedding_size is divisible by n_factors
    if args.embedding_size % args.n_factors != 0:
        raise ValueError(f"embedding_size ({args.embedding_size}) must be divisible by n_factors ({args.n_factors}) for DGCF.")

    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'dataset')

    config_dict = {
        # General settings
        'gpu_id': args.gpu_id,
        'use_gpu': True,
        'state': 'INFO',
        'reproducibility': True,
        'data_path': data_path,
        'checkpoint_dir': os.path.join(current_dir, 'saved'),
        'save_dataset': False,
        'save_dataloaders': False,
        
        # Training settings
        'epochs': args.epochs,
        'train_batch_size': args.batch_size,
        'eval_batch_size': 40960,
        'learning_rate': args.lr,
        'stopping_step': 10,
        
        # Evaluation settings
        'eval_args': {
            'split': {'RS': [0.6, 0.2, 0.2]},
            'group_by': 'user',
            'order': 'RO',
            'mode': 'full'
        },
        'repeatable': True,
        'metrics': ["Recall", "Hit", "MRR", "NDCG"],
        'topk': [20],
        'valid_metric': 'Recall@20',
        
        # Dataset fields configuration
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'threshold': {'rating': 4},
        'load_col': {'inter': ['user_id', 'item_id', 'rating']},
        
        # Base Model hyperparameters
        'embedding_size': args.embedding_size,
        'n_layers': args.n_layers,
        'reg_weight': args.reg_weight,

        # DGCF hyperparameters
        'n_factors': args.n_factors,
        'n_iterations': args.n_iterations,
        'cor_weight': args.cor_weight
    }

    # Run RecBole training for DGCF
    run_recbole(model='DGCF', dataset=args.dataset, config_dict=config_dict)
