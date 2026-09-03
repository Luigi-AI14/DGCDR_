import argparse
from recbole.quick_start import run_recbole

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run NCL on a single domain dataset")
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
                        help='embedding dimension size')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of GCN layers')
    parser.add_argument('--reg_weight', type=float, default=1e-05,
                        help='L2 regularization weight')

    # NCL-specific hyperparameters
    parser.add_argument('--ssl_temp', type=float, default=0.1,
                        help='temperature parameter in contrastive loss')
    parser.add_argument('--ssl_reg', type=float, default=1e-04,
                        help='regularization weight of contrastive loss')
    parser.add_argument('--hyper_layers', type=int, default=1,
                        help='number of hyper-layers for contrastive representation')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='weight of semantic contrastive loss')
    parser.add_argument('--num_clusters', type=int, default=1000,
                        help='number of clusters for prototype contrastive learning')
    parser.add_argument('--proto_reg', type=float, default=1e-07,
                        help='regularization weight of prototype contrastive loss')

    args = parser.parse_args()

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

        # NCL hyperparameters
        'ssl_temp': args.ssl_temp,
        'ssl_reg': args.ssl_reg,
        'hyper_layers': args.hyper_layers,
        'alpha': args.alpha,
        'num_clusters': args.num_clusters,
        'proto_reg': args.proto_reg
    }

    # Run RecBole training for NCL
    run_recbole(model='NCL', dataset=args.dataset, config_dict=config_dict)
