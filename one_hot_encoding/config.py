import argparse

def get_config():
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--data_dir', default='../data', help='path to dataset')
    parser.add_argument('--model_dir', default='./checkpoints', help='path to model to save model!')
    parser.add_argument('--pred_dir', default='./pred', help='path to model to save prediction!')
    parser.add_argument('--label_mapping_file', default="./label_mapping.json", help="Path to label mapping file (.json)")
    parser.add_argument('--writer_path', default='./runs', help='Path to save record curve')
    parser.add_argument('--writer_overwrite', action="store_true", help='Whether to overwrite the writer folder')
    parser.add_argument('--resume', default=None, help='path to latest checkpoint (default: None)')
    parser.add_argument('--dataset_type', default="courses", help=["subgroups", "courses"])
    parser.add_argument('--feature_name', default=[], nargs='+')

    # Model parameters
    # parser.add_argument('--model_name', default="hfl/chinese-roberta-wwm-ext", help="Which model you want to use")
    # parser.add_argument('--max_len', default=12, help="Maximum sequence length")
    # parser.add_argument('--fix_encoder', action="store_true", help="Whether to fix encoder during training (default: False)")

    # Leanring rate scheduler
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--warmup_lr", type=float, default=1e-9, help="Warmup start learning rate")

    # Training hyperparameters
    parser.add_argument("--n_epochs", type=int, default=23, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument('--n_cpus', type=int, default=0, help='number of cpu workers (default: 4)')
    parser.add_argument('--seed', type=int, default=2022, help='Fix random seed for reproduce (default: 2022)')

    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-6, help="learning rate (default: 1e-5")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay parameter (default: 1e-4)")
    parser.add_argument("--unbalanced_weight", action="store_true", help="Binary Cross Entropy unbalaned weight")

    # Inference parameters
    parser.add_argument("--thres", type=float, default=0.0, help="threshold for obtaining prediction")
    parser.add_argument("--k", type=int, default=50, help="MAP@k")

    args = parser.parse_args()

    return args