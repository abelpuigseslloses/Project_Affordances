from utils import SunDataset, get_model, setup_paths, make_dataloaders, apply_transforms, setup_device, aggregate_and_log_results
from train_eval import train_model
from pathlib import Path
from torch import nn
import torch.optim as optim
import argparse
import os
import yaml
import wandb
from sklearn.model_selection import KFold
from torch.utils.data import Subset


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Getting hyperparameters")
    parser.add_argument('--sweep_id', default=None, type=str, help="Sweep ID to join an existing sweep")
    parser.add_argument('--start_sweep', action='store_true', help="Flag to indicate if a new sweep should be started")
    parser.add_argument('--model_name', default='resnet_places365', type=str, help="Model name")
    parser.add_argument('--pretrained', action='store_true', help="Use pretrained model")
    parser.add_argument("--type_attributes", default="affordances", type=str, help="Type of attributes to train on")
    parser.add_argument("--early_stopping", action="store_true", help="Use early stopping")
    parser.add_argument('--filter_attributes', action='store_false', help="Filter infrequent attributes")
    parser.add_argument("--frozen_layers", action="store_true", help="Unfreeze all layers")
    parser.add_argument("--num_conv_unfrozen", default=0, type=int, help="Number of conv blocks to unfreeze")
    parser.add_argument("--optimizer", default='sgd', type=str, help="Optimizer")
    parser.add_argument("--num_epochs", default=10000, type=int, help="Number of epochs (large number because of early stopping)")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument('--host', default='local', type=str, help="'local' or 'server'")
    parser.add_argument("--n_splits", default=5, type=int, help="Number of folds for cross-validation")
    parser.add_argument("--min_delta", default=0.001, type=float, help="Min delta for early stopping")
    parser.add_argument("--patience_threshold", default=10, type=int, help="Patience for early stopping")
    parser.add_argument("--cross_validation", action="store_false", help="Use k-fold cross validation")

    return parser.parse_args()


def cross_validate_model(dataset, args, device, train_model_path, class_names, num_attributes, model_full_name):
    """Perform K-Fold Cross Validation on the given dataset and model configuration."""
    all_folds_results = []
    all_folds_precision = []
    all_folds_best_metrics = {}
    all_folds_best_precision = {}

    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(dataset)))):
        print(f'Fold {fold + 1}/{args.n_splits}')
        print('-' * 10)

        train_dataset = Subset(dataset, train_ids)
        test_dataset = Subset(dataset, val_ids)

        apply_transforms(train_dataset.dataset)
        apply_transforms(test_dataset.dataset)

        train_loader, test_loader = make_dataloaders(train_dataset, test_dataset, batch_size=args.batch_size, num_workers=2)

        model = get_model(model_name=args.model_name, num_attributes=num_attributes, frozen_layers=args.frozen_layers, num_conv_unfrozen=args.num_conv_unfrozen, pretrained=args.pretrained, device=device)
        model.to(device)

        loss_fn = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-4) if args.optimizer == 'sgd' else optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        fold_results, fold_precision, best_epoch_metrics, best_epoch_precision = train_model(model=model, train_loader=train_loader, test_loader=test_loader, loss_fn=loss_fn, optimizer=optimizer, num_epochs=args.num_epochs, model_full_name=model_full_name, num_attributes=num_attributes, min_delta=args.min_delta, patience_threshold=args.patience_threshold, type_attributes=args.type_attributes, early_stopping=args.early_stopping, device=device, fold=fold)

        all_folds_results.append(fold_results)
        all_folds_precision.append(fold_precision)
        all_folds_best_metrics[f"Fold {fold}"] = best_epoch_metrics
        all_folds_best_precision[f"Fold {fold}"] = best_epoch_precision

    aggregate_and_log_results(all_folds_results, all_folds_best_metrics, all_folds_precision, args.num_epochs, class_names, train_model_path)


def training_logic(args):
    """Main training logic for model training and cross-validation."""
    run = wandb.init(project="Prova", entity="abelpuigseslloses", config=args)
    run_name = wandb.run.name

    device = setup_device()
    im_label_path, im_path, label_path, attribute_names_path = setup_paths(args)

    dataset = SunDataset(im_label_path, im_path, label_path, attribute_names_path, args.type_attributes, filter_attributes=args.filter_attributes)
    class_names = dataset.attribute_names
    num_attributes = len(class_names)

    formatted_learning_rate = "{:.3f}".format(args.learning_rate).replace('.', '_')
    model_full_name = f'{run_name}_ft_pretrained_{args.model_name}_{args.type_attributes}_{formatted_learning_rate}_{args.optimizer}'

    train_path = Path('results')
    train_path.mkdir(parents=True, exist_ok=True)
    train_model_path = Path(os.path.join(train_path, model_full_name))
    train_model_path.mkdir(parents=True, exist_ok=True)

    if args.cross_validation:
        cross_validate_model(dataset, args, device, train_model_path, class_names, num_attributes, model_full_name)
    else:
        # single split code
        pass


def main():
    wandb.login()
    args = parse_arguments()

    if args.sweep_id:
        wandb.agent(args.sweep_id, function=lambda: training_logic(args))
    else:
        if args.start_sweep:
            with open('sweep_config.yaml', 'r') as file:
                sweep_config = yaml.safe_load(file)
            sweep_id = wandb.sweep(sweep_config, project="Prova", entity="abelpuigseslloses")
            wandb.agent(sweep_id, function=lambda: training_logic(args))
        else:
            training_logic(args)


if __name__ == "__main__":
    main()