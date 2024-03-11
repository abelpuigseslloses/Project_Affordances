#!/usr/bin/env python3

from utils_filtering import SunDataset, get_model, set_seeds, save_model, save_results, plot_class_precision
import torch
from get_data import make_datasets,apply_transforms, make_dataloaders

from pathlib import Path
from torch import nn
import torch.optim as optim
from train_model_wandb import train_model, test_model
import argparse
import os
import wandb
import numpy as np

## to visualize I can import from torchinfo import summary (needs install on server)

## add a scheduler?

def main():

    wandb.login()

    # Create a parser
    parser = argparse.ArgumentParser(description="Getting hyperparameters")

    # Get an arg for the command
    parser.add_argument('--command',
                        default='eval',
                        type=str,
                        help="'train' or 'eval'")

    # Get an arg for the model name
    parser.add_argument('--model_name',
                        default='resnet_places365',
                        type=str,
                        help="Use 'resnet_places365' or 'resnet_imagenet'")

    # Get an arg for num_attributes
    parser.add_argument("--num_attributes",
                        default=36,
                        type=int,
                        help="the number of attributes to train on")

    # Get an arg for number of frozen layers
    parser.add_argument("--frozen_layers",
                        default=12,
                        type=int,
                        help="the number of layers to freeze (from 0 to 12 for resnet-places365; from 0 to 161 for resnet-alexnet")

    # Get an arg for loading weights of trained model
    parser.add_argument('--model_weights_path',
                        default=None,
                        help="get a path of a pretrained model")

    # Get an arg for optimizer
    parser.add_argument("--optimizer",
                        default='SGD',
                        type=str,
                        help="'SGD' or 'Adam'")

    # Get an arg for the test_size
    parser.add_argument('--test_size',
                        default='0.1',
                        type=float,
                        help="test-data ratio (total size = 1)")

    # Get an arg for num_epochs
    parser.add_argument("--num_epochs",
                        default=7,
                        type=int,
                        help="the number of epochs to train for")

    # Get an arg for batch_size
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="the number of samples per batch")

    # Get an arg for learning_rate
    parser.add_argument("--learning_rate",
                        default=0.1,
                        type=float,
                        help="learning rate to use for model")

    # Get an arg for where it is run
    parser.add_argument('--host',
                        default='server',
                        type=str,
                        help="'local' or 'server'")

    # Get our arguments from the parser
    args = parser.parse_args()

    # W&B: initialize for experiment tracking
    wandb.init(
        project="Affordance",
        config={
            "command": args.command,
            "model_name": args.model_name,
            "num_attributes": args.num_attributes,
            "frozen_layers": args.frozen_layers,
            "model_weights_path": args.model_weights_path,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "test_size": args.test_size,
            "host": args.host
        }
    )

    config = wandb.config

    command = config.command
    model_name = config.model_name
    num_attributes = config.num_attributes
    frozen_layers = config.frozen_layers
    model_weights_path = config.model_weights_path
    optimizer = config.optimizer
    test_size = config.test_size
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    host = config.host

    # Folder to save models
    models_folder_path = 'saved_models'
    if not os.path.exists(models_folder_path):
        os.makedirs(models_folder_path)
        print(f"Directory '{models_folder_path}' created")
    else:
        print(f"Directory '{models_folder_path}' already exists")

    # Check if run is local or server
    if host == 'local':
        base_path = 'C:/Users/abelp/Desktop/Project_Affordance/Project_Affordances/data'
    else:
        base_path = '/var/scratch/apuigses/data'

    # Get image paths and labels (scores for each image):
    im_label_path = os.path.join(base_path, 'images.mat')
    im_path = os.path.join(base_path, 'images')
    label_path = os.path.join(base_path, 'attributeLabels_continuous.mat')
    attribute_names_path = os.path.join(base_path, 'attributes.mat')

    # Create dataset
    dataset = SunDataset(im_label_path, im_path, label_path, attribute_names_path, num_attributes) # hyperparameter
    class_names = dataset.attribute_names
    # print(class_names)
    # img_tmp, label_tmp = dataset[1]
    # print(img_tmp.shape, label_tmp.shape)
    # print(label_tmp)

    # Create training and testing datasets
    train_dataset, test_dataset = make_datasets(dataset=dataset,
                                                test_size=test_size) # 10%
    print(test_dataset[0][1].shape) # torch.Size([3, 400, 400])

    # Apply transforms to datasets
    apply_transforms(train_dataset.dataset)
    apply_transforms(test_dataset.dataset)
    # print(test_dataset[0][0].shape) # torch.Size([3, 224, 224])

    # Create dataloaders
    train_loader, test_loader = make_dataloaders(train_dataset, test_dataset, batch_size=batch_size, num_workers=2)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    #set_seeds() # to control randomness

    # Setup model
    model = get_model(model_name=model_name,
                      device=device,
                      frozen_layers=frozen_layers,
                      num_attributes=num_attributes)

    model_name = f'{model_name}attri{num_attributes}frozen{frozen_layers}'

    # Load weights
    if model_weights_path is not None:
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(device)))

    # For visualization:
    # print(summary(model=model,
    #         input_size=(1,3,224,224),
    #         verbose=0,
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"]))

    # Setup loss_fn
    loss_fn = nn.BCELoss()

    # Train model
    if command == 'train':

        train_path = Path('results/training')
        train_path.mkdir(parents=True,
                          exist_ok=True)

        test_path = Path('results/testing')
        test_path.mkdir(parents=True,
                         exist_ok=True)

        train_model_path = Path(os.path.join(train_path, model_name))
        train_model_path.mkdir(parents=True,
                         exist_ok=True)

        test_model_path = Path(os.path.join(test_path, model_name))
        test_model_path.mkdir(parents=True,
                              exist_ok=True)

        # Set up optimizer
        if optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model (while also testing) and store the results
        train_results, test_results, train_results_precision, test_results_precision = train_model(model=model,
                                                                                       train_loader=train_loader,
                                                                                       test_loader=test_loader,
                                                                                       loss_fn=loss_fn,
                                                                                       optimizer=optimizer,
                                                                                       device=device,
                                                                                       num_epochs=num_epochs,
                                                                                       class_names=class_names,
                                                                                       num_attributes=num_attributes,
                                                                                       model_name=model_name,
                                                                                       test_model_path=test_model_path,
                                                                                       train_model_path=train_model_path)

        # Print out
        print(f"Results of training: {train_results}")

        # Store, save the results
        save_results(results=train_results,
                     results_precision=train_results_precision,
                     model_name=model_name,
                     target_dir=train_model_path)

        save_results(results=test_results,
                     results_precision=test_results_precision,
                     model_name=model_name,
                     target_dir=test_model_path)

        # Save model
        save_model(model=model,
                   target_dir=models_folder_path,
                   model_name=model_name + '.pth')

    # Evaluate the model
    elif command == 'eval':

        # Create a path to the model
        eval_path = Path('eval_results')
        eval_path.mkdir(parents=True,
                        exist_ok=True)

        model_path = os.path.join(eval_path, model_name)
        model_path = Path(model_path)
        model_path.mkdir(parents=True,
                         exist_ok=True)

        # Test the model
        results_eval, results_precision = test_model(model=model,
                                          dataloader=test_loader,
                                          loss_fn=loss_fn,
                                          print_time_eval=True,
                                          num_attributes=num_attributes,
                                          device=device)

        # Print out
        print("Results of evaluation:", results_eval)

        # Store, save the results
        save_results(results=results_eval,
                     results_precision=results_precision,
                     model_name=model_name,
                     target_dir=model_path)

        # Plot and export precision to wandb
        eval_clean_class_precision = [-0.1 if np.isnan(prec) else prec for prec in results_precision.to("cpu").numpy()] # Prepare precision scores. Indicate the nan with negative bar

        plot_class_precision(class_precision=eval_clean_class_precision,
                             class_names=class_names,
                             epoch=None, # epoch not applicable during evaluation test
                             target_dir=model_path)

    wandb.finish()

if __name__ == "__main__":
    # This ensures the following code only runs when the script is the main program.
    main()