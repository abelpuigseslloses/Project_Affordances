#!/usr/bin/env python3

from utils_filtering import SunDataset, get_model, accuracy_fn, set_seeds
import torch
from get_data import make_datasets,apply_transforms, make_dataloaders
from torchinfo import summary
from torch import nn
import torch.optim as optim
from train_model_filtering import train_model, test_model
import argparse
import os
import pandas as pd

### TO DO:
# scheduler
# experiment tracking


def main():

    # Create a parser
    parser = argparse.ArgumentParser(description="Getting hyperparameters")

    # Get an arg for the command
    parser.add_argument('--command',
                        default='train',
                        type=str,
                        help="'train' or 'eval'")

    # Get an arg for the model name
    parser.add_argument('--model_name',
                        default='resnet_places365',
                        type=str,
                        help="Use 'resnet_places' or 'resnet_objects'")

    # Get an arg for num_attributes
    parser.add_argument("--num_attributes",
                        default=102,
                        type=int,
                        help="the number of attributes to train on")

    # Get an arg for number of frozen layers
    parser.add_argument("--frozen_layers",
                        default=12,
                        type=int,
                        help="the number of layers to freeze (from 0 to 12 for resnet-places365)")

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
                        default=5,
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
                        default='local',
                        type=str,
                        help="'local' or 'server'")

    # Get our arguments from the parser
    args = parser.parse_args()

    command = args.command
    model_name = args.model_name
    num_attributes = args.num_attributes
    frozen_layers = args.frozen_layers
    model_weights_path = args.model_weights_path
    optimizer = args.optimizer
    test_size = args.test_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    host = args.host

    # Folder to save models
    folder_path = 'saved_models'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created")
    else:
        print(f"Directory '{folder_path}' already exists")

    if host == 'local':
        # Get image paths and labels (scores for each image):
        im_label_path = 'C:/Users/abelp/Desktop/Project_Affordance/Project_Affordances/data/images.mat'
        im_path = 'C:/Users/abelp/Desktop/Research_Project/SUNAttributeDB_Images/images'
        label_path = 'C:/Users/abelp/Desktop/Project_Affordance/Project_Affordances/data/attributeLabels_continuous.mat'
        attribute_names_path = 'C:/Users/abelp/Desktop/Project_Affordance/Project_Affordances/data/attributes.mat'

    else:
        # Get image paths and labels (scores for each image):
        im_label_path = '/home/apuigses/workspace/data/images.mat'
        im_path = '/home/apuigses/workspace/data/images'
        label_path = '/home/apuigses/workspace/data/attributeLabels_continuous.mat'
        attribute_names_path = '/home/apuigses/workspace/data/attributes.mat'

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
    # print(test_dataset[0][0].shape) # torch.Size([3, 400, 400])

    # Apply transforms to datasets
    apply_transforms(train_dataset.dataset)
    apply_transforms(test_dataset.dataset)
    # print(test_dataset[0][0].shape) # torch.Size([3, 224, 224])

    # Create dataloaders
    train_loader, test_loader = make_dataloaders(train_dataset, test_dataset, batch_size=batch_size, num_workers=2)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds()
    #print(device)

    # Setup model
    model = get_model(model_name=model_name,
                      device=device,
                      frozen_layers=frozen_layers,
                      num_attributes=num_attributes)

    # Load weights
    if model_weights_path is not None:
        model.load_state_dict(torch.load(model_weights_path))

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
        # Set up optimizer
        if optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model and store the results
        results = train_model(model=model,
                              train_loader=train_loader,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              accuracy_fn=accuracy_fn,
                              device=device,
                              num_epochs=num_epochs)
        # Print out
        print(f"Results of training: {results}")

        # Save model
        # model_file_path = f'{folder_path}/{model_name}attri{num_attributes}frozen{frozen_layers}.pth'
        # print(f"Saving model to {model_file_path}")
        # torch.save(model.state_dict(), model_file_path)


    # Evaluate the model
    if command == 'eval':
        results_eval, results_precision = test_model(model=model,
                                          test_loader=test_loader,
                                          loss_fn=loss_fn,
                                          accuracy_fn=accuracy_fn,
                                          device=device)
        # Print out
        print("Results of evaluation:", results_eval)

        # Make a pandas df
        main_results_df = pd.DataFrame([results_eval])
        precision_class_results_df = pd.DataFrame(results_precision)

        # Specify the folder path for saving the results
        folder_path_results = 'results'
        if not os.path.exists(folder_path_results):
            os.makedirs(folder_path_results)
            print(f"Directory '{folder_path_results}' created")
        else:
            print(f"Directory '{folder_path_results}' already exists")

        # Define file paths for saving the CSV files
        if model_weights_path != None:
            pretrained = 'trained'
        else:
            pretrained = 'untrained'

        main_results_path = os.path.join(folder_path_results, f"results_{pretrained}_{model_name}attri{num_attributes}frozen{frozen_layers}.csv")
        precision_class_results_path = os.path.join(folder_path_results, f"precision_{pretrained}_{model_name}attri{num_attributes}frozen{frozen_layers}.csv")
        main_results_df.to_csv(main_results_path, index=False)
        precision_class_results_df.to_csv(precision_class_results_path, index=False)

if __name__ == "__main__":
    # This ensures the following code only runs when the script is the main program.
    main()