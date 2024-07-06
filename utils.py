
import torch
from PIL import Image
from pathlib import Path
from scipy.io import loadmat
import os
from net2brain.architectures.implemented_models import places365_net
from net2brain.architectures.implemented_models.places365_net import Lambda
from torch import nn
import numpy as np
import torchvision.models as models
import pandas as pd
from net2brain.architectures.timm_models import Timm
from torch.utils.data import Subset
from torchvision import transforms
import wandb

class SunDataset(object):
    def __init__(self, im_label_path, im_path, label_path, attribute_names_path, type_attributes, filter_attributes=True, transform=None):

        images = loadmat(im_label_path)
        im_list = [list(images['images'][i][0])[0] for i in range(len(images['images']))]
        imgs = [os.path.join(im_path, i) for i in im_list]

        # Labels / Scores
        labels_load = loadmat(label_path)
        labels = labels_load['labels_cv']

        # Ambiguous images excluded
        ambiguous_mask = np.isclose(labels, 1 / 3) # Identify ambiguous attributes with a score of 0.333...
        labels[ambiguous_mask] = -1

        labels[labels > 0] = 1 # Binary Classification task: either 0 or 1

        # Attribute names
        attribute_names = loadmat(attribute_names_path)
        attribute_names = [str(attribute[0][0]) for attribute in attribute_names['attributes']]

        # Attributes to exclude (less than 300 instances)
        attributes_to_exclude = [4,10,11,12,15,17,25,28,30,33,34,35,39,46,52,57,63,65,67,69,70,72,73,80,84,99]

        # Create a mask for the attributes to keep
        keep_attributes_mask = np.ones(len(attribute_names), dtype=bool)
        keep_attributes_mask[attributes_to_exclude] = False

        # Filter labels and attribute names
        filtered_labels = labels[:, keep_attributes_mask]
        filtered_attribute_names = np.array(attribute_names)[keep_attributes_mask].tolist()

        if type_attributes == "all":
            if filter_attributes:
                self.labels = filtered_labels
                self.attribute_names = filtered_attribute_names
            else:
                self.labels = labels
                self.attribute_names = attribute_names

        elif type_attributes == 'affordances':
            if filter_attributes:
                self.labels = filtered_labels[:, :np.sum(keep_attributes_mask[:36])]
                self.attribute_names = filtered_attribute_names[:np.sum(keep_attributes_mask[:36])]
            else:
                self.labels = labels[:, :36]
                self.attribute_names = attribute_names[:36]

        elif type_attributes == "action_affordances":
            # Define the indices for action affordances attributes
            action_affordances_indices = [0, 1, 2, 6, 7, 14]

            self.labels = labels[:, action_affordances_indices]
            self.attribute_names = [attribute_names[i] for i in action_affordances_indices]

        elif type_attributes == 'materials':
            if filter_attributes:
                self.labels = filtered_labels[:, np.sum(keep_attributes_mask[:36]):np.sum(keep_attributes_mask[:74])]
                self.attribute_names = filtered_attribute_names[np.sum(keep_attributes_mask[:36]):np.sum(keep_attributes_mask[:74])]
            else:
                self.labels = labels[:, 36:74]
                self.attribute_names = self.attribute_names[36:74]

        elif type_attributes == 'surface_properties':
            if filter_attributes:
                self.labels = filtered_labels[:, np.sum(keep_attributes_mask[:74]):np.sum(keep_attributes_mask[:87])]
                self.attribute_names = filtered_attribute_names[np.sum(keep_attributes_mask[:74]):np.sum(keep_attributes_mask[:87])]
            else:
                self.labels = labels[:, 74:87]
                self.attribute_names = self.attribute_names[74:87]

        elif type_attributes == 'spatial_envelope':
            if filter_attributes:
                self.labels = filtered_labels[:, np.sum(keep_attributes_mask[:87]):]
                self.attribute_names = filtered_attribute_names[np.sum(keep_attributes_mask[:87]):]
            else:
                self.labels = labels[:, 87:]
                self.attribute_names = self.attribute_names[87:]

        # Now, filter images where not a single attribute has a score of 1
        has_positive_attribute = np.max(self.labels, axis=1) == 1
        self.imgs = [img for img, has_attr in zip(imgs, has_positive_attribute) if has_attr]
        self.labels = self.labels[has_positive_attribute]

        # Transforms
        self.transform = transform

    def __getitem__(self, idx):
        # load images
        img = Image.open(self.imgs[idx])
        img = img.convert('RGB')  # Ensure image is in RGB format

        if self.transform:
            img = self.transform(img)  # Apply the transformations defined outside
        else:
            pass
            # # Fallback transformations if none are provided
            # img = transforms.functional.resize(img, (400, 400))
            # img = transforms.functional.to_tensor(img)
            # img = img.view(3, 400, 400)
            # img = img / 255  # This line is technically unnecessary due to to_tensor

        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.imgs)

def make_dataloaders(train_dataset, test_dataset, batch_size, num_workers=2):
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return train_loader, test_loader


def apply_transforms(dataset, image_size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Apply transformations to a dataset.

    Parameters:
    - dataset: The dataset to transform.
    - image_size: A tuple of the desired image size.
    - mean: The mean for normalization.
    - std: The standard deviation for normalization.

    Returns:
    - None, but modifies the dataset's transform attribute in place.
    """
    dataset.transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def get_model(model_name, frozen_layers, num_conv_unfrozen, num_attributes, pretrained, device):
    if model_name == 'resnet_places365':
        # Instantiate a new model
        model = places365_net.get_resnet50_places365(pretrained=pretrained)
        # Freeze the specified parameters
        if frozen_layers:
            for param in model.model.parameters():
                param.requires_grad = False  # Freezing all the parameters
        # Change head of the model
        num_ftrs = model.model[-1][-1].in_features
        model.model[-1] = nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x),
            nn.Linear(num_ftrs, num_attributes),  # (2048, 102)
            nn.Sigmoid())  # add a sigmoid (not necessary to do here)

    elif model_name == 'resnet_imagenet':
        # Instantiate a new model
        model = models.wide_resnet50_2(pretrained=pretrained)
        # Freeze the specified parameters
        if frozen_layers:
            for param in model.parameters():
                param.requires_grad = False  # Freezing all the parameters
        # Change head of the model
        num_ftrs = model.fc.in_features # Getting the number of input features to the last layer
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_attributes),  # Change the output features to 'num_attributes'
            nn.Sigmoid())

    elif model_name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
        if frozen_layers:
            for param in model.parameters():
                param.requires_grad = False  # Freezing all the parameters
        # Replace the classifier part of AlexNet
        num_ftrs = model.classifier[6].in_features  # Getting the number of input features to the last layer
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, num_attributes),  # Change the output features to 'num_attributes'
            nn.Sigmoid()
        )

    elif model_name == "vit":
        # Instantiate the Timm class with the specific model and device
        timm_instance = Timm(model_name='vit_base_patch16_224', device=device)
        # Get the pretrained model (set pretrained=True to use the pretrained version)
        model = timm_instance.get_model(pretrained=pretrained)
        if frozen_layers:
            for param in model.parameters():
                param.requires_grad = False  # Freezing all the parameters
        num_ftrs = model.head.in_features # Getting the number of input features to the last layer
        model.head = nn.Sequential(
            nn.Linear(num_ftrs, num_attributes),  # Change the output features to 'num_attributes'
            nn.Sigmoid())

    # Unfreeze a certain number of layers
    if num_conv_unfrozen != 0:
        unfreeze_n_last_trainable_layers(model, num_conv_unfrozen)
    return model

def save_model(model, target_dir, model_name):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)

# Set seeds
def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def unfreeze_n_last_trainable_layers(model, num_conv_unfrozen):
    conv_layers_indices = []

    for i, layer in enumerate(model.modules()):
        if isinstance(layer, torch.nn.Conv2d):  # Check if layer is a Conv2D layer
            conv_layers_indices.append(i)

    index_conv = conv_layers_indices[-num_conv_unfrozen]
    layers = [layer for layer in model.modules()]

    for layer in layers[index_conv:]:
        for param in layer.parameters():
            param.requires_grad = True

def create_model_path(model_full_name, type_attributes, fold, last_epoch=False):
    # Base folder for saving models
    if last_epoch:
        base_models_folder_path = "last_epoch_models"
    else:
        base_models_folder_path = 'saved_models'

    # Specific folder for the type of attributes
    specific_models_folder_path = os.path.join(base_models_folder_path, type_attributes)

    # Ensure the base models folder exists
    if not os.path.exists(base_models_folder_path):
        os.makedirs(base_models_folder_path)
        print(f"Directory '{base_models_folder_path}' created")
    else:
        print(f"Directory '{base_models_folder_path}' already exists")

    # Ensure the specific type_attributes folder exists
    if not os.path.exists(specific_models_folder_path):
        os.makedirs(specific_models_folder_path)
        print(f"Subdirectory '{specific_models_folder_path}' created")
    else:
        print(f"Subdirectory '{specific_models_folder_path}' already exists")

    # Prepare the model filename with or without the fold suffix
    if fold is not None:
        fold_suffix = f"_fold{fold}"
    else:
        fold_suffix = ''

    # Construct the full path to the model file
    model_path = os.path.join(specific_models_folder_path, f'{model_full_name}{fold_suffix}.pth')

    return model_path

def save_best_results(all_folds_best_metrics, all_folds_best_precision, class_names, train_model_path):

    # Ensure the specific type_attributes folder exists
    if not os.path.exists(train_model_path):
        os.makedirs(train_model_path)
        print(f"Subdirectory '{train_model_path}' created")
    else:
        print(f"Subdirectory '{train_model_path}' already exists")

    df = pd.DataFrame.from_dict(all_folds_best_metrics, orient='index')
    filename = "fold_best_metrics.xlsx"

    df_precision = pd.DataFrame.from_dict(all_folds_best_precision, orient='index')
    filename_precision = "fold_best_precision.xlsx"
    # Set the column names to class_names
    df_precision.columns = class_names
    df_precision.reset_index(inplace=True)

    full_path = os.path.join(train_model_path, filename)
    full_path_precision = os.path.join(train_model_path, filename_precision)

    # Save the averages to an Excel file
    df.to_excel(full_path, index=False)
    df_precision.to_excel(full_path_precision, index=False)


def setup_device():
    """Setup device for PyTorch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device

def aggregate_epoch_wise_results(all_folds_results, num_epochs):
    """Aggregate results for each epoch across all folds."""
    epoch_wise_results = [{
        'epoch': epoch,
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': [],
        'train_precision': [],
        'test_precision': [],
        'train_recall': [],
        'test_recall': [],
        'train_f1': [],
        'test_f1': []
    } for epoch in range(num_epochs)]

    for result in all_folds_results:
        for key in result:
            for epoch in range(num_epochs):
                value = result[key][epoch]
                if not np.isnan(value):
                    epoch_wise_results[epoch][key].append(value)

    for epoch_data in epoch_wise_results:
        for key in epoch_data:
            if key != 'epoch':
                values = epoch_data[key]
                avg_value = np.mean(values) if values else np.nan
                epoch_data[key] = round(avg_value, 4) if 'loss' in key else round(avg_value, 2)

    return epoch_wise_results


def save_epoch_results_to_excel(epoch_wise_results, class_names, train_model_path):
    """Save epoch-wise results and precision per class to Excel."""
    df_results = pd.DataFrame(epoch_wise_results)

    main_results_path = Path(os.path.join(train_model_path, f"results.xlsx"))
    df_results.to_excel(main_results_path, index=False)


def log_results_to_wandb(epoch_wise_results, num_epochs):
    """Log results to Weights and Biases."""
    aggregated_results = {metric_name: [epoch_data[metric_name] for epoch_data in epoch_wise_results] for metric_name in epoch_wise_results[0] if metric_name != 'epoch'}

    for epoch in range(num_epochs):
        log_data = {metric_name: values[epoch] for metric_name, values in aggregated_results.items()}
        wandb.log(log_data, step=epoch)

def aggregate_and_log_results(all_folds_results, all_folds_best_metrics, all_folds_best_precision, num_epochs, class_names, train_model_path):
    """Aggregate and log results from all folds."""
    best_average_precision = np.mean([metrics["test_precision"] for metrics in all_folds_best_metrics.values()])
    best_average_recall = np.mean([metrics["test_recall"] for metrics in all_folds_best_metrics.values()])
    wandb.log({"best_average_precision": best_average_precision, "best_average_recall": best_average_recall})

    max_test_precision = max(fold['test_precision'] for fold in all_folds_best_metrics.values())
    max_test_recall = max(fold['test_recall'] for fold in all_folds_best_metrics.values())
    wandb.log({"max_test_precision": max_test_precision, "max_test_recall": max_test_recall})

    save_best_results(all_folds_best_metrics, all_folds_best_precision, class_names, train_model_path)

    epoch_wise_results = aggregate_epoch_wise_results(all_folds_results, num_epochs)
    save_epoch_results_to_excel(epoch_wise_results, class_names, train_model_path)
    log_results_to_wandb(epoch_wise_results, num_epochs)


def setup_paths(args):
    """Setup paths for data and results. (locally vs cluster Das-4)"""
    if args.host == 'local':
        base_path = 'C:/Users/abelp/Desktop/Project_Affordance/Project_Affordances/data'
    else:  # path in DAS-4 server
        base_path = '/var/scratch/apuigses/data'

    im_label_path = os.path.join(base_path, 'images.mat')
    im_path = os.path.join(base_path, 'images')
    label_path = os.path.join(base_path, 'attributeLabels_continuous.mat')
    attribute_names_path = os.path.join(base_path, 'attributes.mat')

    return im_label_path, im_path, label_path, attribute_names_path