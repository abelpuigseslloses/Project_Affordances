
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from scipy.io import loadmat
import os
from net2brain.architectures.implemented_models import places365_net
from net2brain.architectures.implemented_models.places365_net import Lambda
from torch import nn
import numpy as np
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import wandb
import pandas as pd

# if torchvision is updated: then from torchvision.models.resnet import ResNet50_Weights

class SunDataset(object):
    def __init__(self, im_label_path, im_path, label_path, attribute_names_path, num_attributes, transform=None):
        # Images
        images = loadmat(im_label_path)
        im_list = [list(images['images'][i][0])[0] for i in range(len(images['images']))]
        self.imgs = [os.path.join(im_path, i) for i in im_list]

        # Labels / Scores
        labels_load = loadmat(label_path)
        labels = labels_load['labels_cv']
        # Identify ambiguous attributes with a score of 0.333...
        ambiguous_mask = np.isclose(labels, 1 / 3)
        labels[ambiguous_mask] = -1
        labels[labels > 0] = 1
        self.labels = labels[:, :num_attributes]  # Trim the scores for each label based on num_classes (e.g., only first 36)

        # Attribute names
        attribute_names = loadmat(attribute_names_path)
        self.attribute_names = [str(attribute[0][0]) for attribute in attribute_names['attributes']]
        self.attribute_names = self.attribute_names[:num_attributes]  # Trim the attribute names list based on num_classes

        # Transforms
        self.transform = transform

    def __getitem__(self, idx):
        # load images
        img = Image.open(self.imgs[idx])
        img = img.convert('RGB')  # Ensure image is in RGB format

        if self.transform:
            img = self.transform(img)  # Apply the transformations defined outside
        else:
            # Fallback transformations if none are provided
            img = transforms.functional.resize(img, (400, 400))
            img = transforms.functional.to_tensor(img)
            img = img.view(3, 400, 400)
            img = img / 255  # This line is technically unnecessary due to to_tensor

        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.imgs)

def get_model(model_name, device, frozen_layers, num_attributes):
    if model_name == 'resnet_places365':
        # Instantiate a new model
        model = places365_net.get_resnet50_places365(pretrained=True).to(device)
        # Freeze the specified parameters
        if frozen_layers != 0:
            for i in range(frozen_layers):
                for param in model.model[i].parameters():
                    param.requires_grad = False  # Freezing all the parameters
        # Change head of the model
        num_ftrs = model.model[-1][-1].in_features
        model.model[-1] = nn.Sequential(
            Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x),
            nn.Linear(num_ftrs, num_attributes),  # (2048, 102)
            nn.Sigmoid()  # add a sigmoid (not necessary to do here)
        )

    elif model_name == 'resnet_imagenet':
        pass
        # Instantiate a new model
        # weights = ResNet50_Weights.DEFAULT
        # model = resnet50(weights=weights)  # Trained for ImageNet object classification
        #
        # # Freeze all the parameters
        # if frozen_layers != 0:
        #     p_count = 0
        #     for param in model.parameters():
        #         p_count += 1
        #         if p_count < frozen_layers: # frozen layers: from 0 (no freeze) to 161 (full freeze)
        #             param.requires_grad = False

        # # Change the head of the model
        # num_ftrs = 2048
        # model.fc = nn.Sequential(
        #  nn.Linear(in_features=num_ftrs, out_features=102),
        #  nn.Sigmoid()) # We include Sigmoid here --> output will be a tensor with probs from 0 (absence) to 1 (presence)

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


def accuracy_fn(final_output, labels):
    """Calculates accuracy.

     Args:
        final_output: tensor of 1s or 0s, the output of applying a threshold to y_preds
        labels: tensor of the target labels, also either 1s or 0s

    Returns:
        batch_acc: proportion of correct over total predictions
    """

    correct_preds = torch.sum(final_output == labels)
    # total_preds = labels.size(0) * labels.size(1) # batch size * num_classes, e.g.: 128*102
    total_preds = len(final_output)
    batch_acc = correct_preds.float() / torch.tensor(total_preds).float()

    return batch_acc


def plot_class_precision(class_precision, class_names, epoch, target_dir):
    """Plots class precision per attribute"""
    assert len(class_precision) == len(class_names), "class_precision and class_names must be the same length"

    plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
    plt.bar(class_names, class_precision, color='skyblue')  # Create a bar chart
    plt.xlabel('Class Name')  # Label for the x-axis
    plt.ylabel('Precision Score')  # Label for the y-axis
    plt.title('Class Precision Scores')  # Title of the plot
    plt.xticks(rotation=90)  # Rotate class names for better visibility if needed
    plt.ylim(-0.1, 1)
    plt.grid(axis='y')  # Add horizontal grid lines for better readability

    plt.tight_layout()  # Adjust subplot parameters to give specified padding

    if epoch is not None:
        filename = f'precision_epoch_{epoch}.png'
    else:
        filename = 'precision_epoch_0.png'

    plot_path = os.path.join(target_dir, filename)

    # After generating plot and defining path, save it using:
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory

    if epoch is not None:
        wandb.log({f"Class Precision Scores Epoch {epoch}": [wandb.Image(plot_path, caption=f"Epoch {epoch}")]})
    else:
        wandb.log({f"Evaluation - Class Precision Scores": [wandb.Image(plot_path, caption="Evaluation")]})

def save_results(results, results_precision, model_name, target_dir):

    # Make a pandas df
    main_results_df = pd.DataFrame([results])

    if not isinstance(results_precision, dict):
        results_precision = results_precision.to("cpu")

    precision_class_results_df = pd.DataFrame(results_precision)

    # Make directories
    precision_dir = Path(os.path.join(target_dir, 'precision'))
    precision_dir.mkdir(parents=True,
                        exist_ok=True)

    main_dir = Path(os.path.join(target_dir, 'main'))
    main_dir.mkdir(parents=True,
                        exist_ok=True)

    # Main results
    main_results_path = Path(os.path.join(main_dir, f"results_{model_name}.csv"))
    main_results_df.to_csv(main_results_path, index=False)

    # Precision per class results
    precision_class_results_path = Path(os.path.join(precision_dir, f"precision_{model_name}.csv"))
    precision_class_results_df.to_csv(precision_class_results_path, index=False)