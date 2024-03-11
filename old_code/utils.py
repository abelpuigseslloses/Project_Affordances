
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from scipy.io import loadmat
import os
from net2brain.architectures.implemented_models import places365_net
from net2brain.architectures.implemented_models.places365_net import Lambda
from torch import nn

class SunDataset(object):
    def __init__(self, im_label_path, im_path, label_path, attribute_names_path, num_classes = 102, transform=None):
        # Images
        images = loadmat(im_label_path)
        im_list = [list(images['images'][i][0])[0] for i in range(len(images['images']))]
        self.imgs = [os.path.join(im_path, i) for i in im_list]

        # Labels / Scores
        labels_load = loadmat(label_path)
        labels = labels_load['labels_cv']
        # self.labels = (labels > 0).astype(int) # change this
        self.labels_pre = labels
        self.labels = (labels[:, :num_classes] > 0).astype(int)

        # Attribute names
        attribute_names = loadmat(attribute_names_path)
        self.attribute_names = [str(attribute[0][0]) for attribute in attribute_names['attributes']]

        # Trim the attribute names list based on num_classes
        if num_classes > 0:  # Assuming num_classes should always be positive
            self.attribute_names = self.attribute_names[:num_classes]

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


def get_model(model_name, device, frozen_layers=0, num_attributes=102):
    if model_name == 'resnet_places365':
        # Instantiate a new model
        model = places365_net.get_resnet50_places365(pretrained=True).to(device)
        # Freeze all the parameters
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
    return model

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
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
    total_preds = labels.size(0) * labels.size(1) # batch size * num_classes, e.g.: 128*102
    batch_acc = correct_preds.float() / torch.tensor(total_preds).float()

    return batch_acc
