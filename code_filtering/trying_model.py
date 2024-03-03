
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchinfo import summary
from torch import nn



weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)  # Trained for ImageNet object classification


# Freeze all the parameters

# for param in model.parameters():
#     param.requires_grad = False

frozen_layers = 161
if frozen_layers != 0:
    p_count = 0
    for param in model.parameters():
        p_count += 1
        if p_count < frozen_layers:
            param.requires_grad = False

# # Change the head of the model
# num_ftrs = 2048
# model.fc = nn.Sequential(
#     nn.Linear(num_ftrs, 102),
#     nn.Sigmoid()) # We include Sigmoid here --> output will be a tensor with probs from 0 (absence) to 1 (presence)

print(summary(model=model,
              input_size=(1, 3, 224, 224),
              verbose=0,
              col_names=["input_size", "output_size", "num_params", "trainable"],
              col_width=20,
              row_settings=["var_names"]))