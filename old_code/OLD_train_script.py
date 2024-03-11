#!/usr/bin/env python3

from utils import SunDataset, get_model, accuracy_fn
import torch
from old_no_filter.get_data import make_datasets,apply_transforms, make_dataloaders
from torch import nn
import torch.optim as optim
from train_model import train_model

### TO DO:
# precision, recall, f1 metrics ### done in eval and training
# save model
# scheduler
# argparse
# experiment tracking

### ARGPARSE
# im_label_path
# im_path
# label_path
# test_size = 0.1
# batch_size = 128
# model_name ="resnet_places365"
# command = "train" / "test"
# num_attributes = 102 (num. of attributes: either 102 or 36)
# frozen_layers=12 (12 for all of them, 0 for no freeze)
# loss_fn = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

local_run = True
num_attributes = 102

def main():

    if local_run:
        # Get image paths and labels (scores for each image):
        im_label_path = '/data/images.mat'
        im_path = 'C:/Users/abelp/Desktop/Research_Project/SUNAttributeDB_Images/images'
        label_path = '/data/attributeLabels_continuous.mat'
        attribute_names_path = '/data/attributes.mat'

    else:
        # Get image paths and labels (scores for each image):
        im_label_path = '/home/apuigses/workspace/data/images.mat'
        im_path = '/home/apuigses/workspace/data/images'
        label_path = '/home/apuigses/workspace/data/attributeLabels_continuous.mat'
        attribute_names_path = '/home/apuigses/workspace/data/attributes.mat'

    # Create dataset
    dataset = SunDataset(im_label_path, im_path, label_path, attribute_names_path, num_attributes) # hyperparameter
    class_names = dataset.attribute_names
    print(class_names)

    # Create training and testing datasets
    train_dataset, test_dataset = make_datasets(dataset=dataset,
                                                test_size=0.1) # 10%

    # print(test_dataset[0][0].shape) # torch.Size([3, 400, 400])

    # Apply transforms to datasets
    apply_transforms(train_dataset.dataset)
    apply_transforms(test_dataset.dataset)

    # print(test_dataset[0][0].shape) # torch.Size([3, 224, 224])

    # Create dataloaders
    train_loader, test_loader = make_dataloaders(train_dataset, test_dataset, batch_size=128, num_workers=2)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Setup model

    model = get_model(model_name="resnet_places365", device=device, frozen_layers=12, num_attributes=num_attributes)

    load_model = False
    if load_model:
        # model.load_state_dict(torch.load(args.path_to_model))
        pass

    # print(summary(model=model,
    #         input_size=(1,3,224,224),
    #         verbose=0,
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"]))

    # Setup loss function, optimizer, accuracy_fn
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)



    # Train model
    results = train_model(model=model,
                          train_loader=train_loader,
                          loss_fn=loss_fn,
                          optimizer=optimizer,
                          accuracy_fn=accuracy_fn,
                          device=device,
                          num_epochs=1)
    print(f"Results of training: {results}")

    save_model = True
    model_name = "resnet_places365.pth"
    if save_model:
        torch.save(model.state_dict(), f'saved_models/{model_name}')

    # Evaluate the model
    # results_eval = test_model(model=model,
    #                           test_loader=test_loader,
    #                           loss_fn=loss_fn,
    #                           accuracy_fn=accuracy_fn,
    #                           device=device)
    #
    #
    # print("Results of evaluation:", results_eval)

if __name__ == "__main__":
    # This ensures the following code only runs when the script is the main program.
    main()