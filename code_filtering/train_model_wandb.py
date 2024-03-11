
import time
from tqdm.auto import tqdm
import torch
from utils_filtering import accuracy_fn, plot_class_precision
import numpy as np
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score


# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(model, dataloader, loss_fn, optimizer, num_attributes, device=device):

    # Put the model in train mode
    model.train()
    model.to(device)

    # Initialize train loss,  batch number, counters
    train_loss, batch = 0, 0

    TP = torch.zeros(num_attributes, device=device)  # True Positives for each attribute
    FP = torch.zeros(num_attributes, device=device)  # False Positives for each attribute
    FN = torch.zeros(num_attributes, device=device) # False Negatives for each attribute
    TN = torch.zeros(num_attributes, device=device) # True Negatives

    # Loop through data loader and data batches
    for inputs, labels in dataloader: # for example, 128x3x224x224  and 128x102

        # Update batch number
        batch += 1
        print(f"Train batch: {batch}")

        # Send data to target device
        inputs, labels = inputs.to(device), labels.float().to(device)

        # Forward pass
        y_preds = model(inputs)  # y_preds will be in the form of probabilities (after softmax)

        final_output = torch.round(y_preds)

        # Filter out attributes with single vote
        mask = labels != -1  # -1 is a placeholder for excluded attributes with a single vote --> these are ambiguous

        for i in range(num_attributes):
            TP[i] += torch.logical_and(final_output[:, i] == 1, labels[:, i] == 1).sum()
            FP[i] += torch.logical_and(final_output[:, i] == 1, labels[:, i] == 0).sum()
            FN[i] += torch.logical_and(final_output[:, i] == 0, labels[:, i] == 1).sum()
            TN[i] += torch.logical_and(final_output[:, i] == 0, labels[:, i] == 0).sum()

        # Calculate the loss
        loss = loss_fn(y_preds[mask], labels[mask])
        # Add batch loss to total train_loss
        train_loss += loss.item()

        # Optimizer zero grad
        optimizer.zero_grad()
        # Loss backward
        loss.backward()
        # Optimizer step
        optimizer.step()

    # Calculate accuracy per class
    accuracy_per_class = (TP + TN) / (TP + TN + FP + FN)
    valid_accuracies = accuracy_per_class[torch.isfinite(accuracy_per_class)]  # Exclude NaNs and Infs
    average_accuracy = valid_accuracies.mean().item()  # Average accuracy across valid classes

    # Calculate precision per attribute and average
    precision_per_class = TP / (TP + FP)
    valid_precisions = precision_per_class[torch.isfinite(precision_per_class)]  # Exclude NaNs and Infs
    average_precision = valid_precisions.mean().item()  # Average precision across valid classes

    # Calculate recall per attribute and average
    recall_per_class = TP / (TP + FN)
    valid_recalls = recall_per_class[torch.isfinite(recall_per_class)]  # Exclude NaNs and Infs
    average_recall = valid_recalls.mean().item()  # Average precision across valid classes

    # Calculate F1 score per class and average
    f1_score_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
    valid_f1_scores = f1_score_per_class[torch.isfinite(f1_score_per_class)]  # Exclude NaNs and Infs
    average_f1_score = valid_f1_scores.mean().item()  # Average F1 score across valid classes

    # Train loss average
    train_loss /= len(dataloader)

    # Return a dictionary with the averages + list of precision per attribute
    metrics = {"train_loss": round(train_loss, 4),
               "train_average_accuracy": round(average_accuracy, 2),
               "train_average_precision": round(average_precision, 2),
               "train_average_recall": round(average_recall, 2),
               "train_average_f1": round(average_f1_score, 2)
               }

    return metrics, precision_per_class



def train_model(model, train_loader, test_loader, loss_fn, optimizer, num_epochs, class_names, num_attributes, model_name, train_model_path, test_model_path, device=device):
    # Start timer
    time_start = time.time()

    # Create results dictionary
    train_metric_names = ["train_loss",
                        "train_average_accuracy",
                        "train_average_precision",
                        "train_average_recall",
                        "train_average_f1"]

    test_metric_names = ["test_loss",
                          "test_average_accuracy",
                          "test_average_precision",
                          "test_average_recall",
                          "test_average_f1"]

    train_results = {metric: [] for metric in train_metric_names}
    test_results = {metric: [] for metric in test_metric_names}

    train_results_precision = {"class_precision": []}
    test_results_precision = {"class_precision": []}

    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch nº {epoch}")
        print("-" * 10)

        # Training step
        train_metrics, train_class_precision = train_step(model=model,
                                               dataloader=train_loader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               num_attributes=num_attributes,
                                               device=device)


        # Validation (results of testing model after every epoch)
        test_metrics, test_class_precision = test_model(model=model,
                                                        dataloader=test_loader,
                                                        loss_fn=loss_fn,
                                                        num_attributes=num_attributes,
                                                        print_time_eval=False,
                                                        device=device)


        train_class_precision, test_class_precision = train_class_precision.to("cpu"), test_class_precision.to("cpu")

        # Experiment tracking with W&B
        train_metrics_with_epoch = train_metrics.copy()
        train_metrics_with_epoch["epoch"] = epoch

        test_metrics_with_epoch = test_metrics.copy()
        test_metrics_with_epoch["epoch"] = epoch

        wandb.log(train_metrics_with_epoch)
        wandb.log(test_metrics_with_epoch)

        train_clean_class_precision = [-0.1 if np.isnan(prec) else prec for prec in train_class_precision]
        test_clean_class_precision = [-0.1 if np.isnan(prec) else prec for prec in test_class_precision]

        # Plots for train results
        plot_class_precision(class_precision=train_clean_class_precision,
                             class_names=class_names,
                             epoch=epoch,
                             target_dir=train_model_path)

        # Plots for test results
        plot_class_precision(class_precision=test_clean_class_precision,
                             class_names=class_names,
                             epoch=epoch,
                             target_dir=test_model_path)

        # Append results
        for metric_name, metric_value in train_metrics.items(): # Train metrics
            train_results[metric_name].append(metric_value)
            print(f"{metric_name}: {metric_value}", end=" | ")
        print()

        for metric_name, metric_value in test_metrics.items(): # Test metrics
            test_results[metric_name].append(metric_value)
            print(f"{metric_name}: {metric_value}", end=" | ")
        print()

        train_results_precision["class_precision"].append(train_class_precision) # Train precision
        test_results_precision["class_precision"].append(test_class_precision) # Test precision


    # Stop the timer and calculate elapsed time
    time_end = time.time()
    time_elapsed = time_end - time_start

    # Print out time elapsed
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_results, test_results, train_results_precision, test_results_precision


def test_model(model, dataloader, loss_fn, num_attributes, print_time_eval=True, device=device):

    if print_time_eval:
        # Start timer
        time_start = time.time()

    # Put model in eval mode and send it to device
    model.eval()
    model.to(device)

    # Initialize train loss,  batch number, counters
    test_loss, batch = 0, 0

    TP = torch.zeros(num_attributes, device=device)  # True Positives for each attribute
    FP = torch.zeros(num_attributes, device=device)  # False Positives for each attribute
    FN = torch.zeros(num_attributes, device=device)  # False Negatives for each attribute
    TN = torch.zeros(num_attributes, device=device)  # True Negatives

    # Turn on inference context manager
    with torch.inference_mode():

        # Loop through DataLoader batches
        for inputs, labels in dataloader:

            # Update batch number
            batch += 1
            print(f"Test batch: {batch}")

            # Send data to target device
            inputs, labels = inputs.to(device), labels.float().to(device)

            # 1. Forward pass
            y_preds = model(inputs)  # test_preds will be in the form of probabilities (after softmax)
            final_output = torch.round(y_preds)

            # Filter out attributes with single vote
            mask = labels != -1 # -1 indicates attributes with a single vote --> ambiguous

            for i in range(num_attributes):
                TP[i] += torch.logical_and(final_output[:, i] == 1, labels[:, i] == 1).sum()
                FP[i] += torch.logical_and(final_output[:, i] == 1, labels[:, i] == 0).sum()
                FN[i] += torch.logical_and(final_output[:, i] == 0, labels[:, i] == 1).sum()
                TN[i] += torch.logical_and(final_output[:, i] == 0, labels[:, i] == 0).sum()

            # Calculate batch loss
            loss = loss_fn(y_preds[mask], labels[mask])
            #  Add batch loss to total train_loss
            test_loss += loss.item()  # * inputs.size(0) # Normalize by multiplying by batch_size

        # Calculate accuracy per class
        accuracy_per_class = (TP + TN) / (TP + TN + FP + FN)
        valid_accuracies = accuracy_per_class[torch.isfinite(accuracy_per_class)]  # Exclude NaNs and Infs
        average_accuracy = valid_accuracies.mean().item()  # Average accuracy across valid classes

        # Calculate precision per attribute and average
        precision_per_class = TP / (TP + FP)
        valid_precisions = precision_per_class[torch.isfinite(precision_per_class)]  # Exclude NaNs and Infs
        average_precision = valid_precisions.mean().item()  # Average precision across valid classes

        # Calculate recall per attribute and average
        recall_per_class = TP / (TP + FN)
        valid_recalls = recall_per_class[torch.isfinite(recall_per_class)]  # Exclude NaNs and Infs
        average_recall = valid_recalls.mean().item()  # Average precision across valid classes

        # Calculate F1 score per class and average
        f1_score_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
        valid_f1_scores = f1_score_per_class[torch.isfinite(f1_score_per_class)]  # Exclude NaNs and Infs
        average_f1_score = valid_f1_scores.mean().item()  # Average F1 score across valid classes

        if print_time_eval:
            # Stop the timer and calculate elapsed time
            time_end = time.time()
            time_elapsed = time_end - time_start

            # Print out time elapsed
            print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # Train loss average
        test_loss /= len(dataloader)

        metrics = {"test_loss": round(test_loss, 4),
                   "test_average_accuracy": round(average_accuracy, 2),
                   "test_average_precision": round(average_precision, 2),
                   "test_average_recall": round(average_recall, 2),
                   "test_average_f1": round(average_f1_score, 2)
                   }

        wandb.log(metrics)

    return metrics, precision_per_class
