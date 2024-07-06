from tqdm.auto import tqdm
import torch
from utils import create_model_path
import numpy as np


# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_step(model, dataloader, loss_fn, optimizer, num_attributes, device=device):

    # Put the model in train mode
    model.train()

    # Send the model to device
    model.to(device)

    # Initialize train loss and batch number counters
    train_loss, batch = 0, 0

    TP = torch.zeros(num_attributes, device=device)  # True Positives for each attribute
    FP = torch.zeros(num_attributes, device=device)  # False Positives for each attribute
    FN = torch.zeros(num_attributes, device=device) # False Negatives for each attribute
    TN = torch.zeros(num_attributes, device=device) # True Negatives

    # Loop through data loader and data batches
    for inputs, labels in dataloader: # for example, 64x3x224x224  and 64x102

        # Update batch number
        batch += 1
        if batch % 50 == 0:
            print(f"Train batch: {batch}")

        # Send data to target device
        inputs, labels = inputs.to(device), labels.float().to(device)

        # Forward pass
        y_preds = model(inputs)  # y_preds will be in the form of probabilities (there is a final sigmoid function)

        # Threshold function
        final_output = torch.round(y_preds)

        # Filter out attributes with single vote
        mask = labels != -1  # -1 is a placeholder for excluded attributes with a single vote --> these tend to be ambiguous

        # Recount TP, FP, FN, TN
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

    # Calculate precision_model per attribute and average
    precision_per_class = TP / (TP + FP)
    valid_precisions = precision_per_class[torch.isfinite(precision_per_class)]  # Exclude NaNs and Infs
    average_precision = valid_precisions.mean().item()  # Average precision_model across valid classes

    # Calculate recall per attribute and average
    recall_per_class = TP / (TP + FN)
    valid_recalls = recall_per_class[torch.isfinite(recall_per_class)]  # Exclude NaNs and Infs
    average_recall = valid_recalls.mean().item()  # Average precision_model across valid classes

    # Calculate F1 score per class and average macro f1 score
    f1_score_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
    valid_f1_scores = f1_score_per_class[torch.isfinite(f1_score_per_class)]  # Exclude NaNs and Infs
    # macro_f1_score = valid_f1_scores.mean().item()  # Average F1 score across valid classes

    # Calculate micro F1 score average
    average_f1_score = 2 * average_precision * average_recall / (average_precision + average_recall)

    # Train loss average
    train_loss /= len(dataloader)

    # Return a dictionary with the averages + list of precision_model per attribute
    metrics = {"train_loss": round(train_loss, 4),
               "train_average_accuracy": round(average_accuracy, 2),
               "train_average_precision": round(average_precision, 2),
               "train_average_recall": round(average_recall, 2),
               "train_average_f1": round(average_f1_score, 2)
               }

    return metrics, precision_per_class.tolist()


def train_model(model, train_loader, test_loader, loss_fn, optimizer,
                num_epochs, model_full_name, num_attributes, min_delta,
                patience_threshold, type_attributes, early_stopping,
                device=device, fold=None):

    model_path = create_model_path(model_full_name=model_full_name, type_attributes=type_attributes, fold=fold, last_epoch=False)
    last_epoch_path = create_model_path(model_full_name=model_full_name, type_attributes=type_attributes, fold=fold, last_epoch=True)

    # Initialize
    fold_results = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': [], 'train_precision': [], 'test_precision': [],
                    'train_recall': [], 'test_recall':[], 'train_f1': [], 'test_f1':[]}

    fold_class_precision = {'train_class_precision': [], 'test_class_precision': []}

    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience = 0

    # Variables for tracking the best_beh precision_model and corresponding weights
    best_model_weights = None
    best_epoch_metrics = None
    best_loss_epoch = None
    best_avg_precision = 0

    # Initialize training loop
    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch nº {epoch}")
        print("-" * 5)

        # Training step
        train_metrics, train_class_precision = train_step(model=model,
                                               dataloader=train_loader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               num_attributes=num_attributes,
                                               device=device)

        # After receiving train_metrics and train_class_precision, append
        fold_results['train_loss'].append(train_metrics['train_loss'])
        fold_results['train_accuracy'].append(train_metrics['train_average_accuracy'])
        fold_results['train_precision'].append(train_metrics['train_average_precision'])
        fold_results['train_recall'].append(train_metrics['train_average_recall'])
        fold_results['train_f1'].append(train_metrics['train_average_f1'])
        fold_class_precision['train_class_precision'].append(train_class_precision)

        # Validation (results of testing model after every epoch)
        test_metrics, test_class_precision = test_model(model=model,
                                                        dataloader=test_loader,
                                                        loss_fn=loss_fn,
                                                        num_attributes=num_attributes,
                                                        device=device)

        print("Average class precision_model :", test_metrics["test_average_precision"])
        print("Average class recall :", test_metrics["test_average_recall"])
        print("Average f1 :", test_metrics["test_average_f1"])


        # Calculate and print the average class precision_model for test data,
        # Ensuring it doesn't include more than 10 NaNs
        valid_precisions = [p for p in test_class_precision if not np.isnan(p)]

        # Get at least to recognize 60% of the attributes. THIS COULD BE AN ARGUMENT/PARAMETER TO MODIFY.
        threshold_nan = num_attributes - (60*num_attributes/100)

        if len(test_class_precision) - len(valid_precisions) <= int(threshold_nan):
            avg_class_precision = np.mean(valid_precisions)
            print(f"Average class precision_model (with <={int(threshold_nan)} NaNs): {avg_class_precision}")
        else:
            print(f"More than {int(threshold_nan)} NaNs in this epoch, average precision_model won't be considered for picking the best model")
            avg_class_precision = float('-inf')

        print("Validation loss: ", test_metrics["test_loss"])

        if avg_class_precision > best_avg_precision:
            best_avg_precision = avg_class_precision
            best_epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_metrics['train_loss'],
                'test_loss': test_metrics['test_loss'],
                'train_accuracy': train_metrics['train_average_accuracy'],
                'test_accuracy': test_metrics['test_average_accuracy'],
                'train_precision': train_metrics['train_average_precision'],
                'test_precision': test_metrics['test_average_precision'],
                'train_recall': train_metrics['train_average_recall'],
                'test_recall': test_metrics['test_average_recall'],
                'train_f1': train_metrics['train_average_f1'],
                'test_f1': test_metrics['test_average_f1'],
            }
            best_epoch_precision = test_class_precision

        # Similarly, after the test/evaluation step
        fold_results['test_loss'].append(test_metrics['test_loss'])
        fold_results['test_accuracy'].append(test_metrics['test_average_accuracy'])
        fold_results['test_precision'].append(test_metrics['test_average_precision'])
        fold_results['test_recall'].append(test_metrics['test_average_recall'])
        fold_results['test_f1'].append(test_metrics['test_average_f1'])
        fold_class_precision['test_class_precision'].append(test_class_precision)
        current_val_loss = test_metrics['test_loss']

        if early_stopping:
            print(f"Current validation loss: {current_val_loss}")
            print(f"Best validation loss: {best_val_loss}")
            if current_val_loss < best_val_loss - min_delta:
                best_val_loss = current_val_loss
                patience = 0
                best_model_weights = model.state_dict().copy()
                best_loss_epoch = epoch
            else:
                patience += 1

            print("Patience: ", patience)

        last_epoch_weights = model.state_dict().copy()

        if patience >= patience_threshold:
            print(f'Stopping early at epoch {epoch} due to no improvement in {early_stopping}.')
            break

    print("-----------------------------------------------------")

    # Save model weights
    if best_model_weights is not None:
        # Save the best_beh model weights based on precision_model
        torch.save(best_model_weights, model_path)
        print(f"Model saved with lowest validation loss: {best_val_loss} at epoch {best_loss_epoch}")
    else:
        print("No valid precision_model found within the given NaN constraint.")

    print("-----------------------------------------------------")
    print("Saving last epoch metrics and weights...")
    torch.save(last_epoch_weights, last_epoch_path)
    print("-----------------------------------------------------")

    return fold_results, fold_class_precision, best_epoch_metrics, best_epoch_precision


def test_model(model, dataloader, loss_fn, num_attributes, device=device):

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

        # Calculate precision_model per attribute and average
        precision_per_class = TP / (TP + FP)
        valid_precisions = precision_per_class[torch.isfinite(precision_per_class)]  # Exclude NaNs and Infs
        average_precision = valid_precisions.mean().item()  # Average precision_model across valid classes
        print("Precision per class")
        print(precision_per_class.tolist())
        print("------")
        # Calculate recall per attribute and average
        recall_per_class = TP / (TP + FN)
        valid_recalls = recall_per_class[torch.isfinite(recall_per_class)]  # Exclude NaNs and Infs
        average_recall = valid_recalls.mean().item()  # Average precision_model across valid classes

        print("Recall per class")
        print(recall_per_class.tolist())

        # Calculate F1 score per class and average macro f1 score
        f1_score_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
        valid_f1_scores = f1_score_per_class[torch.isfinite(f1_score_per_class)]  # Exclude NaNs and Infs
        macro_f1_score = valid_f1_scores.mean().item()  # Average F1 score across valid classes

        # Calculate micro F1 score average
        if average_precision + average_recall == 0:
            average_f1_score = 0
        else:
            average_f1_score = 2 * average_precision * average_recall / (average_precision + average_recall)

        # Train loss average
        test_loss /= len(dataloader)

        metrics = {"test_loss": round(test_loss, 4),
                   "test_average_accuracy": round(average_accuracy, 2),
                   "test_average_precision": round(average_precision, 2),
                   "test_average_recall": round(average_recall, 2),
                   "test_average_f1": round(average_f1_score, 2)
                   }

    return metrics, precision_per_class.tolist()
