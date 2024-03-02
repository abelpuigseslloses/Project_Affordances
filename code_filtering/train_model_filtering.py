
import time
from tqdm.auto import tqdm
import torch
from utils_filtering import accuracy_fn
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn=accuracy_fn,
               device=device):

    # Put the model in train mode
    model.train()

    # Initialize train loss, metric values, batch number
    train_loss, train_acc, train_recall, train_f1, batch = 0, 0, 0, 0, 0
    class_precision = np.zeros(102) # we calculate precision per attribute

    # Loop through data loader and data batches
    for inputs, labels in dataloader:

        # Update batch number
        batch += 1
        print(f"Train batch: {batch}")

        # Send data to target device
        inputs, labels = inputs.to(device), labels.float().to(device)

        # 1. Forward pass
        y_preds = model(inputs)  # y_preds will be in the form of probabilities (after softmax)

        # Filter out attributes with single vote
        mask = labels != -1  # -1 is a placeholder for attributes with a single vote --> ambiguous
        labels_masked = labels[mask] # we get a flattened list that filters out attributes with score of -1
        y_preds_masked = y_preds[mask] # also get only the predictions of interest

        # Calculate the loss
        loss = loss_fn(y_preds_masked, labels_masked)

        # Add batch loss to total train_loss
        train_loss += loss.item()  # * inputs.size(0) # Optionally, we can normalize by multiplying by batch_size

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Threshold the output probabilities
        final_output_masked = torch.round(y_preds_masked).to(device)
        final_output = torch.round(y_preds).to(device)

        # Calculate the accuracy
        batch_acc = accuracy_fn(final_output_masked, labels_masked)
        train_acc += batch_acc # Add this batch accuracy to total train_acc

        # Convert to NumPy
        labels_np = labels.cpu().detach().numpy()
        final_output_np = final_output.cpu().detach().numpy()
        labels_np_masked = labels_masked.cpu().detach().numpy()
        final_output_np_masked = final_output_masked.cpu().detach().numpy()

        # Precision
        num_attributes = labels_np.shape[1]

        for attr_index in range(num_attributes):
            # Extract labels and predictions for the current attribute
            current_labels = labels_np[:, attr_index]
            current_predictions = final_output_np[:, attr_index]

            # Ignore instances marked with -1
            valid_indices = current_labels != -1
            valid_labels = current_labels[valid_indices]
            valid_predictions = current_predictions[valid_indices]

            # Calculate precision for the current attribute, excluding -1 labels
            if np.any(valid_indices):  # Check if there's at least one valid label
                class_precision[attr_index] += precision_score(valid_labels, valid_predictions, zero_division=1)
            else:
                class_precision[attr_index] = np.nan  # Handle case with no valid data


        # OLD #  class_precision += precision_score(labels_np, final_output_np, average=None, zero_division=1)

        # Recall
        train_recall += recall_score(labels_np_masked, final_output_np_masked, zero_division=1)

        # F1
        train_f1 += f1_score(labels_np_masked, final_output_np_masked, zero_division=1)

        # Print out info. every 50 batches
        if batch % 50 == 0:
            print(
                f"[INFO] Batch: {batch} | Batch average loss: {train_loss / batch} | Batch average accuracy: {train_acc / batch} | Batch average precision: {class_precision / batch}")

    # Adjust metrics to get average loss and average accuracy per batch
    class_precision /= len(dataloader)
    average_loss = train_loss / len(dataloader)
    average_acc = (train_acc / len(dataloader)).item()
    average_recall = train_recall / len(dataloader)
    average_f1 = train_f1 / len(dataloader)
    average_precision = np.mean(class_precision)

    class_precision = [round(value, 2) for value in class_precision]

    # Return a dictionary with the averages + list of precision per attribute
    return ({"train_loss": round(average_loss, 4),
            "train_acc": round(average_acc, 2),
            "train_average_precision": round(average_precision, 2),
            "train_average_recall": round(average_recall, 2),
            "train_average_f1": round(average_f1, 2)},
            class_precision)

def train_model(model, train_loader, loss_fn, optimizer, accuracy_fn, device=device,num_epochs=1):
    # Start timer
    time_start = time.time()

    # Create results dictionary
    metric_names = ["train_loss",
                    "train_acc",
                    "train_average_precision",
                    "train_average_recall",
                    "train_average_f1"]

    results = {metric: [] for metric in metric_names}

    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch nº {epoch}")
        print("-" * 10)

        # Training step
        metrics, class_precision = train_step(model=model,
                                   dataloader=train_loader,
                                   loss_fn=loss_fn,
                                   optimizer=optimizer,
                                   device=device)

        # Update the results dictionary
        for metric_name, metric_value in metrics.items():
            results[metric_name].append(metric_value)
            print(f"{metric_name}: {metric_value}", end=" | ")
        print()

    # Stop the timer and calculate elapsed time
    time_end = time.time()
    time_elapsed = time_end - time_start

    # Print out time elapsed
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return results


def test_model(model, test_loader, loss_fn, accuracy_fn, num_classes = 102, device=device):
    # Start timer
    time_start = time.time()

    # Put model in eval mode
    model.eval()

    # Set up the metrics and batch number at zero
    test_loss, test_acc, test_recall, test_f1, batch = 0, 0, 0, 0, 0

    test_precision = np.zeros(102) # initialize precision per attribute

    # Turn on inference context manager
    with torch.inference_mode():

        # Loop through DataLoader batches
        for inputs, labels in test_loader:

            # Update batch number
            batch += 1
            print(f"Test batch: {batch}")

            # Send data to target device
            inputs, labels = inputs.to(device), labels.float().to(device)

            # 1. Forward pass
            y_preds = model(inputs)  # test_preds will be in the form of probabilities (after softmax)

            # Filter out attributes with single vote
            mask = labels != -1 # -1 indicates attributes with a single vote --> ambiguous
            labels_masked = labels[mask] # obtain a flattened list of the labels of interest, filtering out ambiguous attributes
            y_preds_masked = y_preds[mask] # obtain a flattened list of the predictions of interest

            # Loss
            # Calculate batch loss
            loss = loss_fn(y_preds_masked, labels_masked)
            #  Add batch loss to total train_loss
            test_loss += loss.item()  # * inputs.size(0) # Normalize by multiplying by batch_size

            # Threshold the output probabilities
            final_output_masked = torch.round(y_preds_masked).to(device)
            final_output = torch.round(y_preds).to(device)

            # Calculate the accuracy
            batch_acc = accuracy_fn(final_output_masked, labels_masked)
            test_acc += batch_acc  # Add this batch accuracy to total train_acc

            # Convert to numpy
            labels_np = labels.cpu().detach().numpy()
            final_output_np = final_output.cpu().detach().numpy()
            labels_np_masked = labels_masked.cpu().detach().numpy()
            final_output_np_masked = final_output_masked.cpu().detach().numpy()

            # Precision
            num_attributes = labels_np.shape[1]

            for attr_index in range(num_attributes):
                # Extract labels and predictions for the current attribute
                current_labels = labels_np[:, attr_index]
                current_predictions = final_output_np[:, attr_index]

                # Ignore instances marked with -1
                valid_indices = current_labels != -1
                valid_labels = current_labels[valid_indices]
                valid_predictions = current_predictions[valid_indices]

                # Calculate precision for the current attribute, excluding -1 labels
                if np.any(valid_indices):  # Check if there's at least one valid label
                    test_precision[attr_index] += precision_score(valid_labels, valid_predictions, zero_division=1)
                else:
                    test_precision[attr_index] = np.nan  # Handle case with no valid data

            # OLD #  class_precision += precision_score(labels_np, final_output_np, average=None, zero_division=1)

            # Recall
            test_recall += recall_score(labels_np_masked, final_output_np_masked, zero_division=1)

            # F1
            test_f1 += f1_score(labels_np_masked, final_output_np_masked, zero_division=1)

            # # Printing out
            # if batch % 3 == 0:
            #     print(
            #         f"[INFO] Batch: {batch} | Batch average loss: {test_loss / batch} | Batch average acc: {test_acc / batch}")

        # Adjust metrics to get average
        test_precision /= len(test_loader)  # class precision: shape (102,)
        average_loss = test_loss / len(test_loader)
        average_acc = (test_acc / len(test_loader)).item()
        average_precision = np.mean(test_precision)
        average_recall = test_recall / len(test_loader)
        average_f1 = test_f1 / len(test_loader)

        # Stop the timer and calculate elapsed time
        time_end = time.time()
        time_elapsed = time_end - time_start

        # Print out time elapsed
        print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return ({"model_loss": round(average_loss, 4),
            "model_acc": round(average_acc, 2),
            "average_precision": round(average_precision, 2),
            "average_recall": round(average_recall, 2),
            "average_f1": round(average_f1, 2)},
            test_precision)
