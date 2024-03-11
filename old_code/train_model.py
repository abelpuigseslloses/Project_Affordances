
import time
from tqdm.auto import tqdm
import torch
from utils import accuracy_fn
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
    train_loss, train_acc, class_precision, class_recall, class_f1, batch = 0, 0, 0, 0, 0, 0

    # Loop through data loader and data batches
    for inputs, labels in dataloader:

        # Update batch number
        batch += 1
        print(f"Train batch: {batch}")

        # Send data to target device
        inputs, labels = inputs.to(device), labels.float().to(device)

        # 1. Forward pass
        y_preds = model(inputs)  # y_preds will be in the form of probabilities (after softmax)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_preds, labels)

        # Add batch loss to total train_loss
        train_loss += loss.item()  # * inputs.size(0) # Normalize by multiplying by batch_size

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Threshold the output probabilities
        final_output = torch.round(y_preds).to(device)

        # Calculate the accuracy
        batch_acc = accuracy_fn(final_output, labels)
        train_acc += batch_acc # Add this batch accuracy to total train_acc

        # Precision, recall, f1
        labels_np = labels.cpu().detach().numpy()
        final_output_np = final_output.cpu().detach().numpy()
        class_precision += precision_score(labels_np, final_output_np, average=None, zero_division=1)
        class_recall += recall_score(labels_np, final_output_np, average=None, zero_division=1)
        class_f1 += f1_score(labels_np, final_output_np, average=None, zero_division=1)

        # Print out info. every 50 batches
        if batch % 50 == 0:
            print(
                f"[INFO] Batch: {batch} | Batch average loss: {train_loss / batch} | Batch average accuracy: {train_acc / batch} | Batch average precision: {class_precision / batch}")

    # Adjust metrics to get average loss and average accuracy per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    class_precision /= len(dataloader)
    class_recall /= len(dataloader)
    class_f1 /= len(dataloader)

    average_precision = np.mean(class_precision)
    average_recall = np.mean(class_recall)
    average_f1 = np.mean(class_f1)

    return train_loss, train_acc, class_precision, class_recall, class_f1, average_precision, average_recall, average_f1



def train_model(model, train_loader, loss_fn, optimizer, accuracy_fn, device=device,num_epochs=1):
    # Start timer
    time_start = time.time()

    # Create results dictionary
    metric_names = ["train_loss", "train_acc",
        "class_precision", "class_recall", "class_f1",
        "average_precision", "average_recall", "average_f1"]

    results = {metric: [] for metric in metric_names}

    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch nº {epoch}")
        print("-" * 10)

        # Training step
        metrics = train_step(model=model,
                             dataloader=train_loader,
                             loss_fn=loss_fn,
                             optimizer=optimizer,
                             device=device)

        # Update the results dictionary
        for metric_name, metric_value in zip(metric_names, metrics):
            results[metric_name].append(metric_value)

            if metric_name in ["train_loss", "train_acc", "average_precision"]:
                print(f"{metric_name}: {metric_value:.4f}", end=" | ")
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
    test_loss, test_acc, test_precision, test_recall, test_f1, batch = 0, 0, 0, 0, 0, 0

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
            test_preds = model(inputs)  # test_preds will be in the form of probabilities (after softmax)

            # Loss
            # Calculate batch loss
            loss = loss_fn(test_preds, labels)
            #  Add batch loss to total train_loss
            test_loss += loss.item()  # * inputs.size(0) # Normalize by multiplying by batch_size

            # Accuracy
            test_final_output = torch.round(test_preds).to(device)
            acc = accuracy_fn(test_final_output, labels)
            test_acc += acc.item()

            # Precision, recall, f1
            labels_np = labels.cpu().detach().numpy()
            test_final_output_np = test_final_output.cpu().detach().numpy()
            test_precision += precision_score(labels_np, test_final_output_np, average=None, zero_division=1)
            test_recall += recall_score(labels_np, test_final_output_np, average=None, zero_division=1)
            test_f1 += f1_score(labels_np, test_final_output_np, average=None, zero_division=1)

            # Printing out
            if batch % 10 == 0:
                print(
                    f"[INFO] Batch: {batch} | Batch average loss: {test_loss / batch} | Batch average acc: {test_acc / batch}")

        # Adjust metrics to get average loss and accuracy per batch
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        test_precision /= len(test_loader)
        test_recall /= len(test_loader)
        test_f1 /= len(test_loader)

        average_precision = np.mean(test_precision)
        average_recall = np.mean(test_recall)
        average_f1 = np.mean(test_f1)

        # Stop the timer and calculate elapsed time
        time_end = time.time()
        time_elapsed = time_end - time_start

        # Print out time elapsed
        print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return {"model_loss": test_loss,
            "model_acc": test_acc,
            "average_precision": average_precision,
            "average_recall": average_recall,
            "average_f1": average_f1,
            "precision_class": test_precision}
