import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.cuda import empty_cache

torch.cuda.empty_cache()


# Evaluation function to calculate loss and accuracy
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = (total_correct / total_samples) * 100
    return avg_loss, accuracy


# Training Function
def train_model(model, train_loader, val_loader, device, criterion, optimizer, epochs):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            train_correct += (predictions == targets).sum().item()
            train_samples += targets.size(0)

        # Calculate training loss and accuracy
        train_loss /= len(train_loader)
        train_accuracy = (train_correct / train_samples) * 100

        # Validation
        val_loss, val_accuracy = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        # Store in history
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

    return history


# Inference function to evaluate the test dataset and print classification metrics.
def test_model(model, test_loader, device, criterion):
    test_loss, test_accuracy = evaluate(model, test_loader, device, criterion)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Further metrics, such as precision, recall, f1-score can be obtained by making predictions on all data
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    result = dict()
    precision = precision_score(all_targets, all_predictions, average="weighted")
    recall = recall_score(all_targets, all_predictions, average="weighted")
    f1 = f1_score(all_targets, all_predictions, average="weighted")

    # Store in history
    result["precision"] = precision
    result["recall"] = recall
    result["f1"] = f1
    result["test_Loss"] = test_loss
    result["test_accuracy"] = test_accuracy

    print(
        f"Test Precision: {precision:.2f}, Test Recall: {recall:.2f}, Test F1 Score: {f1:.2f}"
    )

    return result
