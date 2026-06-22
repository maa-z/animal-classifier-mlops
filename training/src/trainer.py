import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models


def create_model(num_classes):

    model = models.resnet18(weights="DEFAULT")

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model


def train_model(
    model,
    train_loader,
    val_loader,
    config,
    device
):

    learning_rate = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_accuracy = 0.0
    best_model = copy.deepcopy(model)

    history = {
        "train_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):

        print(f"Epoch [{epoch+1}/{epochs}]")
        model.train()

        running_loss = 0.0
        i = 0
        for images, labels in train_loader:
            i += 1
            print(f"Training on batch {i} of size {images.size(0)} out of {len(train_loader.dataset)} samples.")

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = (
            running_loss /
            len(train_loader)
        )

        history["train_loss"].append(
            avg_train_loss
        )

        val_accuracy = evaluate_model(
            model,
            val_loader,
            device
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            best_model = copy.deepcopy(
                model
            )

    metrics = {
        "best_accuracy": best_accuracy,
        "history": history
    }

    return best_model, metrics


def evaluate_model(
    model,
    val_loader,
    device
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = correct / total

    return accuracy