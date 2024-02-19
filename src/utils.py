from matplotlib import pyplot as plt
from prettytable import PrettyTable


def plot_accuracy_loss(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(train_accuracies) + 1), train_accuracies, label="Training Accuracy"
    )
    plt.plot(
        range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("training-validation-history.png")
    plt.tight_layout()
    plt.show()


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
        summery = table.get_string()
    with open("model_summery.txt", "w") as f:
        f.write(summery)
        f.write("\n Total Trainable Params:")
        f.write(str(total_params))
    print(table)
    print(f"Total Trainable Params: {total_params}")
