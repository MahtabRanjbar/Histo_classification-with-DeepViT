import torch
import torch.nn as nn
import torch.optim as optim
from train_and_evaluate import test_model, train_model

from config import cfg
from data_loader import get_data_loaders
from model import DeepViT
from utils import count_parameters, plot_accuracy_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize data loaders
data_dir = "./working/dataset"
train_loader = get_data_loaders(data_dir, cfg["batch_size"]["train"], train=True)
val_loader, test_loader = get_data_loaders(
    data_dir, cfg["batch_size"]["validation"], train=False
)

# Initialize model
model = DeepViT(
    image_size=cfg["model"]["image_size"],
    patch_size=cfg["model"]["patch_size"],
    num_classes=cfg["model"]["num_classes"],
    dim=cfg["model"]["dim"],
    depth=cfg["model"]["depth"],
    heads=cfg["model"]["heads"],
    mlp_dim=cfg["model"]["mlp_dim"],
    pool=cfg["model"]["pool"],
    channels=cfg["model"]["channels"],
    dim_head=cfg["model"]["dim_head"],
    dropout=cfg["model"]["dropout"],
    emb_dropout=cfg["model"]["emb_dropout"],
)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"],
)

# Training and testing
history = train_model(
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    optimizer,
    cfg["training"]["epochs"],
)
result = test_model(model, test_loader, device, criterion)

# savinf teaining history
plot_accuracy_loss(
    history["train_loss"],
    history["train_accuracy"],
    history["val_loss"],
    history["val_accuracy"],
)
# save number of parameters in each layer
count_parameters(model)
