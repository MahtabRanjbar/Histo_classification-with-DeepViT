from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(data_dir, batch_size, train=False):
    if train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2
                ),
                transforms.Resize(256),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    all_data = datasets.ImageFolder(data_dir, transform=transform)
    data_lengths = [int(len(all_data) * x) for x in (0.7, 0.15, 0.15)]

    if train:
        train_data, val_data, _ = random_split(all_data, data_lengths)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return train_loader
    else:
        _, val_data, test_data = random_split(all_data, data_lengths)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return val_loader, test_loader
