cfg = {
    "normalization": {
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
    },
    "validation_set": {
        "validation_split": 0.1,
    },
    "batch_size": {
        "train": 32,
        "validation": 32,
        "test": 32,
    },
    "training": {
        "epochs": 40,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
    },
    "model": {
        "image_size": 224,
        "patch_size": 8,
        "num_classes": 2,
        "dim": 64,
        "depth": 4,
        "heads": 4,
        "mlp_dim": 256,
        "pool": 'cls',
        "channels": 3,
        "dim_head": 64,
        "dropout": 0.1,
        "emb_dropout": 0.1
        
    }
}
    