
base_config = {
    # training
    "n_epochs": 4,
    "lr": 1e-4,
    "lr_min": 2e-5,
    "scheduler_steps": 1000,

    # hyperparameters
    "n_channels": 30,
    "seq_len": 128,
    "pred_len": 32,
    "batch_size": 16,
    "d_embd": 128,
    "dropout": 0.2,

    # model
    "n_timeBlocks": 8,
    "k_periods": 6,

    # gaussian bandpass filter
    "sigma": 0.5,

    # weighting of instant amplitudes in aggregation
    "alpha": 0.1
}


sweep_config = {
    "method": "random",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {

        "lr": {
            "values": [5e-5, 1e-4, 2e-4]
        },

        "d_embd": {
            "values": [64, 128, 256]
        },

        "dropout": {
            "values": [0.1, 0.2, 0.3]
        },

        "n_timeBlocks": {
            "values": [6, 8, 12]
        },

        "k_periods": {
            "values": [4, 6, 8]
        },

        "sigma": {
            "values": [0.3, 0.5, 0.7]
        },

        "alpha": {
            "values": [0.05, 0.1, 0.15]
        },


        "n_epochs": {"value": 2},
        "lr_min": {"value": 2e-5},
        "scheduler_steps": {"value": 1000},

        "n_channels": {"value": 30},
        "seq_len": {"value": 128},
        "pred_len": {"value": 32},
        "batch_size": {"value": 16}
    }
}
