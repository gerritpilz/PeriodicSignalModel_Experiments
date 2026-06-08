config = {
    # training
    "n_epochs": 4,
    "eval_iter": 5,
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
}