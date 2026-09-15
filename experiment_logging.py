import os
import csv


def log_hyperparameters(config):
    if not os.path.exists(os.path.dirname(config.LOG_FILE)):
        os.makedirs(os.path.dirname(config.LOG_FILE))
    with open(config.LOG_FILE, 'a') as f:
        f.write("Hyperparameters:\n")
        for key, value in vars(config).items():
            if not key.startswith('__'):
                f.write(f"{key}: {value} -- ")
        f.write("\n")


def log_to_csv(config, metrics):
    if not os.path.exists(os.path.dirname(config.CSV_LOG_FILE)):
        os.makedirs(os.path.dirname(config.CSV_LOG_FILE))

    hyperparams = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    fieldnames = list(hyperparams.keys()) + list(metrics.keys())

    file_exists = os.path.isfile(config.CSV_LOG_FILE)

    with open(config.CSV_LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({**metrics, **hyperparams})
