from LDAGM import LDAGM
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
)
import itertools
from datetime import datetime


class MyDataset(Dataset):
    def get_data(self, A, A_encoders, ij):
        data = []
        
        for item in ij:
            feature = np.array(
                [
                    np.hstack((A[0][item[0]], A_encoders[0][item[0]])),
                    np.hstack((A[0][item[1]], A_encoders[0][item[1]])),
                ]
            )
            for dim in range(1, A.shape[0]):
                feature = np.concatenate(
                    (
                        feature,
                        np.array(
                            [
                                np.hstack((A[dim][item[0]], A_encoders[dim][item[0]])),
                                np.hstack((A[dim][item[1]], A_encoders[dim][item[1]])),
                            ]
                        ),
                    )
                )
            data.append(feature)
        return np.array(data)

    def __init__(self, network_num, fold, positive_ij, negative_ij, mode, dataset, file_name):
        super().__init__()
        Ai = []
        A_encoders = []
        for i in range(network_num):
            A = np.load(
                "./our_dataset/"
                + dataset
                + "/A/A_"
                + str(fold + 1)
                + "_"
                + str(i + 1)
                + ".npy"
            )
            encoder_folder_name = f'./our_dataset/{dataset}/A_encoder/{file_name}'
            A_encoder = np.load(
                encoder_folder_name
                + '/A_'
                + str(fold + 1)
                + "_"
                + str(i + 1)
                + ".npy"
            )
            Ai.append(A)
            A_encoders.append(A_encoder)
        Ai = np.array(Ai)
        A_encoders = np.array(A_encoders)
        positive_data = torch.Tensor(self.get_data(Ai, A_encoders, positive_ij))
        negative_data = torch.Tensor(self.get_data(Ai, A_encoders, negative_ij))

        data = torch.cat((positive_data, negative_data)).transpose(2, 1)
        self.data = data
        self.target = torch.Tensor(
            [1] * positive_data.shape[0] + [0] * negative_data.shape[0]
        )
        print(f"{dataset} {mode} the data is loaded and the shape is {data.shape}")

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


def train(
    dataset,
    hidden_dimension,
    hiddenLayer_num,
    drop_rate,
    use_aggregate,
    batch_size,
    epochs,
    device,
    learn_rate,
    weight_decay,
):
    dataloader = DataLoader(
        dataset, batch_size, shuffle=True, drop_last=False, pin_memory=True
    )
    feature_num = dataset.data.shape[1]
    input_dimension = dataset.data.shape[2]
    model = LDAGM(
        input_dimension,
        hidden_dimension,
        feature_num,
        hiddenLayer_num,
        drop_rate=drop_rate,
        use_aggregate=use_aggregate,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learn_rate, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    epoch = 0
    loss_record = []
    while epoch < epochs:
        model.train()
        for x, y in dataloader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pre = model(x)
            loss = loss_fn(pre, y)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().cpu().item())
        epoch += 1
        print(f"In round {epoch}, the loss is: {loss.detach().cpu().item()}")
    print("End of training")
    return loss_record, model


def test(model, dataset, batch_size, device):
    dataloader = DataLoader(
        dataset, batch_size, shuffle=False, drop_last=False, pin_memory=True
    )
    preds = []
    labels = []
    model.eval()
    print(f"Start testing on {dataset}")
    for x, y in dataloader:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
        labels.append(y.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return labels, preds


if __name__ == "__main__":
    # preset parameter
    fold = 0
    # dataset1
    # dataset = "dataset1"
    # network_num = 4
    # dataset2
    dataset = "dataset2"
    network_num = 4
    # dataset3
    # dataset = "dataset3"
    # network_num = 2
    
    # Global file name for encoder folder (corresponds to A_embedding_dimension_network_num_epochs)
    list_file_name = ["A_16_256_4_100_do"]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Read the indexes of positive and negative samples for the training and test sets
    positive5foldsidx = np.load(
        "./our_dataset/" + dataset + "/index/positive5foldsidx.npy",
        allow_pickle=True,
    )
    negative5foldsidx = np.load(
        "./our_dataset/" + dataset + "/index/negative5foldsidx.npy",
        allow_pickle=True,
    )
    positive_ij = np.load("./our_dataset/" + dataset + "/index/positive_ij.npy")
    negative_ij = np.load("./our_dataset/" + dataset + "/index/negative_ij.npy")
    train_positive_ij = positive_ij[positive5foldsidx[fold]["train"]]
    train_negative_ij = negative_ij[negative5foldsidx[fold]["train"]]
    test_positive_ij = positive_ij[positive5foldsidx[fold]["test"]]
    test_negative_ij = negative_ij[negative5foldsidx[fold]["test"]]

    for file_name_object in itertools.product(list_file_name):
        file_name = file_name_object[0]
        print(f"Start testing on {dataset} with {file_name}")

        # Generate dataset objects based on indexes
        train_dataset = MyDataset(
            network_num, fold, train_positive_ij, train_negative_ij, "训练", dataset, file_name
        )
        test_dataset = MyDataset(
            network_num, fold, test_positive_ij, test_negative_ij, "测试", dataset, file_name
        )
        # Setting Model Parameters
        # dataset1
        hidden_dimension = 40
        hiddenLayer_num = 2
        drop_rate = 0.1
        batch_size = 32
        epochs = [1,2,3,4,5]
        use_aggregate = True
        learn_rate = 1e-2
        weight_decay = 1e-4

        # dataset2
        # hidden_dimension = 40
        # hiddenLayer_num = 2
        # drop_rate = 0.1
        # batch_size = 32
        # epochs = 10
        # use_aggregate = True
        # learn_rate = 1e-4
        # weight_decay = 1e-3

        # dataset3
        # hidden_dimension = 5
        # hiddenLayer_num = 4
        # drop_rate = 0.1
        # batch_size = 32
        # epochs = 5
        # use_aggregate = True
        # learn_rate = 1e-3
        # weight_decay = 1e-5
        test_times = 10
        # train and test 10 times
        print(f"Start training on {dataset} with file {file_name}")
        for epoch in epochs:
            results = []
            for i in range(test_times):
                print(f"--- Running iteration {i+1}/10 ---")
                loss_record, model = train(
                    train_dataset,
                    hidden_dimension,
                    hiddenLayer_num,
                    drop_rate,
                    use_aggregate,
                    batch_size,
                    epoch,
                    device,
                    learn_rate,
                    weight_decay,
                )
                # test
                test_target, pre_target = test(model, test_dataset, batch_size, device)
                # np.save("./result/" + dataset + "/label", test_target)
                # np.save("./result/" + dataset + "/predict", pre_target)
                test_target = np.array(test_target)
                # Getting a specific score
                AUC = roc_auc_score(test_target, pre_target)
                precision, recall, _ = precision_recall_curve(test_target, pre_target)

                fpr, tpr, thresholds = roc_curve(test_target, pre_target)
                AUPR = auc(recall, precision)
                preds = np.array([1 if p > 0.5 else 0 for p in pre_target])
                # preds = np.array([1 if p > 0.5 else 0 for p in pre_target])
                MCC = matthews_corrcoef(test_target, preds)
                ACC = accuracy_score(test_target, preds)
                P = precision_score(test_target, preds)
                R = recall_score(test_target, preds)
                F1 = f1_score(test_target, preds)
                print(f"Iteration {i+1} scores: {AUC}, {AUPR}, {MCC}, {ACC}, {P}, {R}, {F1}")
                results.append(
                    {
                        "hidden_layer": hiddenLayer_num,
                        "file_name": file_name,
                        "epoch": epoch,
                        "iteration": i+1,
                        "AUC": AUC,
                        "AUPR": AUPR,
                        "MCC": MCC,
                        "ACC": ACC,
                        "Precision": P,
                        "Recall": R,
                        "F1": F1,
                        "network_num": network_num,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

            # Save comprehensive results
            if results:
                results_df = pd.DataFrame(results)
                
                # Create result directory if it doesn't exist
                import os
                result_dir = f"our_dataset/{dataset}/result"
                os.makedirs(result_dir, exist_ok=True)
                
                # Check if results file exists and append if it does
                results_file_path = f"our_dataset/{dataset}/result/gate_results.csv"
                try:
                    existing_results = pd.read_csv(results_file_path)
                    # Append new results to existing ones
                    combined_results = pd.concat([existing_results, results_df], ignore_index=True)
                    combined_results.to_csv(results_file_path, index=False)
                    print(f"\nAppended {len(results_df)} new results to existing file")
                    print(f"Result for epoch {epoch} - network_num {network_num} - hidden layer {hiddenLayer_num}")
                    print(f"AUC mean: {results_df['AUC'].mean()}")
                    print(f"AUC values: {results_df['AUC'].values}")
                    print(f"AUPR mean: {results_df['AUPR'].mean()}")
                    print(f"AUPR values: {results_df['AUPR'].values}")
                except FileNotFoundError:
                    # File doesn't exist, create new one
                    results_df.to_csv(results_file_path, index=False)
                    print(f"\nCreated new results file with {len(results_df)} results")
                    combined_results = results_df
            
                
                print(f"\nResults saved to {results_file_path}")
            else:
                print("No results to save.")
