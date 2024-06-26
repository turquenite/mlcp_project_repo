import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

default_included_features = [
    "bandwidth_0",
    "centroid_0",
    "centroid_1",
    "centroid_2",
    "centroid_3",
    "centroid_4",
    "centroid_5",
    "centroid_6",
    "energy_0",
    "flatness_0",
    "flux_0",
    "power_0",
    "yin_0",
    "zcr_0"
]

important_words = [
    "Staubsauger",
    "Radio",
    "Alarm",
    "Ofen",
    "Lüftung",
    "Licht",
    "Heizung",
    "Fernseher",
    "an",
    "aus"
]

mel_features = [f"melspect_{i}" for i in range(64)]

class MLPC_Dataset(Dataset):
    def __init__(self, included_features=None):
        numpy_dataset = np.load("data/development.npy")

        development_metadata = pd.read_csv("data/development.csv")
        idx_from_feature_name = pd.read_csv("data/idx_to_feature_name.csv")

        if included_features==None:
            included_features = default_included_features
        elif included_features=="all":
            #just use all features
            included_features = list(idx_from_feature_name["feature_name"])
        elif included_features=="mel":
            #just use all features
            included_features = mel_features

        included_feature_idx = idx_from_feature_name[idx_from_feature_name["feature_name"].isin(included_features)]
        included_idx = included_feature_idx["index"]

        self.feature_names = list(included_feature_idx["feature_name"])

        filtered_dataset = numpy_dataset[:, included_idx]

        frame_iter = [f"Frame {i}" for i in range(1, 45)]

        index = pd.MultiIndex.from_product([included_feature_idx["feature_name"], frame_iter])

        dataset_with_index = pd.DataFrame(filtered_dataset.reshape((filtered_dataset.shape[0], -1)), index=development_metadata["id"], columns=index)
        
        dataset = pd.concat([development_metadata, dataset_with_index], axis=1)

        dataset.loc[~dataset["word"].isin(important_words), "word"] = "other"

        # Replace categorical attributes with numerical Labels
        dataset["word"] = dataset["word"].astype("category")
        dataset["word_num_label"] = dataset["word"].cat.codes

        self.dataset = dataset

    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset.iloc[index]
    
    def stack(self, batch):
        train_params = list()
        insight_params = list()
        labels = list()

        frame_iter = [f"Frame {i}" for i in range(1, 45)]

        for sample in batch:
            labels.append(torch.tensor(sample["word_num_label"], dtype=torch.float32))
            train_params.append(torch.tensor([[sample[(att, i)] for i in frame_iter] for att in self.feature_names], dtype=torch.float32).flatten())
            insight_params.append(sample)

        return torch.stack(train_params), insight_params, torch.stack(labels)
    
    def get_data_loaders(self, batch_size: int = 32, shuffle: bool = True, seed: int = 12, split = [0.8, 0.2]):
        train_set, eval_set = torch.utils.data.random_split(self, split, torch.Generator().manual_seed(seed))
        return (torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, collate_fn=self.stack), torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=shuffle, collate_fn=self.stack))
    
    def get_dataset(self, seed: int = 12, test_size: int = 0.2):
        frame_iter = [f"Frame {i}" for i in range(1, 45)]
        columns_to_keep = [(att, i) for i in frame_iter for att in self.feature_names]
        X = self.dataset[columns_to_keep].values
        y = self.dataset['word_num_label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        return X_train, X_test, y_train, y_test
