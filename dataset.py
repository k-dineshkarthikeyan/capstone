import torch
import random
import pandas as pd
from torch.utils.data import Sampler
import torchaudio


class TaskSampler(Sampler):
    def __init__(self, csv_file, n_ways, n_shot, n_query):
        super().__init__(data_source=None)
        self.n_ways = n_ways
        self.n_shot = n_shot
        self.n_query = n_query

        self.df = pd.read_csv(csv_file)
        labels = self.df.speaker_id.unique()
        self.labels = {k: v for v, k in enumerate(labels)}
        self.items_to_labels = {}
        for i in range(len(self.df)):
            if self.df.speaker_id[i] not in self.items_to_labels:
                self.items_to_labels[self.labels[self.df.speaker_id[i]]] = [i]

            else:
                self.items_to_labels[self.labels[self.df.speaker_id[i]]].append(i)

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        yield torch.cat([
            torch.tensor(
                random.sample(
                    list(self.items_to_labels[label]), self.n_shot + self.n_query
                )
            )
            for label in random.sample(list(self.labels.values()), self.n_ways)
        ])

    def collate_fn(self, input_data):
        speaker_path = self.arrange_data(input_data)
        true_class_id = list([x[0] for x in speaker_path])
        all_paths = torch.cat()

    def arrange_data(self, input_data):
        return [
            (item["client_id"][0], item["audio"]["array"]) for item in input_data.keys()
        ]
