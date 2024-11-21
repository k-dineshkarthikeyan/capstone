import torch
import random
import pandas as pd
from torch.utils.data import Sampler, Dataset
import torchaudio
from torchaudio import transforms
from typing import Optional, Sized, Tuple, List


class CommonVoice(Dataset):
    def __init__(self, csv_path) -> None:
        super().__init__()
        self.csv_name = csv_path
        self.data = pd.read_csv(csv_path)
        self.audio_len = self.data.resampled_shapes.min()
        self.resampler = transforms.Resample(orig_freq=48000, new_freq=16000)
        self.speakers = self.data.speaker_id.unique().tolist()
        self.speakers_to_labels = {v: k for k, v in enumerate(self.speakers)}

    def __len__(self):
        if "train" in self.csv_name:
            return len(self.data) // 10
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        waveform, _ = torchaudio.load(sample["audio_path"])
        resampled_audio = self.resampler(waveform)
        resampled_audio = resampled_audio[:, : int(self.audio_len)]
        return resampled_audio, torch.tensor(
            self.speakers_to_labels[sample["speaker_id"]]
        )


class TaskSampler(Sampler):
    def __init__(
        self,
        n_ways: int,
        n_shot: int,
        n_query: int,
        csv_path: str,
        n_tasks: int,
        data_source: Optional[Sized] = None,
    ) -> None:
        super().__init__(data_source)
        self.csv_name = csv_path
        self.n_ways = n_ways
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.data = pd.read_csv(csv_path)
        min_len = self.data.resampled_shapes.min()
        self.some_shape = [1, min_len]
        self.speaker_id = self.data.speaker_id.unique().tolist()
        self.speaker_to_item = {}
        for i, row in self.data.iterrows():
            if row["speaker_id"] not in self.speaker_to_item:
                self.speaker_to_item[row["speaker_id"]] = [i]
            else:
                self.speaker_to_item[row["speaker_id"]].append(i)
        exclude_speakers = []
        for i in self.speaker_to_item:
            if len(self.speaker_to_item[i]) < self.n_shot + self.n_query:
                exclude_speakers.append(i)

        self.speaker_id = [i for i in self.speaker_id if i not in exclude_speakers]
        self.speaker_to_item = {
            k: v for k, v in self.speaker_to_item.items() if k not in exclude_speakers
        }

    def __len__(self):
        if "train" in self.csv_name:
            return len(self.data) // 10
        return len(self.data)

    def __iter__(self):
        for _ in range(self.n_tasks):
            yield torch.cat([
                torch.tensor(
                    random.sample(
                        self.speaker_to_item[speaker], self.n_shot + self.n_query
                    )
                )
                for speaker in random.sample(self.speaker_id, self.n_ways)
            ]).tolist()

    def collate_fn(self, inputs: List[Tuple[torch.Tensor, torch.Tensor]]):
        data = self.type_cast(inputs)
        true_class_ids = list({i[1] for i in data})
        samples = torch.cat([i[0].unsqueeze(0) for i in data]).reshape(
            self.n_ways, self.n_shot + self.n_query, *self.some_shape
        )
        labels = torch.tensor([
            torch.tensor(true_class_ids.index(i[1])) for i in data
        ]).reshape(self.n_ways, self.n_shot + self.n_query)
        support_samples = samples[:, : self.n_shot].reshape((-1, *self.some_shape))
        query_samples = samples[:, self.n_shot].reshape((-1, *self.some_shape))
        support_labels = labels[:, : self.n_shot].flatten()
        query_labels = labels[:, self.n_shot :].flatten()

        return support_samples, support_labels, query_samples, query_labels

    def type_cast(self, inputs: List[Tuple[torch.Tensor, torch.Tensor]]):
        return [(image, int(label)) for (image, label) in inputs]
