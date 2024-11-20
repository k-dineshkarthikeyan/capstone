import torch
import random
import torchaudio
from torch.utils.data import Dataset
import pandas as pd


class CommonVoice(Dataset):
    def __init__(self, csv_path, n_shot, n_ways, n_query, bundle_freq_rate) -> None:
        super().__init__()
        self.bundle_freq_rate = bundle_freq_rate
        self.data = pd.read_csv(csv_path)
        self.n_shot = n_shot
        self.n_ways = n_ways
        self.n_query = n_query
        self.audio_len = self.data.resampled_shapes.min()
        self.random_sample = torch.randn(1, self.audio_len)
        self.speaker_sample_len = {}
        self.speaker_indices = {}
        # for index, row in self.data.iterrows():
        #     if row[0] not in self.speaker_sample_len:
        #         self.speaker_sample_len[row[0]] = 1
        #     else:
        #         self.speaker_sample_len[row[0]] += 1

        #     if row[0] not in self.speaker_indices:
        #         self.speaker_indices[row[0]] = [index]
        #     else:
        #         self.speaker_indices[row[0]].append(index)
        for index, speaker in enumerate(self.data.speaker_id):
            if speaker not in self.speaker_sample_len:
                self.speaker_sample_len[speaker] = 1
            else:
                self.speaker_sample_len[speaker] += 1
            if speaker not in self.speaker_indices:
                self.speaker_indices[speaker] = [index]
            else:
                self.speaker_indices[speaker].append(index)
        self.unique_speakers = self.data.speaker_id.unique().tolist()
        self.speakers_to_labels = {
            k: v for v, k in enumerate(list(self.unique_speakers))
        }
        for key in self.unique_speakers:
            if len(self.speaker_indices[key]) < self.n_shot + self.n_query:
                del self.speaker_indices[key]
                del self.speaker_sample_len[key]
                del self.speakers_to_labels[key]
                self.unique_speakers.remove(key)

    def __len__(self) -> int:
        return len(self.data)

    def load_audio(self, index: int) -> torch.Tensor:
        audio_path = self.data.audio_path[index]
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=self.bundle_freq_rate
        )
        waveform = waveform[:, : self.audio_len]
        return waveform

    def __getitem__(self, _):
        speakers = random.sample(self.unique_speakers, self.n_ways)
        unique_speakers = list({i for i in speakers})
        unique_labels = [unique_speakers.index(i) for i in speakers]
        query_labels = torch.tensor([
            i for i in unique_labels for _ in range(self.n_query)
        ])
        support_labels = torch.tensor([
            i for i in unique_labels for _ in range(self.n_shot)
        ])
        audio_samples = torch.cat([
            torch.cat([
                self.load_audio(i).unsqueeze(0)
                for i in random.sample(
                    self.speaker_indices[speaker], self.n_shot + self.n_query
                )
            ])
            for speaker in speakers
        ]).reshape(self.n_ways, self.n_shot + self.n_query, *self.random_sample.shape)
        support_samples = audio_samples[:, : self.n_shot].reshape((
            -1,
            *self.random_sample.shape,
        ))
        query_samples = audio_samples[:, self.n_shot :].reshape((
            -1,
            *self.random_sample.shape,
        ))
        return support_samples, support_labels, query_samples, query_labels
