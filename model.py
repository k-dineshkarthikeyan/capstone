import torch
from torch import nn
import torchaudio


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
        self.model = bundle.get_model()
        self.bundle_sample_rate = bundle.sample_rate
        self.w = nn.Parameter(torch.tensor([1.0]))

    def first_half(self, support_tensors, query_tensors, labels):
        avgpool = nn.AdaptiveMaxPool1d((1))
        out = avgpool(support_tensors)
        support = out.view((out.size(0), -1))

        out = avgpool(query_tensors)
        query = out.view((out.size(0), -1))

        n_way = len(torch.unique(labels))
        proto = torch.cat([
            support[torch.nonzero(labels == label)].mean(0) for label in range(n_way)
        ])

        dists = torch.cdist(query, proto)
        return dists

    def forward(self, support_samples, support_labels, query_samples):
        print(query_samples.shape)
        l = [i for i in query_samples]
        print(l)
        support = torch.cat([self.model(sample)[0] for sample in support_samples])
        query = torch.cat([self.model(sample)[0] for sample in query_samples])

        d = self.first_half(support, query, support_labels)
        result = self.w * d
        return result
