import torch
from new_dataset import CommonVoice
from model import Model
from torch import nn, optim
from torch.utils.data import DataLoader
import torchaudio
import os

epochs = 100
lr = 0.0001
train_csv = "./train.csv"
val_csv = "./val.csv"
n_shot = 5
n_ways = 1
n_query = 1
show_every = 500
save_location = "./checkpoints"

bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
freq = bundle.sample_rate
train_ds = CommonVoice(
    train_csv, n_shot=n_shot, n_ways=n_ways, n_query=n_query, bundle_freq_rate=freq
)
val_ds = CommonVoice(
    val_csv, n_shot=n_shot, n_ways=n_ways, n_query=n_query, bundle_freq_rate=freq
)

train_dl = DataLoader(train_ds, num_workers=12, pin_memory=True)
val_dl = DataLoader(val_ds, num_workers=12, pin_memory=True)

device = "cuda" if torch.cuda.is_available else "cpu"
model = Model()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    for id, (support_samples, support_labels, query_samples, query_labels) in enumerate(
        train_dl
    ):
        support_samples, support_labels, query_samples, query_labels = (
            support_samples.to(device),
            support_labels.to(device),
            query_samples.to(device),
            query_labels.to(device),
        )
        optimizer.zero_grad()
        output = model(support_samples, query_samples, support_labels)
        loss = loss_fn(output, query_labels)
        loss.backward()
        optimizer.step()
        if id % show_every == 0:
            print(f"Epoch: {epoch}\t loss: {loss.item()}")

    model.eval()
    total_predictions = 0
    correct_predictions = 0
    with torch.no_grad():
        for _, (
            support_samples,
            support_labels,
            query_samples,
            query_labels,
        ) in enumerate(val_dl):
            support_samples, support_labels, query_samples, query_labels = (
                support_samples.to(device),
                support_labels.to(device),
                query_samples.to(device),
                query_labels.to(device),
            )

            correct = (
                (
                    torch.max(
                        model(support_samples, query_samples, support_labels)
                        .detach()
                        .data,
                        1,
                    )[1]
                    == query_labels
                )
                .sum()
                .item()
            )
            correct_predictions += correct
            total_predictions += len(query_labels)
    print()

    acc = 100 * (correct_predictions / total_predictions)
    print(f"Accuracy: {(100 * correct_predictions / total_predictions):.2f}%")
    torch.save(
        model.state_dict(), os.path.join(save_location, f"epoch_{epoch}_acc_{acc:.2f}")
    )
