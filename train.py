import torch
from new_dataset import CommonVoice
from model import Model
from torch import nn, optim
from torch.utils.data import DataLoader
import torchaudio
import os
from tqdm import tqdm

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
torch.save(model.state_dict(), os.path.join(save_location, "dummy.pth"))
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    for id, (support_samples, support_labels, query_samples, query_labels) in tqdm(
        enumerate(train_dl), desc="Training..."
    ):
        support_samples, support_labels, query_samples, query_labels = (
            support_samples.squeeze(0).to(device),
            support_labels.squeeze(0).to(device),
            query_samples.squeeze(0).to(device),
            query_labels.squeeze(0).to(device),
        )
        optimizer.zero_grad()
        output = model(support_samples, support_labels, query_samples)
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
        ) in tqdm(enumerate(val_dl), desc="Validating..."):
            support_samples, support_labels, query_samples, query_labels = (
                support_samples.squeeze(0).to(device),
                support_labels.squeeze(0).to(device),
                query_samples.squeeze(0).to(device),
                query_labels.squeeze(0).to(device),
            )

            correct = (
                (
                    torch.max(
                        model(
                            support_samples,
                            support_labels,
                            query_samples,
                        )
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
