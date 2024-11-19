import asyncio
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
from torch.utils.data import DataLoader
from torchaudio.functional import resample
import torchaudio

login(token="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")

dataset = load_dataset("mozilla-foundation/common_voice_17_0", "en", num_proc=20)
print("downloaded dataset")


async def write_csv(split):
    dl = DataLoader(dataset[split])
    df = pd.DataFrame(data={"speaker_id": [], "audio_path": [], "tensor_shape": []})

    for batch in dl:
        audio = batch["audio"]["array"]
        audio_sample_rate = torchaudio.info(batch["path"][0]).sample_rate
        audio = resample(audio, audio_sample_rate, 16000)
        shape = audio.shape
        df.loc[len(df)] = [
            batch["client_id"][0],
            batch["path"][0],
            # batch["audio"]["array"].shape,
            shape,
        ]

    df.to_csv(f"{split}.csv", index=False)


async def main():
    await asyncio.gather(write_csv("train"), write_csv("test"), write_csv("validation"))


if __name__ == "__main__":
    asyncio.run(main())
