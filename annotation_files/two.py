from operator import index
import pandas as pd
from time import time

train = pd.read_csv("../train.csv")
val = pd.read_csv("../validation.csv")
test = pd.read_csv("../test.csv")


def get_shape(shape: str) -> int:
    shape = shape[shape.find(",") + 1 : shape.find("]")]
    return int(shape)


def main(split, name):
    asdf = time()
    split_shapes = []
    for i in split["tensor_shape"]:
        shape = get_shape(i)
        new_shape = int(shape * 16 / 48)
        split_shapes.append(new_shape)

    split["resampled_shapes"] = split_shapes
    split.to_csv(name, index=False)
    print(f"time to perform {name} is {time() - asdf}")


main(train, "train.csv")
main(val, "val.csv")
main(test, "test.csv")

test = pd.read_csv("./test.csv")
print(type(test.resampled_shapes.min()))
print(test.resampled_shapes.min())
print(test.resampled_shapes[1])
print(type(test.resampled_shapes))
print(len(test.resampled_shapes))
# for i in test.resampled_shapes:
#     pass
