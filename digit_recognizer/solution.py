import pandas as pd
import torch
from torch import nn
import numpy as np
import random
from typing import Tuple
import os
import pathlib

from mylib import gen_batch, train_with, infer_for, calc_score
from mylib.chdir import chdir

# args
train_data_path = "data/train.csv"
test_data_path = "data/test.csv"

train_batch_size = 128
split_ratio = 0.7
nepoch = 100 # 1 epoch has really poor perf

shrink = True
if shrink:
    nepoch = 2
    train_data_path = "data/tiny-train.csv"
    test_data_path = "data/tiny-test.csv"

class ClassifierDNN(nn.Module):
    def __init__(self):
        super(ClassifierDNN, self).__init__()
        self.linear1 = nn.Linear(784, 100)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, inp):
        out = self.linear1(inp)
        out = self.relu1(out)
        out = self.linear2(out)

        # The CrossEntropyLoss will do softmax inside
        # out = self.softmax(out)
        return out

class ClassifierCNN(nn.Module):
    def __init__(self):
        super(ClassifierCNN, self).__init__()

        # in: 1 x 28 x 28
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, 5), # out: 3 x 24 x 24
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # out: 3 x 12 x 12
            nn.Conv2d(3, 10, 5), # out: 10 x 8 x 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # out: 10 x 4 x 4
        )

        self.flatten_size = 160
        # in: flatten_size
        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_size, 40),
            nn.ReLU(),
            nn.Linear(40, 10),
        )

    def forward(self, inp):
        # import pdb; pdb.set_trace()
        inp = inp.view(inp.size(0), 1, 28, 28) # a single channel
        inp = self.conv(inp)
        inp = inp.view(inp.size(0), self.flatten_size)
        return self.mlp(inp)

# ModelClass = ClassifierDNN
ModelClass = ClassifierCNN

   
def get_training_data() -> Tuple[torch.Tensor, torch.Tensor]:
    with chdir(str(pathlib.Path(__file__).parent)):
        train_data = pd.read_csv(train_data_path)
        all_x : np.ndarray = train_data.loc[:, train_data.columns != "label"].values
        all_y : np.ndarray = train_data["label"].values
    
        # convert numpy.ndarray to Tensor
        all_x = torch.Tensor(all_x)
        all_x = all_x / 255 # normalize
        all_y = torch.LongTensor(all_y)
        return all_x, all_y

def get_test_data() -> torch.Tensor:
    with chdir(str(pathlib.Path(__file__).parent)):
        test_data = pd.read_csv(test_data_path)
        test_data = torch.Tensor(test_data.values) / 255 # normalize
        return test_data

def get_example_batch(batch_size=32) -> torch.Tensor:
    return get_test_data()[:batch_size]

def main():
    model = ModelClass()

    all_x, all_y = get_training_data()
    ntrain = int(len(all_x) * split_ratio)
    train_with(model, all_x[:ntrain], all_y[:ntrain], nepoch=nepoch, batch_size=train_batch_size)
    predicted = infer_for(model, all_x[ntrain:])
    score = calc_score(predicted, all_y[ntrain:])
    print(f"Test score is {score}")

    test_prediction = infer_for(model, get_test_data())
    out_df = pd.DataFrame(data={"ImageId": list(range(1, len(test_prediction) + 1)), "Label": test_prediction})
    out_df.to_csv("/tmp/submission.csv", index=False)
    print("bye")

if __name__ == "__main__":
    main()
