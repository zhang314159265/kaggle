import pandas as pd
import torch
from torch import nn
import numpy as np
import random

# args
# train_data_path = "data/tiny-train.csv"
train_data_path = "data/train.csv"
test_data_path = "data/test.csv"
# test_data_path = "data/tiny-test.csv"

train_batch_size = 128
split_ratio = 0.7
nepoch = 100 # 1 epoch has really poor perf

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
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

def gen_batch(all_x, all_y, batch_size, shuffle=False):
    if shuffle:
        nitem = len(all_x)
        shuf = list(range(nitem))
        random.shuffle(shuf)
        all_x = all_x[shuf]
        if all_y is not None:
            all_y = all_y[shuf]

    start = 0
    while start < len(all_x):
        x = all_x[start : start + batch_size]
        y = all_y[start : start + batch_size] if all_y is not None else None

        if all_y is not None:
            yield x, y
        else:
            yield x
        start += len(x)
    
def train_with(model, all_x, all_y):
    assert len(all_x) == len(all_y)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch_id in range(nepoch):
        # import pdb; pdb.set_trace()
        print(f"Epoch {epoch_id + 1}/{nepoch}")
        for inp, label in gen_batch(all_x, all_y, train_batch_size, shuffle=True):
            out = model(inp)
            loss = loss_fn(out, label).sum()
            optim.zero_grad()
            loss.backward()
            optim.step()

@torch.no_grad()
def infer_for(model, all_x):
    all_predicted = torch.LongTensor()
    for x in gen_batch(all_x, None, 32):
        out = model(x)
        predicted = out.argmax(dim=1)
        all_predicted = torch.cat([all_predicted, predicted])
    return all_predicted

def calc_score(predicted, ans):
    assert(len(predicted) == len(ans))
    assert(predicted.dim() == 1)
    assert(len(predicted) > 0)
    return (predicted == ans).sum().item() / len(predicted)

def main():
    train_data = pd.read_csv(train_data_path)
    all_x : np.ndarray = train_data.loc[:, train_data.columns != "label"].values
    all_y : np.ndarray = train_data["label"].values

    # convert numpy.ndarray to Tensor
    all_x = torch.Tensor(all_x)
    all_x = all_x / 255 # normalize
    all_y = torch.LongTensor(all_y)

    ntrain = int(len(all_x) * split_ratio)

    model = Classifier()
    train_with(model, all_x[:ntrain], all_y[:ntrain])
    predicted = infer_for(model, all_x[ntrain:])
    score = calc_score(predicted, all_y[ntrain:])
    print(f"Test score is {score}")

    test_data = pd.read_csv(test_data_path)
    test_data = torch.Tensor(test_data.values) / 255 # normalize
    test_prediction = infer_for(model, test_data)
    out_df = pd.DataFrame(data={"ImageId": list(range(1, len(test_prediction) + 1)), "Label": test_prediction})
    out_df.to_csv("/tmp/submission.csv", index=False)
    print("bye")

if __name__ == "__main__":
    main()
