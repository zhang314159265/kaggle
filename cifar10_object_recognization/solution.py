import torch
from torch import nn
import png
import pathlib
import pandas as pd
from typing import Tuple, Dict
from mylib import gen_batch, train_with, infer_for, calc_score
import os
import pickle
import time

proj_folder = str(pathlib.Path(__file__).parent)
train_folder = "data/train"
test_folder = "data/test"
label_path = "data/trainLabels.csv"

split_ratio = 0.8

max_train = 50000
max_test = 300000
nepoch = 15

shrink = False
if shrink:
    max_train = 100
    max_test = 10
    nepoch = 2

class ClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # in: 3 x 32 x 32
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5), # out: 6 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # out: 6 x 14 x 14
            nn.Conv2d(6, 10, 5), # out: 6 x 10 x 10
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # out: 10 x 5 x 5
        )
        self.flatten_size = 250
        # in: flatten_size
        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_size, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def forward(self, inp):
        inp = self.conv(inp)
        inp = inp.view(inp.size(0), self.flatten_size)
        return self.mlp(inp)

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = None
        
        # TODO check how this control flow is handled in Executorch/TorchDynamo etc.
        # Note this control flow is not data dependent
        if out_channel != in_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # TODO check how this control flow is handled in Executorch/TorchDynamo etc.
        # Note this control flow is not data dependent
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ClassifierResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2]):
        """
        default to 18 layer ResNet
        """
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer0 = self.make_layer(block, 64, 64, layers[0])
        self.layer1 = self.make_layer(block, 64, 128, layers[1])
        self.layer2 = self.make_layer(block, 128, 256, layers[2])
        self.layer3 = self.make_layer(block, 256, 512, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def make_layer(self, block, in_channel, out_channel, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(in_channel, out_channel))
            in_channel = out_channel
        return nn.Sequential(*layers)
        
    def forward(self, inp):
        out = self.prep(inp)

        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# ModelClass = ClassifierCNN
ModelClass = ClassifierResNet

labelStr2Int : Dict[str, int] = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}
labelInt2Str : Dict[int, str] = {v: k for k, v in labelStr2Int.items()}

def convertLabelStr2Int(labelstr):
    assert labelstr in labelStr2Int, f"label is {labelstr}, dict is {labelStr2Int}"
    return labelStr2Int[labelstr]

def convertLabelInt2Str(ind):
    assert ind in labelInt2Str, f"ind is {ind}, dict is {labelInt2Str}"
    return labelInt2Str[ind]

def getPngPath(folder, fileid):
    return f"{proj_folder}/{folder}/{fileid}.png"

def load_png(path) -> torch.Tensor:
    """
    Return a tensor of shape 3 x 32 x 32
    """
    reader = png.Reader(filename=path)
    pngdata = reader.read()
    width = pngdata[0]
    height = pngdata[1]
    assert width == 32
    assert height == 32
    assert "palette" not in pngdata[3] # the file does not contains palette
    raw_pixels = list(pngdata[2])

    rows = []
    for row_raw in raw_pixels:
        rows.append(torch.Tensor(row_raw).view(32, 3))

    # R x C x 3
    imgdata = torch.stack(rows)

    # 3 x R x C
    imgdata = imgdata.permute([2, 0, 1])
    reader.file.close()

    return imgdata


def load_files(folder, max_nfile) -> torch.Tensor:
    img_tensors = []
    for fileid in range(1, max_nfile + 1):
        path = getPngPath(folder, fileid) 
        img_tensors.append(load_png(path))

        if fileid % 10 == 0:
            print(f"Load file {fileid}/{max_nfile}")
    return torch.stack(img_tensors)

def normalize(data):
    # return data # no normalize
    return data / 255 # no normalize

def get_training_data() -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: refactor the caching mechanism to be generic
    def load():
        all_x = load_files(train_folder, max_train)
        all_y = []
        with open(f"{proj_folder}/{label_path}") as f:
            f.readline() # skip the first line
            for _ in range(max_train):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                _, labelstr = line.split(",")
    
                all_y.append(convertLabelStr2Int(labelstr))
    
        all_y = torch.LongTensor(all_y)
        assert len(all_x) == len(all_y)
        return all_x, all_y

    cache_file = f"{proj_folder}/data/cache/training_data_{max_train}.cache"
    if not os.path.exists(cache_file):
        print("Start loading training data from png files")
        start_ts = time.time()
        all_x, all_y = load()
        print(f"Done loading training data from png files, elapse {time.time() - start_ts} seconds")
        # save to cache
        print("Start writing cache")
        start_ts = time.time()
        with open(cache_file, "wb") as f:
            pickle.dump((all_x, all_y), f)
        print(f"Done writing cache, elapse {time.time() - start_ts} seconds")
    else:
        # load from cache
        print("Start loading training data cache")
        start_ts = time.time()
        with open(cache_file, "rb") as f:
            all_x, all_y = pickle.load(f)
        print(f"Done loading training data cache, elapse {time.time() - start_ts} seconds")

    return normalize(all_x), all_y

def get_test_data() -> torch.Tensor:
    # TODO: refactor the caching mechanism to be generic
    def load():
        return load_files(test_folder, max_test) 

    cache_file = f"{proj_folder}/data/cache/test_data_{max_test}.cache"
    if not os.path.exists(cache_file):
        print("Start loading testing data from png files")
        start_ts = time.time()
        test_data = load()
        print(f"Done loading testing data from png files, elapse {time.time() - start_ts} seconds")
        # save to cache
        print("Start writing cache")
        start_ts = time.time()
        with open(cache_file, "wb") as f:
            pickle.dump(test_data, f)
        print(f"Done writing cache, elapse {time.time() - start_ts} seconds")
    else:
        # load from cache
        print("Start loading testing data cache")
        start_ts = time.time()
        with open(cache_file, "rb") as f:
            test_data = pickle.load(f)
        print(f"Done loading testing data cache, elapse {time.time() - start_ts} seconds")

    return normalize(test_data)

def main():
    model = ModelClass()
    all_x, all_y = get_training_data()
    ntrain = int(len(all_x) * split_ratio)
    def get_score():
        predicted = infer_for(model, all_x[ntrain:])
        score = calc_score(predicted, all_y[ntrain:])
        return score
    train_with(model, all_x[:ntrain], all_y[:ntrain], nepoch=nepoch, batch_size=128, get_score=get_score, print_stats_inside_epoch=True)

    test_prediction = infer_for(model, get_test_data())
    out_df = pd.DataFrame(data={"id": list(range(1, len(test_prediction) + 1)), "label": [convertLabelInt2Str(elem.item()) for elem in test_prediction]})
    out_df.to_csv("/tmp/submission.csv", index=False)
    print("bye");

if __name__ == "__main__":
    main()
