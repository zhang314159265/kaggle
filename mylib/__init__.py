import random
import torch
from torch import nn

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

def train_with(model, all_x, all_y, nepoch, batch_size, get_score=None, print_stats_inside_epoch=False):
    assert len(all_x) == len(all_y)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    print("Start training...")
    for epoch_id in range(nepoch):
        tot_batch = int((len(all_x) + batch_size - 1) / batch_size)
        trained_batch = 0
        for inp, label in gen_batch(all_x, all_y, batch_size, shuffle=True):
            out = model(inp)
            loss = loss_fn(out, label).sum()
            optim.zero_grad()
            loss.backward()
            optim.step()

            trained_batch += 1
            if trained_batch % 1 == 0:
                print(f"  trained batch {trained_batch}/{tot_batch}, epoch {epoch_id + 1}")
        score = None
        if get_score:
            score = get_score()
        print(f"Epoch {epoch_id + 1}/{nepoch}", end="")
        if score is not None:
            print(f", score {score}")
        else:
            print()

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
