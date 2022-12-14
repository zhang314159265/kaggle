import pandas as pd
import pathlib
import re
import string
from datasets import Dataset
from functools import lru_cache
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, default_data_collator, Trainer

proj_folder = str(pathlib.Path(__file__).parent)
max_length = 64

shrink=True

def remove_url(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub("", text)

def remove_emoji(text):
    """
    Copied from the example notebook
    """
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" # emoticons
        u"\U0001F300-\U0001F5FF" # symbols & pictographs
        u"\U0001F680-\U0001F6FF" # transport & map symbols
        u"\U0001F1E0-\U0001F1FF" # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub("", text)

def remove_html(text):
    pat = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return pat.sub("", text)

def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)

def get_model():
    return BertForSequenceClassification.from_pretrained("bert-large-uncased")

# A bit hacky since get_model is not really a PyTorch nn.Module class
ModelClass = get_model

@lru_cache(maxsize=None)
def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-large-uncased")

def get_training_data():
    """
    Return a pair of train/valid dataset
    """
    train = pd.read_csv(f"{proj_folder}/data/train.csv")
    train["clean_text"] = train["text"].apply(remove_url)
    train["clean_text"] = train["clean_text"].apply(remove_emoji)
    train["clean_text"] = train["clean_text"].apply(remove_html)
    train["clean_text"] = train["clean_text"].apply(remove_punct)
    train["clean_text"] = train["clean_text"].apply(lambda x: x.lower())
    train["input_ids"] = train["clean_text"].apply(lambda x: get_tokenizer()(x, max_length=max_length, padding="max_length")["input_ids"])
    train.rename(columns={"target": "labels"}, inplace=True)
    train = train[["input_ids", "labels"]]

    train_split = int(len(train) * 0.01) # leave 1 percent for validation
    if not shrink:
        train_df = train[:-train_split].reset_index(drop=True)
    else:
        train_df = train[:100]
    valid_df = train[-train_split:].reset_index(drop=True)

    train_ds = Dataset.from_pandas(train_df)
    valid_ds = Dataset.from_pandas(valid_df)
    return train_ds, valid_ds

def get_test_data():
    """
    Return the test dataset
    """
    test = pd.read_csv(f"{proj_folder}/data/test.csv")
    test["clean_text"] = test["text"].apply(remove_url)
    test["clean_text"] = test["clean_text"].apply(remove_emoji)
    test["clean_text"] = test["clean_text"].apply(remove_html)
    test["clean_text"] = test["clean_text"].apply(remove_punct)
    test["clean_text"] = test["clean_text"].apply(lambda x: x.lower())
    test["input_ids"] = test["clean_text"].apply(lambda x: get_tokenizer()(x, max_length=max_length, padding="max_length")["input_ids"])
    test = test[["input_ids"]]
    if shrink:
        test = test[:100]
    
    test_ds = Dataset.from_pandas(test)
    return test_ds

def get_example_batch(batch_size=32):
    ds = get_test_data()
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )
    return next(iter(dl))

def main():
    # The loaded model is pre-trained. The data for this kaggle contest
    # will be used to fine-tune the model. This is a kind of transfer learning.
    model = get_model()

    train_ds, valid_ds = get_training_data()

    batch_size = 16
    args = TrainingArguments(
        f"{proj_folder}/data/checkpoint",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        gradient_accumulation_steps=8,
        num_train_epochs=3 if not shrink else 1,
        warmup_ratio=0.1,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )
    data_collator = default_data_collator
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=get_tokenizer(),
    )
    trainer.train()

    test_ds = get_test_data()
    outputs = trainer.predict(test_ds)

    sub = pd.read_csv(f"{proj_folder}/data/sample_submission.csv")
    if shrink:
        sub = sub[:100]
    sub["target"] = outputs.predictions.argmax(1)
    sub.to_csv("/tmp/submission.csv", index=False)
    print("bye")

if __name__ == "__main__":
    main()
