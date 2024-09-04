import os
import sys

# 將當前目錄（src）添加到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import spacy
from spacy.tokens import DocBin
from training_data import TRAINING_DATA
import subprocess

def train_model():
    nlp = spacy.blank("zh")
    ner = nlp.add_pipe("ner")

    for _, annotations in TRAINING_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    db = DocBin()
    for text, annot in TRAINING_DATA:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                print(f"Skipping entity: {text[start:end]} {label}")
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)

    # 創建必要的目錄
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    os.makedirs(data_dir, exist_ok=True)

    db.to_disk(os.path.join(data_dir, "train.spacy"))

    # 創建 models 目錄
    models_dir = os.path.join(os.path.dirname(current_dir), "models")
    os.makedirs(models_dir, exist_ok=True)

    config_path = os.path.join(os.path.dirname(current_dir), "config", "config.cfg")

    # 運行 spacy train 命令
    subprocess.run([
        "python", "-m", "spacy", "train",
        config_path,
        "--output", models_dir,
        "--paths.train", os.path.join(data_dir, "train.spacy"),
        "--paths.dev", os.path.join(data_dir, "train.spacy")
    ])

if __name__ == "__main__":
    train_model()
