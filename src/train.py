import spacy
from spacy.tokens import DocBin
from data.training_data import TRAINING_DATA

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

    db.to_disk("./data/train.spacy")

    from spacy.cli.train import train
    train("config/config.cfg", output_dir="./models", overrides={"paths.train": "./data/train.spacy"})

if __name__ == "__main__":
    train_model()

