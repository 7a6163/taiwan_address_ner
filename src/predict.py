import spacy

def predict(text):
    nlp = spacy.load("./models/model-best")
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

if __name__ == "__main__":
    address = "新北市蘆洲區永康街1巷35號1樓"
    results = predict(address)
    for text, label in results:
        print(f"{label}: {text}")

