import spacy

# Load a pre-trained English model
nlp = spacy.load("en_core_web_sm")

def extract_facts(text):
    facts = {}
    doc = nlp(text)
    
    # Example 1: Favorite Car
    for sent in doc.sents:
        if "favorite car" in sent.text.lower() or "like car" in sent.text.lower():
            for token in sent:
                if token.ent_type_ in ["ORG", "PRODUCT"]:
                    facts["favorite_car"] = token.text
                    
    # Example 2: Favorite Color
    colors = ["red", "blue", "green", "yellow", "black", "white"]
    for token in doc:
        if token.text.lower() in colors:
            facts["favorite_color"] = token.text

    return facts

text = "My favorite car is Audi and my favorite color is blue."
print(extract_facts(text))