import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # enlever URLs
    text = re.sub(r"@\w+", "", text)     # enlever mentions
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # enlever ponctuation et chiffres
    text = re.sub(r"\s+", " ", text).strip()  # enlever espaces multiples
    return text
