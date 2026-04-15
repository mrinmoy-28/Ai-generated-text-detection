# src/phase5_adversarial.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# ── 1. Back-Translation Paraphrase ────────────────────────
# (Use Helsinki-NLP translation models)
from transformers import MarianMTModel, MarianTokenizer

class BackTranslator:
    def __init__(self):
        # English → French → English pipeline
        self.en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        self.en_fr_tok   = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        self.fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        self.fr_en_tok   = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

    def translate(self, texts, model, tokenizer):
        inputs = tokenizer(texts, return_tensors="pt",
                          padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def paraphrase(self, text):
        french  = self.translate([text], self.en_fr_model, self.en_fr_tok)
        english = self.translate(french,  self.fr_en_model, self.fr_en_tok)
        return english[0]


# ── 2. Synonym Substitution ───────────────────────────────
def synonym_substitute(text, rate=0.15):
    words  = text.split()
    result = words.copy()

    for i, word in enumerate(words):
        if np.random.random() < rate:
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.append(lemma.name().replace('_', ' '))
            if synonyms:
                result[i] = np.random.choice(synonyms)

    return " ".join(result)


# ── 3. Augment Dataset ────────────────────────────────────
def augment_ai_samples(df, n_augment=3):
    """
    Take AI-generated samples and paraphrase them.
    These paraphrased samples STILL have label=1 (AI)
    but look more human-like — trains robustness.
    """
    translator = BackTranslator()
    ai_df      = df[df['label'] == 1].sample(
                    min(1000, len(df[df['label']==1]))
                 )

    augmented  = []
    for _, row in ai_df.iterrows():
        text = row['text']

        # Back-translation augmentation
        try:
            para = translator.paraphrase(text)
            augmented.append({'text': para, 'label': 1, 'augmented': True})
        except:
            pass

        # Synonym substitution
        syn = synonym_substitute(text)
        augmented.append({'text': syn, 'label': 1, 'augmented': True})

    aug_df  = pd.DataFrame(augmented)
    full_df = pd.concat([df, aug_df], ignore_index=True)
    full_df.to_csv("data/augmented/hc3_augmented.csv", index=False)
    print(f"Augmented dataset: {len(full_df)} samples")
    return full_df


# ── 4. Evaluate Robustness ───────────────────────────────
def evaluate_robustness(model, tokenizer, original_texts, device='cuda'):
    """
    Compare accuracy on original vs paraphrased AI text.
    This is your KEY RESULT for the project report.
    """
    from transformers import pipeline
    detector = pipeline("text-classification", model=model,
                        tokenizer=tokenizer, device=0)

    results = {'original': [], 'paraphrased': []}

    translator = BackTranslator()
    for text in original_texts[:100]:
        orig_pred = detector(text[:512])[0]['label']
        results['original'].append(1 if orig_pred == 'LABEL_1' else 0)

        para_text = translator.paraphrase(text)
        para_pred = detector(para_text[:512])[0]['label']
        results['paraphrased'].append(1 if para_pred == 'LABEL_1' else 0)

    orig_acc = np.mean(results['original'])
    para_acc = np.mean(results['paraphrased'])

    print(f"Accuracy on Original AI text    : {orig_acc:.2%}")
    print(f"Accuracy on Paraphrased AI text : {para_acc:.2%}")
    print(f"Robustness drop                 : {(orig_acc - para_acc):.2%}")
    return orig_acc, para_acc



if __name__ == "__main__":
    import os
    import torch

    print("📥 Loading dataset...")
    df = pd.read_csv("../data/processed/hc3_cleaned.csv")
    print(f"   Original dataset: {len(df)} samples")

    os.makedirs("../data/augmented", exist_ok=True)

    print("🔄 Starting augmentation (this will take a while)...")
    full_df = augment_ai_samples(df)

    print(f"✅ Done! Augmented dataset saved.")
    print(f"   Original : {len(df)} samples")
    print(f"   Augmented: {len(full_df)} samples")