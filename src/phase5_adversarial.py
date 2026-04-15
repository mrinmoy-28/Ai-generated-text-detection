# src/phase5_adversarial.py
import pandas as pd
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import wordnet
from transformers import MarianMTModel, MarianTokenizer

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# ── 1. Back-Translation Paraphrase ────────────────────────
class BackTranslator:
    def __init__(self):
        print("📥 Loading translation models...")
        self.en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        self.en_fr_tok   = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        self.fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        self.fr_en_tok   = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.en_fr_model = self.en_fr_model.to(self.device)
        self.fr_en_model = self.fr_en_model.to(self.device)
        print(f"✅ Translation models loaded on {self.device}")

    def translate(self, texts, model, tokenizer):
        inputs = tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def paraphrase(self, text):
        try:
            french  = self.translate([text], self.en_fr_model, self.en_fr_tok)
            english = self.translate(french,  self.fr_en_model, self.fr_en_tok)
            return english[0]
        except Exception as e:
            return text  # return original if translation fails


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
def augment_ai_samples(df):
    translator = BackTranslator()

    ai_df = df[df['label'] == 1].sample(
        min(1000, len(df[df['label'] == 1])),
        random_state=42
    )

    augmented = []
    total     = len(ai_df)

    for idx, (_, row) in enumerate(ai_df.iterrows()):
        text = row['text']

        if idx % 50 == 0:
            print(f"   Progress: {idx}/{total} samples processed...")

        # Back-translation augmentation
        try:
            para = translator.paraphrase(text)
            if para and para != text:
                augmented.append({
                    'text':      para,
                    'label':     1,
                    'augmented': True
                })
        except Exception as e:
            pass

        # Synonym substitution
        try:
            syn = synonym_substitute(text)
            augmented.append({
                'text':      syn,
                'label':     1,
                'augmented': True
            })
        except Exception as e:
            pass

    aug_df  = pd.DataFrame(augmented)
    full_df = pd.concat([df, aug_df], ignore_index=True)

    # Save
    os.makedirs("../data/augmented", exist_ok=True)
    full_df.to_csv("../data/augmented/hc3_augmented.csv", index=False)

    return full_df


# ── 4. Evaluate Robustness ────────────────────────────────
def evaluate_robustness(original_texts):
    from transformers import pipeline
    from transformers import RobertaTokenizer, RobertaForSequenceClassification

    print("\n📊 Evaluating robustness on paraphrased text...")

    tokenizer = RobertaTokenizer.from_pretrained("../models/roberta_classifier")
    model     = RobertaForSequenceClassification.from_pretrained("../models/roberta_classifier")
    detector  = pipeline("text-classification", model=model,
                         tokenizer=tokenizer, device=0)

    translator  = BackTranslator()
    results     = {'original': [], 'paraphrased': []}

    for text in original_texts[:100]:
        try:
            orig_pred = detector(text[:512])[0]['label']
            results['original'].append(1 if orig_pred == 'LABEL_1' else 0)

            para_text = translator.paraphrase(text)
            para_pred = detector(para_text[:512])[0]['label']
            results['paraphrased'].append(1 if para_pred == 'LABEL_1' else 0)
        except:
            pass

    orig_acc = np.mean(results['original'])
    para_acc = np.mean(results['paraphrased'])

    print(f"✅ Accuracy on Original AI text    : {orig_acc:.2%}")
    print(f"✅ Accuracy on Paraphrased AI text : {para_acc:.2%}")
    print(f"📉 Robustness drop                 : {(orig_acc - para_acc):.2%}")

    return orig_acc, para_acc


# ── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":

    print("📥 Loading dataset...")
    df = pd.read_csv("../data/processed/hc3_cleaned.csv")
    print(f"   Original dataset : {len(df)} samples")
    print(f"   Human samples    : {(df.label==0).sum()}")
    print(f"   AI samples       : {(df.label==1).sum()}")

    print("\n🔄 Starting augmentation (this will take 20-40 mins)...")
    full_df = augment_ai_samples(df)

    print(f"\n✅ Augmentation complete!")
    print(f"   Original  : {len(df)} samples")
    print(f"   Augmented : {len(full_df)} samples")
    print(f"   Saved to  : ../data/augmented/hc3_augmented.csv")

    # Optional robustness evaluation
    # Only runs if roberta model already trained
    if os.path.exists("../models/roberta_classifier"):
        print("\n🔍 Running robustness evaluation...")
        ai_texts = df[df['label'] == 1]['text'].tolist()
        evaluate_robustness(ai_texts)
    else:
        print("\n⚠️  Skipping robustness eval — train phase3 first, then rerun this.")

    print("\n✅ Phase 5 Complete! Now rerun phase3_transformer.py")