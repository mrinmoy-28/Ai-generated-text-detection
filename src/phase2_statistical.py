# src/phase2_statistical.py
import torch
import numpy as np
import nltk
import spacy
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# ── 1. Perplexity ──────────────────────────────────────────
class PerplexityScorer:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()

    def score(self, text):
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
        with torch.no_grad():
            loss = self.model(**inputs, labels=inputs["input_ids"]).loss
        return torch.exp(loss).item()  # Lower = more AI-like


# ── 2. Burstiness ─────────────────────────────────────────
class BurstinessScorer:
    def score(self, text):
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        if len(sentences) < 2:
            return 0.0

        lengths = [len(sent.split()) for sent in sentences]
        mean = np.mean(lengths)
        std  = np.std(lengths)

        # Coefficient of variation — higher means more human-like
        burstiness = std / mean if mean > 0 else 0
        return burstiness


# ── 3. Stylometric Features ───────────────────────────────
class StylometricScorer:
    def score(self, text):
        doc = nlp(text)
        words = [t.text.lower() for t in doc if t.is_alpha]
        sentences = list(doc.sents)

        features = {
            # Vocabulary richness — AI tends to be repetitive
            "type_token_ratio": len(set(words)) / len(words) if words else 0,

            # Avg sentence length — AI tends to be uniform
            "avg_sent_length": np.mean([len(s.text.split()) for s in sentences]),

            # Punctuation density
            "punct_ratio": sum(1 for t in doc if t.is_punct) / len(doc) if len(doc) > 0 else 0,

            # Avg word length
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        }
        return features


# ── Combined Statistical Score ────────────────────────────
def get_statistical_score(text):
    perp   = PerplexityScorer().score(text)
    burst  = BurstinessScorer().score(text)
    stylo  = StylometricScorer().score(text)

    # Normalize perplexity: lower perplexity = higher AI probability
    # Typical range: AI=20-50, Human=100-300
    perp_score = 1 - min(perp / 300, 1.0)

    # Normalize burstiness: lower burstiness = higher AI probability
    burst_score = 1 - min(burst / 1.0, 1.0)

    # Combine
    ai_probability = (perp_score * 0.6) + (burst_score * 0.4)
    return round(ai_probability, 4)


if __name__ == "__main__":
    ai_text    = "Photosynthesis is the biological process by which plants convert light energy into chemical energy stored in glucose."
    human_text = "ok so basically plants eat sunlight?? i think they make sugar out of it or something. my teacher explained it but i forgot lol"

    print("AI text score    :", get_statistical_score(ai_text))
    print("Human text score :", get_statistical_score(human_text))