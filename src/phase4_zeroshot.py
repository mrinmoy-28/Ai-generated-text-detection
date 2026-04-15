# src/phase4_zeroshot.py
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class ZeroShotDetector:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model     = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()

    def get_log_prob(self, text):
        """Average log probability — higher = more AI-like"""
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            # Negative loss = average log prob
            log_prob = -outputs.loss.item()

        return log_prob

    def detect(self, text, n_perturbations=10):
        """DetectGPT: compare original vs perturbed log-probs"""
        original_score = self.get_log_prob(text)

        # Perturb: randomly mask and replace words
        perturbed_scores = []
        words = text.split()

        for _ in range(n_perturbations):
            perturbed = words.copy()
            # Randomly replace 15% of words with similar words
            for i in range(len(perturbed)):
                if np.random.random() < 0.15:
                    perturbed[i] = np.random.choice(words)  # simple perturbation
            perturbed_text  = " ".join(perturbed)
            perturbed_scores.append(self.get_log_prob(perturbed_text))

            # If original >> perturbed → AI wrote it (sits on probability peak)
            perturbation_gap = original_score - np.mean(perturbed_scores)
            ai_probability   = 1 / (1 + np.exp(-perturbation_gap))  # sigmoid
            return round(ai_probability, 4)
    
    
if __name__ == "__main__":
    detector = ZeroShotDetector()
    
# Test samples
texts = [
    "The quick brown fox jumps over the lazy dog.", # Likely Human
    "In today's digital landscape, the intersection of artificial intelligence and machine learning is pivotal." # Likely AI-style
]

print("\n=== ZERO-SHOT DETECTION RESULTS ===")
for t in texts:
    score = detector.detect(t)
    label = "AI-Generated" if score > 0.5 else "Human-Written"
    print(f"Text: {t[:50]}...")
    print(f"AI Probability: {score} | Prediction: {label}\n")