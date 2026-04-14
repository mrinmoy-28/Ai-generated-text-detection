# src/phase6_explainability.py
import shap
import numpy as np
import matplotlib.pyplot as plt
from transformers import (RobertaTokenizer,
                          RobertaForSequenceClassification, pipeline)

class ExplainableDetector:
    def __init__(self, model_path="./models/roberta_classifier"):
        self.tokenizer  = RobertaTokenizer.from_pretrained(model_path)
        self.model      = RobertaForSequenceClassification.from_pretrained(model_path)
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True
        )

    def explain(self, text):
        # Step 1: Get SHAP values
        explainer = shap.Explainer(self.classifier)
        shap_vals = explainer([text], max_evals=500)  # limit for speed

        # Step 2: Extract word-level contributions
        words  = shap_vals.data[0]
        values = shap_vals.values[0][:, 1]  # index 1 = AI class

        word_scores = []
        for word, score in zip(words, values):
            word_scores.append({
                'word':  word,
                'score': round(float(score), 4),
                'push':  'AI' if score > 0 else 'Human'
            })

        # Sort by absolute importance
        word_scores = sorted(word_scores,
                             key=lambda x: abs(x['score']),
                             reverse=True)

        # Step 3: Print top contributing words
        print("\n🔍 Top words pushing toward AI:")
        for w in [x for x in word_scores if x['push'] == 'AI'][:5]:
            print(f"   '{w['word']}' → +{w['score']}")

        print("\n✅ Top words pushing toward Human:")
        for w in [x for x in word_scores if x['push'] == 'Human'][:5]:
            print(f"   '{w['word']}' → {w['score']}")

        # Step 4: Save SHAP plot
        shap.plots.text(shap_vals, display=False)
        plt.savefig("shap_explanation.png",
                    bbox_inches='tight', dpi=150)
        print("\n✅ Saved: shap_explanation.png")

        return word_scores


# Test it
if __name__ == "__main__":
    detector = ExplainableDetector()
    text = "The mitochondria is the powerhouse of the cell and performs cellular respiration efficiently."
    results = detector.explain(text)