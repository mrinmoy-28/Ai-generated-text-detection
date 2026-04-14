# src/phase6_ensemble.py
import torch
import numpy as np
import shap
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from phase2_statistical import get_statistical_score
from phase4_zeroshot import ZeroShotDetector
from phase4_watermark import WatermarkDetector

class HybridDetector:
    def __init__(self, model_path="./models/roberta_classifier"):
        # Load all components
        self.tokenizer    = RobertaTokenizer.from_pretrained(model_path)
        self.roberta      = RobertaForSequenceClassification.from_pretrained(model_path)
        self.classifier   = pipeline("text-classification", model=self.roberta,
                                     tokenizer=self.tokenizer)
        self.zero_shot    = ZeroShotDetector()
        self.watermark    = WatermarkDetector()

        # Ensemble weights (tune these on validation set)
        self.weights = {
            'statistical':  0.20,
            'roberta':      0.50,
            'zero_shot':    0.20,
            'watermark':    0.10,
        }

    def detect(self, text):
        scores = {}

        # 1. Statistical score
        scores['statistical'] = get_statistical_score(text)

        # 2. RoBERTa score
        result = self.classifier(text[:512])[0]
        scores['roberta'] = result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']

        # 3. Zero-shot score
        scores['zero_shot'] = self.zero_shot.detect(text)

        # 4. Watermark score (convert z-score to probability)
        token_ids = self.tokenizer(text)['input_ids']
        z_score   = self.watermark.detect(token_ids)
        scores['watermark'] = min(max(z_score / 10, 0), 1)  # normalize z to 0-1

        # Weighted ensemble
        final_score = sum(scores[k] * self.weights[k] for k in scores)

        return {
            'verdict':     'AI Generated' if final_score > 0.5 else 'Human Written',
            'confidence':  round(final_score * 100, 1),
            'breakdown':   {k: round(v * 100, 1) for k, v in scores.items()},
        }

    def detect_sentences(self, text):
        """Sentence-level detection for highlighting"""
        import spacy
        nlp      = spacy.load("en_core_web_sm")
        doc      = nlp(text)
        results  = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text.split()) < 5:
                continue
            result = self.detect(sent_text)
            results.append({
                'sentence':   sent_text,
                'ai_score':   result['confidence'],
                'is_ai':      result['verdict'] == 'AI Generated'
            })

        return results

