# src/phase4_watermark.py
import hashlib
import numpy as np

class WatermarkDetector:
    """
    Implements Kirchenbauer et al. 2023 watermark detection.
    Uses z-score test on green-list token frequency.
    """
    def __init__(self, vocab_size=50257, gamma=0.25, delta=2.0, seed=42):
        self.vocab_size = vocab_size
        self.gamma      = gamma   # fraction of vocab in green list
        self.delta      = delta
        self.seed       = seed

    def _get_green_list(self, prev_token_id):
        """Generate green list based on previous token"""
        rng = np.random.default_rng(
            seed=int(hashlib.sha256(str(prev_token_id).encode()).hexdigest(), 16) % (2**32)
        )
        green_size = int(self.gamma * self.vocab_size)
        green_list = set(rng.choice(self.vocab_size, green_size, replace=False))
        return green_list

    def detect(self, token_ids):
        """Returns z-score — higher means more likely watermarked"""
        if len(token_ids) < 2:
            return 0.0

        green_count = 0
        total       = len(token_ids) - 1

        for i in range(1, len(token_ids)):
            green_list = self._get_green_list(token_ids[i-1])
            if token_ids[i] in green_list:
                green_count += 1

        expected   = self.gamma * total
        std_dev    = np.sqrt(total * self.gamma * (1 - self.gamma))
        z_score    = (green_count - expected) / std_dev if std_dev > 0 else 0

        return round(z_score, 4)  # z > 4 → likely watermarked