"""Few-Shot MMR Diversity Selector.
100% Python Standard Library.
"""
import re
import math
from collections import Counter

class FewShotMmrDiversitySelector:
    """Maximal Marginal Relevance (MMR) exemplar selector optimizing prompt examples."""

    @staticmethod
    def _get_tf(text: str) -> dict:
        words = re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())
        return Counter(words)

    @classmethod
    def _cosine_sim(cls, tf1: dict, tf2: dict) -> float:
        intersection = set(tf1.keys()) & set(tf2.keys())
        if not intersection:
            return 0.0
        dot = sum(tf1[w] * tf2[w] for w in intersection)
        norm1 = math.sqrt(sum(v * v for v in tf1.values()))
        norm2 = math.sqrt(sum(v * v for v in tf2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    @classmethod
    def select_exemplars(cls, query: str, pool: list, k: int = 3, lambda_param: float = 0.5) -> dict:
        if not pool:
            return {"selected_exemplars": [], "metrics": {"k": k, "pool_size": 0}}
        k = min(k, len(pool))
        q_tf = cls._get_tf(query)
        pool_tfs = [cls._get_tf(p.get("input", "") + " " + p.get("output", "")) for p in pool]

        relevances = [cls._cosine_sim(q_tf, p_tf) for p_tf in pool_tfs]

        selected_indices = []
        candidate_indices = list(range(len(pool)))

        while len(selected_indices) < k and candidate_indices:
            best_idx = None
            best_score = -float('inf')

            for idx in candidate_indices:
                rel = relevances[idx]
                if not selected_indices:
                    div_penalty = 0.0
                else:
                    div_penalty = max(cls._cosine_sim(pool_tfs[idx], pool_tfs[s_idx]) for s_idx in selected_indices)

                mmr_score = lambda_param * rel - (1.0 - lambda_param) * div_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
            else:
                break

        selected_exemplars = [pool[i] for i in selected_indices]
        return {
            "selected_exemplars": selected_exemplars,
            "metrics": {
                "k": len(selected_exemplars),
                "pool_size": len(pool),
                "lambda_param": lambda_param
            }
        }
