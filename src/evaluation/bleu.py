"""
SacreBLEU & Quality Evaluation Engine.
Computes SacreBLEU and chrF score for translations when reference text is provided.
"""

from typing import Dict, Any, Optional
import sacrebleu


def compute_bleu_score(hypothesis: str, reference: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes BLEU and chrF score using SacreBLEU.
    If reference is missing or empty, gracefully returns N/A without fabricating metrics.
    """
    if not reference or not reference.strip():
        return {
            "bleu": "N/A",
            "chrf": "N/A",
            "status": "Reference translation unavailable"
        }

    if not hypothesis or not hypothesis.strip():
        return {
            "bleu": 0.0,
            "chrf": 0.0,
            "status": "Hypothesis empty"
        }

    try:
        bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]])
        chrf = sacrebleu.corpus_chrf([hypothesis], [[reference]])
        return {
            "bleu": round(bleu.score, 2),
            "chrf": round(chrf.score, 2),
            "status": "Computed successfully"
        }
    except Exception as e:
        return {
            "bleu": "N/A",
            "chrf": "N/A",
            "status": f"Evaluation error: {str(e)}"
        }
