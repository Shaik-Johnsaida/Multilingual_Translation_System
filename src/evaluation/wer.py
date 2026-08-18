"""
Speech Recognition Evaluation Module.
Computes Word Error Rate (WER) and Character Error Rate (CER) via jiwer.
"""

from typing import Dict, Any, Optional
import jiwer


def compute_wer_score(hypothesis: str, reference: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes WER and CER for ASR transcript against ground truth reference.
    """
    if not reference or not reference.strip():
        return {
            "wer": "N/A",
            "cer": "N/A",
            "status": "Reference transcript unavailable"
        }

    if not hypothesis or not hypothesis.strip():
        return {
            "wer": 1.0,
            "cer": 1.0,
            "status": "Hypothesis empty"
        }

    try:
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        return {
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "status": "Computed successfully"
        }
    except Exception as e:
        return {
            "wer": "N/A",
            "cer": "N/A",
            "status": f"WER error: {str(e)}"
        }
