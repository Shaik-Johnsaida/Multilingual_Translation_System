"""
Language Detection Engine for text inputs.
Detects source language strictly within the 10 production supported languages.
"""

import re
from typing import Tuple
from src.language.registry import LANGUAGE_REGISTRY


# Common word indicators for distinguishing Latin-based supported languages
SPANISH_MARKERS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "en", "de", "que", 
    "es", "por", "para", "con", "no", "una", "hola", "buenos", "dias", "gracias", 
    "bienvenido", "bienvenidos", "como", "esta", "estas", "amigo", "muy", "todos"
}

FRENCH_MARKERS = {
    "le", "la", "les", "un", "une", "des", "et", "en", "de", "du", "que", "qui", 
    "est", "pour", "avec", "pas", "bonjour", "merci", "bienvenue", "tous", "comment", 
    "allez", "vous", "nous", "dans", "sur", "au", "aux"
}


def detect_text_language(text: str) -> Tuple[str, float]:
    """
    Detects language of input text strictly within the 10 production supported languages:
    te (tel_Telu), hi (hin_Deva), en (eng_Latn), ur (urd_Arab), ta (tam_Taml),
    ml (mal_Mlym), zh (zho_Hans), ja (jpn_Jpan), es (spa_Latn), fr (fra_Latn).
    
    Returns (flores_code, confidence).
    """
    if not text or not text.strip():
        return "eng_Latn", 1.0

    total_chars = len(text.strip())

    # 1. Check Japanese Kana (Hiragana & Katakana)
    jp_kana_matches = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\uFF65-\uFF9F]', text))
    if jp_kana_matches > 0:
        confidence = min(1.0, (jp_kana_matches * 2) / max(1, total_chars))
        return "jpn_Jpan", round(max(0.7, confidence), 2)

    # 2. Check Chinese Han ideographs (without Japanese kana)
    han_matches = len(re.findall(r'[\u4E00-\u9FFF\u3400-\u4DBF]', text))
    if han_matches > 0:
        confidence = min(1.0, han_matches / max(1, total_chars))
        return "zho_Hans", round(max(0.7, confidence), 2)

    # 3. Check Indic and Arabic/Urdu scripts
    indic_scripts = [
        (r'[\u0C00-\u0C7F]', "tel_Telu"),  # Telugu
        (r'[\u0900-\u097F]', "hin_Deva"),  # Hindi / Devanagari
        (r'[\u0B80-\u0BFF]', "tam_Taml"),  # Tamil
        (r'[\u0D00-\u0D7F]', "mal_Mlym"),  # Malayalam
        (r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]', "urd_Arab"),  # Urdu / Arabic script
    ]

    for pattern, flores_code in indic_scripts:
        matches = len(re.findall(pattern, text))
        if matches > 0:
            confidence = min(1.0, matches / max(1, total_chars))
            return flores_code, round(max(0.7, confidence), 2)

    # 4. For Latin script, differentiate French vs Spanish vs English
    # Check specific diacritics
    spanish_chars = len(re.findall(r'[¡¿ñáéíóúÁÉÍÓÚÑ]', text))
    french_chars = len(re.findall(r'[àâçèêëîïôùûüÿœæÀÂÇÈÊËÎÏÔÙÛÜŸŒÆ]', text))

    if spanish_chars > french_chars and spanish_chars > 0:
        return "spa_Latn", 0.90
    elif french_chars > spanish_chars and french_chars > 0:
        return "fra_Latn", 0.90

    # Word-level vocabulary matching for Latin text
    words = set(re.findall(r'\b\w+\b', text.lower()))
    sp_count = len(words.intersection(SPANISH_MARKERS))
    fr_count = len(words.intersection(FRENCH_MARKERS))

    if sp_count > fr_count and sp_count >= 1:
        return "spa_Latn", 0.85
    elif fr_count > sp_count and fr_count >= 1:
        return "fra_Latn", 0.85

    # Default to English
    return "eng_Latn", 0.95
