"""
Centralized Language Registry for Universal Multilingual Translator.
Restricted strictly to the 10 production supported languages.
"""

from typing import Dict, Any, List, Optional

# Master Language Registry containing EXACTLY the 10 production supported languages
LANGUAGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "tel_Telu": {"name": "Telugu", "code": "te", "script": "Telugu", "translation": True, "speech": True},
    "hin_Deva": {"name": "Hindi", "code": "hi", "script": "Devanagari", "translation": True, "speech": True},
    "eng_Latn": {"name": "English", "code": "en", "script": "Latin", "translation": True, "speech": True},
    "urd_Arab": {"name": "Urdu", "code": "ur", "script": "Arabic", "translation": True, "speech": True},
    "tam_Taml": {"name": "Tamil", "code": "ta", "script": "Tamil", "translation": True, "speech": True},
    "mal_Mlym": {"name": "Malayalam", "code": "ml", "script": "Malayalam", "translation": True, "speech": True},
    "zho_Hans": {"name": "Chinese (Mandarin)", "code": "zh", "script": "Simplified Han", "translation": True, "speech": True},
    "jpn_Jpan": {"name": "Japanese", "code": "ja", "script": "Japanese", "translation": True, "speech": True},
    "spa_Latn": {"name": "Spanish", "code": "es", "script": "Latin", "translation": True, "speech": True},
    "fra_Latn": {"name": "French", "code": "fr", "script": "Latin", "translation": True, "speech": True},
}


def get_supported_languages() -> List[Dict[str, Any]]:
    """Returns list of the 10 supported languages for API responses in canonical order."""
    languages = []
    for flores_code, meta in LANGUAGE_REGISTRY.items():
        languages.append({
            "id": flores_code,
            "flores_code": flores_code,
            "name": meta["name"],
            "code": meta["code"],
            "script": meta["script"],
            "translation": meta["translation"],
            "speech": meta["speech"],
            "subtitles": True
        })
    return languages


# Canonical Aliases mapping directly to the 10 FLORES codes
LANGUAGE_ALIASES: Dict[str, str] = {
    "te": "tel_Telu", "tel": "tel_Telu", "telugu": "tel_Telu", "tel_telu": "tel_Telu",
    "hi": "hin_Deva", "hin": "hin_Deva", "hindi": "hin_Deva", "hin_deva": "hin_Deva",
    "en": "eng_Latn", "eng": "eng_Latn", "english": "eng_Latn", "eng_latn": "eng_Latn",
    "ur": "urd_Arab", "urd": "urd_Arab", "urdu": "urd_Arab", "urd_arab": "urd_Arab",
    "ta": "tam_Taml", "tam": "tam_Taml", "tamil": "tam_Taml", "tam_taml": "tam_Taml",
    "ml": "mal_Mlym", "mal": "mal_Mlym", "malayalam": "mal_Mlym", "mal_mlym": "mal_Mlym",
    "zh": "zho_Hans", "zho": "zho_Hans", "chi": "zho_Hans", "chinese": "zho_Hans", "mandarin": "zho_Hans", "chinese (mandarin)": "zho_Hans", "chinese simplified": "zho_Hans", "zho_hans": "zho_Hans",
    "ja": "jpn_Jpan", "jpn": "jpn_Jpan", "japanese": "jpn_Jpan", "jpn_jpan": "jpn_Jpan",
    "es": "spa_Latn", "spa": "spa_Latn", "spanish": "spa_Latn", "spa_latn": "spa_Latn",
    "fr": "fra_Latn", "fra": "fra_Latn", "fre": "fra_Latn", "french": "fra_Latn", "fra_latn": "fra_Latn",
}


def resolve_language_code(query: str) -> Optional[str]:
    """
    Resolves input (e.g. 'en', 'Telugu', 'tel_Telu', 'te', 'hi') to official FLORES code.
    Returns None if unsupported.
    """
    if not query or query.lower() in ("auto", "auto detect", "autodetect"):
        return "auto"
    
    query_clean = query.strip()
    if query_clean in LANGUAGE_REGISTRY:
        return query_clean
    
    query_lower = query_clean.lower()
    if query_lower in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[query_lower]

    for flores_code, meta in LANGUAGE_REGISTRY.items():
        if meta["name"].lower() == query_lower or meta["code"].lower() == query_lower:
            return flores_code

    return None



def get_language_display_name(code_or_flores: str) -> str:
    """Returns human readable display name for language code."""
    if code_or_flores in ("auto", "Auto Detect"):
        return "Auto Detect"
    flores = resolve_language_code(code_or_flores)
    if flores and flores in LANGUAGE_REGISTRY:
        return LANGUAGE_REGISTRY[flores]["name"]
    return code_or_flores
