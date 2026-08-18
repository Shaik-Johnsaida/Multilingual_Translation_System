"""
Generate 3 sample MP3 audio files (~2 minutes each) in:
  - Hindi (hi)
  - French (fr)
  - Japanese (ja)

Uses gTTS (Google Text-to-Speech), which is already installed.
Output files are saved to: sample_audio/
"""

import os
from gtts import gTTS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── HINDI ────────────────────────────────────────────────────────────────────
HINDI_TEXT = """
नमस्ते। आज हम बहुभाषी अनुवाद प्रणाली के बारे में बात करेंगे।
भाषा मानव संचार का सबसे महत्वपूर्ण माध्यम है।
यह हमें अपने विचारों, भावनाओं और अनुभवों को दूसरों के साथ साझा करने में सहायता करती है।
भारत में लगभग 22 आधिकारिक भाषाएँ हैं, जिनमें हिंदी सबसे अधिक बोली जाने वाली भाषा है।
कृत्रिम बुद्धिमत्ता और मशीन लर्निंग ने अनुवाद की दुनिया में क्रांति ला दी है।
आज के समय में, एक स्वचालित प्रणाली कुछ ही सेकंड में एक भाषा से दूसरी भाषा में अनुवाद कर सकती है।
यह तकनीक विशेष रूप से उन लोगों के लिए उपयोगी है जो विभिन्न देशों और संस्कृतियों के बीच संवाद स्थापित करना चाहते हैं।
हमारी प्रणाली दस प्रमुख भाषाओं का समर्थन करती है, जिनमें तेलुगु, हिंदी, अंग्रेजी, उर्दू, तमिल, मलयालम, चीनी, जापानी, स्पेनिश और फ्रेंच शामिल हैं।
ऑडियो और वीडियो अनुवाद के साथ-साथ, यह प्रणाली वास्तविक समय में वाक्-पाठ और पाठ-वाक् रूपांतरण भी प्रदान करती है।
वक्ता पहचान एक महत्वपूर्ण विशेषता है जो यह सुनिश्चित करती है कि एक वीडियो में विभिन्न वक्ताओं की आवाजें अलग-अलग पहचानी जाएं।
यह तकनीक शिक्षा, व्यवसाय, चिकित्सा और मनोरंजन जैसे अनेक क्षेत्रों में उपयोगी है।
भाषाई विविधता को बनाए रखते हुए वैश्विक संचार को सुगम बनाना हमारा मुख्य लक्ष्य है।
हम आशा करते हैं कि यह प्रणाली आपके जीवन को और अधिक सुविधाजनक बनाएगी।
धन्यवाद।
""".strip()

# ── FRENCH ───────────────────────────────────────────────────────────────────
FRENCH_TEXT = """
Bonjour et bienvenue dans notre présentation sur le système de traduction multilingue.
La langue est l'un des aspects les plus fascinants de la culture humaine.
Elle permet aux individus de communiquer leurs pensées, leurs émotions et leurs expériences avec les autres.
Dans le monde d'aujourd'hui, la barrière des langues représente l'un des plus grands défis pour la communication internationale.
Grâce aux avancées en intelligence artificielle et en apprentissage automatique, il est désormais possible de traduire automatiquement des textes, des fichiers audio et des vidéos d'une langue à une autre.
Notre système prend en charge dix langues principales : le télougou, l'hindi, l'anglais, l'ourdou, le tamoul, le malayalam, le chinois mandarin, le japonais, l'espagnol et le français.
La reconnaissance vocale automatique, ou ASR, est au cœur de notre pipeline de traitement audio et vidéo.
Cette technologie permet de transcrire la parole en texte avec une grande précision.
Ensuite, le moteur de traduction neuronale transforme ce texte en langue cible.
Enfin, la synthèse vocale convertit le texte traduit en parole naturelle dans la langue souhaitée.
L'identification des locuteurs est une fonctionnalité clé qui permet de distinguer les différentes voix dans une vidéo ou un enregistrement audio.
Cela garantit que chaque locuteur reçoit une voix cohérente tout au long de la traduction.
Nos sous-titres synchronisés permettent aux téléspectateurs de suivre facilement le contenu traduit.
Nous sommes convaincus que cette technologie contribuera à rapprocher les peuples du monde entier.
Merci de votre attention.
""".strip()

# ── JAPANESE ─────────────────────────────────────────────────────────────────
JAPANESE_TEXT = """
こんにちは。多言語翻訳システムの紹介へようこそ。
言語は人間のコミュニケーションにおいて最も重要な要素のひとつです。
私たちは言語を通じて、思想、感情、そして経験を他者と共有することができます。
現代の世界では、人工知能と機械学習の進歩により、テキスト、音声、ビデオを自動的に翻訳することが可能になりました。
私たちのシステムは、テルグ語、ヒンディー語、英語、ウルドゥー語、タミル語、マラヤーラム語、中国語、日本語、スペイン語、フランス語の10言語に対応しています。
音声認識技術は、音声をテキストに変換するための核心的な技術です。
この技術により、話された言葉を高い精度で文字起こしすることができます。
次に、ニューラル機械翻訳エンジンがそのテキストをターゲット言語に変換します。
最後に、音声合成技術が翻訳されたテキストを自然な音声に変換します。
話者識別は、動画や音声録音において異なる声を区別するための重要な機能です。
これにより、各話者は翻訳全体を通じて一貫した音声を受け取ることができます。
同期された字幕は、視聴者が翻訳されたコンテンツを簡単に追えるよう支援します。
この技術は、教育、ビジネス、医療、エンターテインメントなど、さまざまな分野で活用されています。
言語の多様性を守りながら、グローバルなコミュニケーションを促進することが私たちの目標です。
ご清聴ありがとうございました。
""".strip()

SAMPLES = [
    ("hindi",    "hi", HINDI_TEXT,    "sample_hindi.mp3"),
    ("french",   "fr", FRENCH_TEXT,   "sample_french.mp3"),
    ("japanese", "ja", JAPANESE_TEXT, "sample_japanese.mp3"),
]

for lang_name, lang_code, text, filename in SAMPLES:
    out_path = os.path.join(OUTPUT_DIR, filename)
    print(f"[gTTS] Generating {lang_name} ({lang_code}) -> {out_path} ...")
    tts = gTTS(text=text, lang=lang_code, slow=False)
    tts.save(out_path)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  [OK] Saved: {filename}  ({size_kb:.1f} KB)")

print("\nAll 3 audio files generated successfully.")
print(f"Location: {OUTPUT_DIR}")
