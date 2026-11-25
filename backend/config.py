from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("VOXPOLY_DATA_DIR", str(BASE_DIR / "data"))).resolve()
MODELS_DIR = DATA_DIR / "models"
VOICES_DIR = DATA_DIR / "voices"
OUTPUTS_DIR = DATA_DIR / "outputs"
VIDEOS_DIR = DATA_DIR / "videos"
WHISPER_DIR = Path(os.getenv("VOXPOLY_WHISPER_DIR", str(MODELS_DIR / "whisper"))).resolve()
SETTINGS_FILE = DATA_DIR / "settings.local.json"
FRONTEND_DIR = BASE_DIR / "frontend"

# Model & Runtime Env Config
QWEN_VARIANTS: Dict[str, Dict[str, object]] = {
    "1.7b": {
        "id": "1.7b",
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "local_dir": MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base",
        "label": "1.7B 旗舰版（推荐 ≥8 GB 显存）",
        "vram_gb": 4.2,
    },
    "0.6b": {
        "id": "0.6b",
        "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "local_dir": MODELS_DIR / "Qwen3-TTS-12Hz-0.6B-Base",
        "label": "0.6B 轻量版（适合 4–6 GB 显存 / Mac / CPU）",
        "vram_gb": 1.8,
    },
}
DEFAULT_QWEN_VARIANT = os.getenv("VOXPOLY_QWEN_VARIANT", "1.7b").strip().lower()
if DEFAULT_QWEN_VARIANT not in QWEN_VARIANTS:
    DEFAULT_QWEN_VARIANT = "1.7b"

QWEN_MODEL_ID = os.getenv("VOXPOLY_QWEN_MODEL_ID", str(QWEN_VARIANTS[DEFAULT_QWEN_VARIANT]["model_id"]))
QWEN_LOCAL_DIR = Path(
    os.getenv("VOXPOLY_QWEN_LOCAL_DIR", str(QWEN_VARIANTS[DEFAULT_QWEN_VARIANT]["local_dir"]))
).resolve()
QWEN_SAMPLE_RATE = 24000
QWEN_BATCH_SIZE = max(1, min(6, int(os.getenv("VOXPOLY_QWEN_BATCH_SIZE", "3"))))
TORCH_COMPILE_ENABLED = os.getenv("VOXPOLY_TORCH_COMPILE", "0").strip().lower() in ("1", "true", "yes")

# Download source preference: "hf" (global default) or "modelscope" (fast in China)
DOWNLOAD_SOURCE = os.getenv("VOXPOLY_DOWNLOAD_SOURCE", "hf").strip().lower()

# Whisper ASR config: defaults to "small" for broad local compatibility; set large-v3-turbo on 12GB+ GPUs / Colab
WHISPER_MODEL_SIZE = os.getenv("VOXPOLY_WHISPER_MODEL", "small").strip()
WHISPER_DEVICE = os.getenv("VOXPOLY_WHISPER_DEVICE", "auto").strip().lower()

# Isolated Worker Paths (zero dependency conflict across all 3 engines!)
CHATTERBOX_PYTHON = os.getenv("VOXPOLY_CHATTERBOX_PYTHON", "/content/chatterbox_env/bin/python").strip()
CHATTERBOX_T3_MODEL = os.getenv("VOXPOLY_CHATTERBOX_T3", "v3").strip()

COSYVOICE_PYTHON = os.getenv("VOXPOLY_COSYVOICE_PYTHON", "/content/cosyvoice_env/bin/python").strip()
COSYVOICE_REPO_DIR = os.getenv("VOXPOLY_COSYVOICE_REPO", "/content/CosyVoice").strip()
COSYVOICE_MODEL_ID = os.getenv("VOXPOLY_COSYVOICE_MODEL_ID", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512").strip()
COSYVOICE_LOCAL_DIR = MODELS_DIR / "Fun-CosyVoice3-0.5B"

# Optional passcode when sharing public Cloudflare HTTPS link
ACCESS_TOKEN = os.getenv("VOXPOLY_ACCESS_TOKEN", "").strip()

MAX_REF_AUDIO_SECONDS = 18.0
MIN_REF_AUDIO_SECONDS = 1.5
MAX_UPLOAD_MB = 50
MAX_VIDEO_MB = 300

for _d in (DATA_DIR, MODELS_DIR, VOICES_DIR, OUTPUTS_DIR, VIDEOS_DIR, WHISPER_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# 23-Language Catalog across 3 Engines:
# - qwen3: 10 languages
# - chatterbox: 23 languages
# - cosyvoice3: 9 languages + 18+ Chinese dialects & emotion instruct
LANGUAGES: List[Dict[str, object]] = [
    {
        "code": "zh",
        "qwen_name": "Chinese",
        "chatterbox_id": "zh",
        "cosyvoice_supported": True,
        "label_zh": "中文",
        "native_name": "普通话/方言",
        "translator_code": "zh-CN",
        "mymemory_code": "zh-CN",
        "region_tag": "CN",
        "engines": ["qwen3", "chatterbox", "cosyvoice3"],
    },
    {
        "code": "en",
        "qwen_name": "English",
        "chatterbox_id": "en",
        "cosyvoice_supported": True,
        "label_zh": "英语",
        "native_name": "English",
        "translator_code": "en",
        "mymemory_code": "en-US",
        "region_tag": "EN",
        "engines": ["qwen3", "chatterbox", "cosyvoice3"],
    },
    {
        "code": "ja",
        "qwen_name": "Japanese",
        "chatterbox_id": "ja",
        "cosyvoice_supported": True,
        "label_zh": "日语",
        "native_name": "日本語",
        "translator_code": "ja",
        "mymemory_code": "ja-JP",
        "region_tag": "JP",
        "engines": ["qwen3", "chatterbox", "cosyvoice3"],
    },
    {
        "code": "ko",
        "qwen_name": "Korean",
        "chatterbox_id": "ko",
        "cosyvoice_supported": True,
        "label_zh": "韩语",
        "native_name": "한국어",
        "translator_code": "ko",
        "mymemory_code": "ko-KR",
        "region_tag": "KR",
        "engines": ["qwen3", "chatterbox", "cosyvoice3"],
    },
    {
        "code": "de",
        "qwen_name": "German",
        "chatterbox_id": "de",
        "cosyvoice_supported": True,
        "label_zh": "德语",
        "native_name": "Deutsch",
        "translator_code": "de",
        "mymemory_code": "de-DE",
        "region_tag": "DE",
        "engines": ["qwen3", "chatterbox", "cosyvoice3"],
    },
    {
        "code": "fr",
        "qwen_name": "French",
        "chatterbox_id": "fr",
        "cosyvoice_supported": True,
        "label_zh": "法语",
        "native_name": "Français",
        "translator_code": "fr",
        "mymemory_code": "fr-FR",
        "region_tag": "FR",
        "engines": ["qwen3", "chatterbox", "cosyvoice3"],
    },
    {
        "code": "es",
        "qwen_name": "Spanish",
        "chatterbox_id": "es",
        "cosyvoice_supported": True,
        "label_zh": "西班牙语",
        "native_name": "Español",
        "translator_code": "es",
        "mymemory_code": "es-ES",
        "region_tag": "ES",
        "engines": ["qwen3", "chatterbox", "cosyvoice3"],
    },
    {
        "code": "ru",
        "qwen_name": "Russian",
        "chatterbox_id": "ru",
        "cosyvoice_supported": True,
        "label_zh": "俄语",
        "native_name": "Русский",
        "translator_code": "ru",
        "mymemory_code": "ru-RU",
        "region_tag": "RU",
        "engines": ["qwen3", "chatterbox", "cosyvoice3"],
    },
    {
        "code": "pt",
        "qwen_name": "Portuguese",
        "chatterbox_id": "pt",
        "cosyvoice_supported": False,
        "label_zh": "葡萄牙语",
        "native_name": "Português",
        "translator_code": "pt",
        "mymemory_code": "pt-BR",
        "region_tag": "PT",
        "engines": ["qwen3", "chatterbox"],
    },
    {
        "code": "it",
        "qwen_name": "Italian",
        "chatterbox_id": "it",
        "cosyvoice_supported": True,
        "label_zh": "意大利语",
        "native_name": "Italiano",
        "translator_code": "it",
        "mymemory_code": "it-IT",
        "region_tag": "IT",
        "engines": ["qwen3", "chatterbox", "cosyvoice3"],
    },
    # Extended 13 Languages (Chatterbox V3)
    {
        "code": "ar",
        "qwen_name": None,
        "chatterbox_id": "ar",
        "cosyvoice_supported": False,
        "label_zh": "阿拉伯语",
        "native_name": "العربية",
        "translator_code": "ar",
        "mymemory_code": "ar-SA",
        "region_tag": "AR",
        "engines": ["chatterbox"],
    },
    {
        "code": "hi",
        "qwen_name": None,
        "chatterbox_id": "hi",
        "cosyvoice_supported": False,
        "label_zh": "印地语",
        "native_name": "हिन्दी",
        "translator_code": "hi",
        "mymemory_code": "hi-IN",
        "region_tag": "HI",
        "engines": ["chatterbox"],
    },
    {
        "code": "tr",
        "qwen_name": None,
        "chatterbox_id": "tr",
        "cosyvoice_supported": False,
        "label_zh": "土耳其语",
        "native_name": "Türkçe",
        "translator_code": "tr",
        "mymemory_code": "tr-TR",
        "region_tag": "TR",
        "engines": ["chatterbox"],
    },
    {
        "code": "nl",
        "qwen_name": None,
        "chatterbox_id": "nl",
        "cosyvoice_supported": False,
        "label_zh": "荷兰语",
        "native_name": "Nederlands",
        "translator_code": "nl",
        "mymemory_code": "nl-NL",
        "region_tag": "NL",
        "engines": ["chatterbox"],
    },
    {
        "code": "pl",
        "qwen_name": None,
        "chatterbox_id": "pl",
        "cosyvoice_supported": False,
        "label_zh": "波兰语",
        "native_name": "Polski",
        "translator_code": "pl",
        "mymemory_code": "pl-PL",
        "region_tag": "PL",
        "engines": ["chatterbox"],
    },
    {
        "code": "sv",
        "qwen_name": None,
        "chatterbox_id": "sv",
        "cosyvoice_supported": False,
        "label_zh": "瑞典语",
        "native_name": "Svenska",
        "translator_code": "sv",
        "mymemory_code": "sv-SE",
        "region_tag": "SV",
        "engines": ["chatterbox"],
    },
    {
        "code": "da",
        "qwen_name": None,
        "chatterbox_id": "da",
        "cosyvoice_supported": False,
        "label_zh": "丹麦语",
        "native_name": "Dansk",
        "translator_code": "da",
        "mymemory_code": "da-DK",
        "region_tag": "DA",
        "engines": ["chatterbox"],
    },
    {
        "code": "fi",
        "qwen_name": None,
        "chatterbox_id": "fi",
        "cosyvoice_supported": False,
        "label_zh": "芬兰语",
        "native_name": "Suomi",
        "translator_code": "fi",
        "mymemory_code": "fi-FI",
        "region_tag": "FI",
        "engines": ["chatterbox"],
    },
    {
        "code": "no",
        "qwen_name": None,
        "chatterbox_id": "no",
        "cosyvoice_supported": False,
        "label_zh": "挪威语",
        "native_name": "Norsk",
        "translator_code": "no",
        "mymemory_code": "nb-NO",
        "region_tag": "NO",
        "engines": ["chatterbox"],
    },
    {
        "code": "el",
        "qwen_name": None,
        "chatterbox_id": "el",
        "cosyvoice_supported": False,
        "label_zh": "希腊语",
        "native_name": "Ελληνικά",
        "translator_code": "el",
        "mymemory_code": "el-GR",
        "region_tag": "EL",
        "engines": ["chatterbox"],
    },
    {
        "code": "he",
        "qwen_name": None,
        "chatterbox_id": "he",
        "cosyvoice_supported": False,
        "label_zh": "希伯来语",
        "native_name": "עברית",
        "translator_code": "iw",
        "mymemory_code": "he-IL",
        "region_tag": "HE",
        "engines": ["chatterbox"],
    },
    {
        "code": "ms",
        "qwen_name": None,
        "chatterbox_id": "ms",
        "cosyvoice_supported": False,
        "label_zh": "马来语",
        "native_name": "Bahasa Melayu",
        "translator_code": "ms",
        "mymemory_code": "ms-MY",
        "region_tag": "MS",
        "engines": ["chatterbox"],
    },
    {
        "code": "sw",
        "qwen_name": None,
        "chatterbox_id": "sw",
        "cosyvoice_supported": False,
        "label_zh": "斯瓦希里语",
        "native_name": "Kiswahili",
        "translator_code": "sw",
        "mymemory_code": "sw-KE",
        "region_tag": "SW",
        "engines": ["chatterbox"],
    },
]

LANGUAGE_BY_CODE: Dict[str, Dict[str, object]] = {str(item["code"]): item for item in LANGUAGES}
DEFAULT_SELECTED_LANGS: List[str] = ["zh", "en", "ja", "fr"]

COSYVOICE_INSTRUCT_PRESETS: List[Dict[str, str]] = [
    {"id": "default", "label": "标准自然原声", "instruct": ""},
    {"id": "cantonese", "label": "粤语 / 广东话", "instruct": "请用广东话表达。"},
    {"id": "sichuan", "label": "四川话（幽默）", "instruct": "请用四川话带一点轻松幽默的语气表达。"},
    {"id": "dongbei", "label": "东北话（热情）", "instruct": "请用东北话热情爽朗地表达。"},
    {"id": "shanghai", "label": "上海话", "instruct": "请用上海话表达。"},
    {"id": "warm", "label": "温暖播客风", "instruct": "请用温暖亲切、像深夜播客一样的语调表达。"},
    {"id": "excited", "label": "发布会激情", "instruct": "请用充满激情和自信的演讲语气表达。"},
    {"id": "whisper", "label": "轻声耳语", "instruct": "请用轻柔低沉、贴近麦克风的耳语语气表达。"},
]

PRESET_SCENARIOS: List[Dict[str, object]] = [
    {
        "id": "intro",
        "title": "全球自我介绍",
        "subtitle": "创作者名片与多语种问候",
        "source_lang": "zh",
        "source_text": "大家好，这是由我自己的真实声纹生成的多语种语音，现在我可以用多国语言自然地和你对话。",
        "scripts": {
            "zh": "大家好，这是由我自己的真实声纹生成的多语种语音，现在我可以用多国语言自然地和你对话。",
            "en": "Hello everyone, this multilingual voice was cloned from my own recording, so I can now speak with you naturally across many languages.",
            "ja": "皆さんこんにちは。これは私自身の声から作られた多言語ボイスです。これから様々な言語で自然にお話しできます。",
            "ko": "안녕하세요 여러분, 이 다국어 음성은 제 실제 목소리로 복제된 것입니다. 이제 여러 언어로 자연스럽게 대화할 수 있습니다.",
            "de": "Hallo zusammen, diese mehrsprachige Stimme wurde aus meiner eigenen Aufnahme geklont, sodass ich jetzt in vielen Sprachen natürlich mit euch sprechen kann.",
            "fr": "Bonjour à tous, cette voix multilingue a été clonée à partir de mon propre enregistrement, je peux désormais vous parler naturellement en plusieurs langues.",
            "es": "Hola a todos, esta voz multilingüe fue clonada a partir de mi propia grabación, así que ahora puedo hablar contigo con naturalidad en muchos idiomas.",
            "ru": "Всем привет! Этот многоязычный голос клонирован с моей собственной записи, и теперь я могу естественно общаться с вами на многих языках.",
            "pt": "Olá a todos, esta voz multilíngue foi clonada da minha própria gravação, e agora posso conversar com vocês naturalmente em vários idiomas.",
            "it": "Ciao a tutti, questa voce multilingue è stata clonata dalla mia registrazione, così ora posso parlare con voi con naturalezza in molte lingue.",
        },
    },
    {
        "id": "longform_podcast",
        "title": "长文播客试炼（自动分句）",
        "subtitle": "测试长稿智能切分、并行批加速与边生成边试听",
        "source_lang": "zh",
        "source_text": (
            "欢迎回到今天的声音实验室。过去我们总觉得，把一段中文内容变成面向全球观众的多语种播客，是一件耗时又昂贵的大工程。\n\n"
            "你不仅要反复打磨脚本，还要寻找不同语种的配音演员，甚至在剪辑室里一条一条对齐停顿与情绪。往往一个十分钟的节目，做完多语种本地化需要整整一周。\n\n"
            "而今天，只需要十秒钟属于你自己的清晰原声，系统就能自动完成长文智能分句、并行批量合成与响度平滑拼接。你可以用自己的真实声音，把同一个故事讲给东京、伦敦、巴黎和马德里的听众。"
        ),
        "scripts": {
            "zh": (
                "欢迎回到今天的声音实验室。过去我们总觉得，把一段中文内容变成面向全球观众的多语种播客，是一件耗时又昂贵的大工程。\n\n"
                "你不仅要反复打磨脚本，还要寻找不同语种的配音演员，甚至在剪辑室里一条一条对齐停顿与情绪。往往一个十分钟的节目，做完多语种本地化需要整整一周。\n\n"
                "而今天，只需要十秒钟属于你自己的清晰原声，系统就能自动完成长文智能分句、并行批量合成与响度平滑拼接。你可以用自己的真实声音，把同一个故事讲给东京、伦敦、巴黎和马德里的听众。"
            ),
            "en": (
                "Welcome back to today's Sound Lab. We used to think turning a single podcast episode into a global multilingual show was a slow, expensive undertaking.\n\n"
                "You had to polish every script, hire voice actors across regions, and spend days lining up pauses and emotion in the editing room. Even a ten-minute episode could take an entire week to localize.\n\n"
                "Today, with just ten seconds of your own clear voice, the studio automatically segments long articles, runs batched neural synthesis, and blends every sentence smoothly. Now your authentic voice can tell the same story to listeners in Tokyo, London, Paris, and Madrid."
            ),
        },
    },
    {
        "id": "product",
        "title": "科技新品发布",
        "subtitle": "Keynote 风格全球同步宣发",
        "source_lang": "zh",
        "source_text": "欢迎来到今晚的新品发布会。只需十秒原声，你的声音就能跨越语言边界，直达世界各地的每一位观众。",
        "scripts": {
            "zh": "欢迎来到今晚的新品发布会。只需十秒原声，你的声音就能跨越语言边界，直达世界各地的每一位观众。",
            "en": "Welcome to tonight's product launch. With just ten seconds of your voice, your message can cross every language barrier and reach audiences worldwide.",
            "ja": "今夜の新製品発表会へようこそ。わずか十秒のあなたの声だけで、言葉の壁を越えて世界中の視聴者へ想いを届けられます。",
            "ko": "오늘 밤 신제품 발표회에 오신 것을 환영합니다. 단 십 초의 목소리만으로 언어의 장벽을 넘어 전 세계 청중에게 다가갈 수 있습니다.",
            "de": "Willkommen zur heutigen Produktvorstellung. Mit nur zehn Sekunden deiner Stimme überwindet deine Botschaft jede Sprachgrenze und erreicht Menschen weltweit.",
            "fr": "Bienvenue au lancement de ce soir. Avec seulement dix secondes de votre voix, votre message franchit toutes les frontières linguistiques à travers le monde.",
            "es": "Bienvenidos al lanzamiento de esta noche. Con solo diez segundos de tu voz, tu mensaje cruza cualquier barrera idiomática para llegar a todo el mundo.",
            "ru": "Добро пожаловать на сегодняшнюю презентацию. Всего десять секунд вашего голоса позволят преодолеть любые языковые барьеры и обратиться к зрителям по всему миру.",
            "pt": "Bem-vindos ao lançamento de hoje. Com apenas dez segundos da sua voz, sua mensagem atravessa qualquer barreira de idioma e alcança o mundo inteiro.",
            "it": "Benvenuti al lancio di questa sera. Con soli dieci secondi della tua voce, il tuo messaggio supera ogni barriera linguistica raggiungendo il pubblico di tutto il mondo.",
        },
    },
    {
        "id": "dialect",
        "title": "方言与情绪台词（CosyVoice 3）",
        "subtitle": "搭配 CosyVoice 3 粤语/四川话/东北话与副语言标签",
        "source_lang": "zh",
        "source_text": "哎呀，今天这顿火锅真是太巴适了[laughter]，朋友难得聚在一起，咱们一会儿顺便用十种语言给全世界的网友打个招呼！",
        "scripts": {
            "zh": "哎呀，今天这顿火锅真是太巴适了[laughter]，朋友难得聚在一起，咱们一会儿顺便用十种语言给全世界的网友打个招呼！",
        },
    },
]

PRESET_BY_ID: Dict[str, Dict[str, object]] = {str(item["id"]): item for item in PRESET_SCENARIOS}
