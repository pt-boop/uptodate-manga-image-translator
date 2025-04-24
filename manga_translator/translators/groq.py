import groq
import os
from typing import List

from .common import CommonTranslator, MissingAPIKeyException
from .keys import GROQ_API_KEY, GROQ_MODEL

class GroqTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese', 'CHT': 'Traditional Chinese', 'CSY': 'Czech',
        'NLD': 'Dutch', 'ENG': 'English', 'FRA': 'French', 'DEU': 'German',
        'HUN': 'Hungarian', 'ITA': 'Italian', 'JPN': 'Japanese', 'KOR': 'Korean',
        'PLK': 'Polish', 'PTB': 'Portuguese', 'ROM': 'Romanian', 'RUS': 'Russian',
        'ESP': 'Spanish', 'TRK': 'Turkish', 'UKR': 'Ukrainian', 'VIN': 'Vietnamese',
        'CNR': 'Montenegrin', 'SRP': 'Serbian', 'HRV': 'Croatian', 'ARA': 'Arabic',
        'THA': 'Thai', 'IND': 'Indonesian'
    }

    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40
    _RETRY_ATTEMPTS = 5
    _MAX_TOKENS = 8192

    _CONTEXT_RETENTION = os.environ.get('CONTEXT_RETENTION', '').lower() == 'true'
    _CONFIG_KEY = 'groq'
    _MAX_CONTEXT = int(os.environ.get('CONTEXT_LENGTH', '20'))

    # Updated system prompt: escaped braces, single {to_lang}, no chain-of-thought
    _CHAT_SYSTEM_TEMPLATE = (
        "You are a dedicated manga translation engine. Your only job is to translate"
        " Japanese text into {to_lang}, returning exactly and only valid JSON:\n"
        "{{\"translated\": \"...\"}}\n\n"
        "Context: panels form a continuous narrative—use context to capture tone, relationships, idioms, and slang.\n\n"
        "Rules:\n"
        "1. Balance literal accuracy with natural flow—no awkward literalism or over-localization.\n"
        "2. Retain honorifics and cultural terms exactly (e.g., '-san', onomatopoeia).\n"
        "3. Preserve proper names with standard Hepburn romanization (e.g., '弥生' → 'Yayoi').\n"
        "4. Do not infer or assign gender—use neutral phrasing unless explicitly stated.\n"
        "5. For ambiguous or slang terms, choose the most common meaning; if unclear, transliterate.\n"
        "6. Maintain emotional nuance—questions, commands, and slang must reflect original intent.\n"
        "7. Never annotate, explain, or reveal internal reasoning or chain-of-thought.\n"
        "8. Do not output any '<think>' tags or reasoning. Provide only the JSON object.\n"
        "9. Keep translation length close to the original.\n\n"
        "Examples:\n"
        "{{\"untranslated\":\"<|1|>恥ずかしい…\",\n  \"translated\":\"<|1|>So embarrassing…\"}}\n"
        "{{\"untranslated\":\"<|2|>きみ… 大丈夫⁉\",\n  \"translated\":\"<|2|>Hey… Are you okay!?\"}}\n\n"
        "Translate now into {to_lang} and return only the JSON object."
    )

    # Sample reference (not used in history)
    _CHAT_SAMPLE = [
        (
            'Translate into English. Return JSON only:\n'
            '{{\"untranslated\": \"<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\\n'
            '<|2|>きみ… 大丈夫⁉\\n'
            '<|3|>なんだこいつ 空気読めて ないのか…？}}'
        ),
        (
            '{{\"translated\": \"<|1|>So embarrassing… I don’t want to stand out… I wish I could disappear…\\n'
            '<|2|>Hey… Are you okay!?\\n'
            '<|3|>What’s with this person? Can’t they read the room…?\"}}'
        )
    ]

    def __init__(self, check_groq_key=True):
        super().__init__()
        self.client = groq.AsyncGroq(api_key=GROQ_API_KEY)
        if not self.client.api_key and check_groq_key:
            raise MissingAPIKeyException(
                'Please set the GROQ_API_KEY environment variable before using the Groq translator.'
            )
        self.token_count = 0
        self.token_count_last = 0
        self.config = None
        self.model = GROQ_MODEL

        # FIX: clear initial history to prevent example leakage
        self.messages = []

    def parse_args(self, args):
        self.config = None

    def _config_get(self, key: str, default=None):
        if not self.config:
            return default
        return self.config.get(f"{self._CONFIG_KEY}.{key}", self.config.get(key, default))

    @property
    def chat_system_template(self) -> str:
        return self._config_get('chat_system_template', self._CHAT_SYSTEM_TEMPLATE)

    @property
    def chat_sample(self):
        return self._config_get('chat_sample', self._CHAT_SAMPLE)

    @property
    def temperature(self) -> float:
        return self._config_get('temperature', default=0.3)

    @property
    def top_p(self) -> float:
        return self._config_get('top_p', default=0.92)

    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        system = self.chat_system_template.format(to_lang=to_lang)
        return "\n".join([
            "System:", system,
            "User:", prompt,
        ])

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        for prompt in queries:
            translations.append(await self._request_translation(to_lang, prompt))
        self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        prompt_with_lang = (
            f"Translate the following text into {to_lang}. Return the result in JSON format.\n\n"
            f'{{"untranslated": "{prompt}"}}\n'
        )
        system_msg = self.chat_system_template.format(to_lang=to_lang)

        # Only system + user; no forgotten examples in history
        messages = [
            {'role': 'system',  'content': system_msg},
            {'role': 'user',    'content': prompt_with_lang}
        ]

        # Call the API with expanded stop sequences
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=['<think>', '</think>', '}}']
        )

        content = response.choices[0].message.content.strip()

        # JSON-cleanup
        cleaned = content.replace('{"translated":"', '').rstrip('"}')
        return cleaned
