import groq
import os
import json
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

    # System prompt with literal JSON braces escaped
    _CHAT_SYSTEM_TEMPLATE = (
        "You are a professional manga translation engine. Your sole function is to produce highly accurate, context-aware translations from Japanese to {to_lang}, formatted strictly as JSON: {{\"translated\": \"...\"}}.\n"
        "Include the following glossary for special terms: 'カモミール': 'chamomile', 'メガネ拭き': 'lens cloth'.\n\n"
        "Analyze panels in sequence, preserve honorifics, names, and tone. Output only JSON: {{\"translated\": \"...\"}}"
    )

    _CHAT_SAMPLE = [
        (
            'Translate into English. Return result in JSON.\n'
            '{{"untranslated": "<|1|>恥ずかしい…\\n<|2|>きみ…\\n<|3|>なんだこいつ…"}}\n'
        ),
        (
            '{{"translated": "So embarrassing…\\nHey…\\nWhat’s with this person?"}}'
        )
    ]

    def __init__(self, check_groq_key=True):
        super().__init__()
        self.client = groq.AsyncGroq(api_key=GROQ_API_KEY)
        if not self.client.api_key and check_groq_key:
            raise MissingAPIKeyException('Set GROQ_API_KEY before using translator')
        self.token_count = 0
        self.token_count_last = 0
        self.config = None
        self.model = GROQ_MODEL
        self.messages = [
            {'role': 'user', 'content': self.chat_sample[0]},
            {'role': 'assistant', 'content': self.chat_sample[1]}
        ]

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

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        results = []
        for prompt in queries:
            response = await self._request_translation(to_lang, prompt)
            results.append(response.get("translated", ""))
        self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
        return results

    async def _request_translation(self, to_lang: str, prompt: str) -> dict:
        # Build prompt
        prompt_with_lang = (
            f"Translate the following into {to_lang}. Return JSON.\n\n"
            f"{{\"untranslated\": \"{prompt}\"}}\n"
        )
        # Append user message
        self.messages.append({'role': 'user', 'content': prompt_with_lang})
        if len(self.messages) > self._MAX_CONTEXT:
            self.messages = self.messages[-self._MAX_CONTEXT:]

        # System message
        system_msg = {'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)}

        # API call without premature stop
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[system_msg] + self.messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p
        )

        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens

        raw = response.choices[0].message.content.strip()

        # Parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback raw text
            data = {"translated": raw.strip('"{}')}

        # Post-process glossary terms
        glossary = {'ウモミール': 'chamomile', 'カモミール': 'chamomile', 'メガネ拭き': 'lens cloth'}
        translated = data.get("translated", "")
        for jp_term, en_term in glossary.items():
            translated = translated.replace(jp_term, en_term)
        data["translated"] = translated

        # Context retention
        if self._CONTEXT_RETENTION:
            self.messages.append({'role': 'assistant', 'content': raw})
        else:
            if self.messages and self.messages[-1]['role'] == 'assistant':
                self.messages.pop()

        return data
