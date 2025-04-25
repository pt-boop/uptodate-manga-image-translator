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

    # API rate limiting and retry settings
    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40
    _RETRY_ATTEMPTS = 5
    _MAX_TOKENS = 8192

    # Context retention settings
    # @TODO Use `gpt_config` file rather than environment variables
    _CONTEXT_RETENTION = os.environ.get('CONTEXT_RETENTION', '').lower() == 'true'
    _CONFIG_KEY = 'groq'
    _MAX_CONTEXT = int(os.environ.get('CONTEXT_LENGTH', '20'))

    
    _CHAT_SYSTEM_TEMPLATE = (
        "You are a dedicated manga translation engine specializing in Japanese→{to_lang}.\n\n"
        "Your output MUST be exactly one valid JSON object:\n"
        "  {{\"translated\": \"…\"}}\n"
        "and nothing else.\n\n"
        "KEY OBJECTIVES:\n"
        "  • Context-aware: Analyze current and prior panels as a continuous narrative.\n"
        "  • Emotional fidelity: Preserve tone, subtext, and pacing (anime/manga style).\n"
        "  • Bubble-fit: Keep translations concise—ideally within 80% of original character count.\n\n"
        "RULES:\n"
        "1. **Honorifics & cultural terms**: Keep Japanese honorifics (-san, -sama, -chan, etc.) unchanged.\n"
        "2. **Names**: Standard Hepburn romanization (e.g., 弥生 → Yayoi).\n"
        "3. **Neutrality**: Never assume gender or add pronouns unless explicitly in the source.\n"
        "4. **Slang & ambiguity**: Use common meaning; if unsure, transliterate.\n"
        "5. **Onomatopoeia & SFX**: Retain original Japanese sounds (ドキドキ, ゴゴゴ).\n"
        "6. **Length control**: Do not exceed 1.2× original length.\n"
        "7. **Formatting**: No extra keys or notes—output raw JSON only.\n\n"
        "POST-PROCESS CHECK:\n"
        "- Validate output is parseable JSON.\n"
        "- Confirm only one object with key “translated”.\n\n"
        "Translate now into {to_lang}."
    )

    _CHAT_SAMPLE = [
        (
            "Translate the following manga dialogue into {to_lang}.\n"
            "Return exactly one JSON object with key \"translated\":\n"
            '{"untranslated": "<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\\n'
            '<|2|>きみ… 大丈夫⁉\\n'
            '<|3|>なんだこいつ 空気読めて ないのか…？"}'
        ),
        (
            '{"translated": "<|1|>So embarrassing… I don’t want to stand out… I just want to vanish…'
            '\\n<|2|>Hey—are you okay!?'
            '\\n<|3|>What’s up with this person? Can’t they read the room…?"}'
        )
    ]

    def __init__(self, check_groq_key=True):
        super().__init__()
        self.client = groq.AsyncGroq(api_key=GROQ_API_KEY)
        if not self.client.api_key and check_groq_key:
            raise MissingAPIKeyException('Please set the GROQ_API_KEY environment variable before using the Groq translator.')
        self.token_count = 0
        self.token_count_last = 0
        self.config = None
        self.model = GROQ_MODEL
        self.messages = [
            {'role': 'user', 'content': self.chat_sample[0]},
            {'role': 'assistant', 'content': self.chat_sample[1]}]



    def parse_args(self, args):
        #todo: is nver set
        self.config = None

    def _config_get(self, key: str, default=None):
        if not self.config:
            return default
        return self.config.get(self._CONFIG_KEY + '.' + key, self.config.get(key, default))

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
        return '\n'.join([
            'System:',
            self.chat_system_template.format(to_lang=to_lang),
            'User:',
            self.chat_sample[0],
            'Assistant:',
            self.chat_sample[1],
            'User:',
            prompt,
        ])

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        for prompt in queries:
    #        self.logger.debug('-- Groq Prompt --\n' + self._format_prompt_log(to_lang, prompt))
            response = await self._request_translation(to_lang, prompt)
            self.logger.debug('-- Groq Response --\n' + response)
            translations.append(response.strip())
        self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        # Build prompt using the unified system template
        system_msg = self.chat_system_template.format(to_lang=to_lang)
        user_msg   = prompt

        # Call the Groq API with strict JSON rules
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg}
            ],
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # Update token usage counters
        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens

        # Extract raw model output
        content = response.choices[0].message.content.strip()

        # Handle context retention
        if self._CONTEXT_RETENTION:
            self.messages.append({'role': 'assistant', 'content': content})
        else:
            self.messages = self.messages[:-1]

        # Clean out the JSON wrapper
        cleaned_content = (
            content
            .replace("{'translated':'", "")
            .replace('}', "")
            .replace("\\'", "'")
            .replace('\\"', '"')
            .strip("'{}")
        )

        # Bubble-fit enforcement
        orig_len  = len(prompt)
        trans_len = len(cleaned_content)
        if trans_len > orig_len * 1.2:
            self.logger.warning(
                f"Translation too long ({trans_len} chars vs {orig_len*1.2:.0f} max)."
            )

        # JSON post-process validation
        try:
            test_obj = json.loads(f'{{"translated": "{cleaned_content}"}}')
            if list(test_obj.keys()) != ["translated"]:
                raise ValueError("Unexpected JSON keys")
        except Exception as e:
            self.logger.error(f"JSON validation failed: {e}")

        return cleaned_content
