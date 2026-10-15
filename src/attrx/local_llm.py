"""Локальная открытая LLM с интерфейсом OpenAI chat.completions — для экспериментов на своей GPU.

Тот же код каскада работает со шлюзом хакатона, с vLLM на ВМ и с этим классом.
По умолчанию загружает модель в 4 бита (bitsandbytes NF4): Qwen3-8B помещается в 8 ГБ видеопамяти.
"""

from __future__ import annotations

from types import SimpleNamespace


class LocalChat:
    def __init__(self, model_path: str, load_in_4bit: bool = True, max_input_tokens: int = 3000):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=False) if load_in_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant, dtype=torch.float16,
                                                          device_map="cuda:0", low_cpu_mem_usage=True)
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _render(self, messages: list[dict]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)  # Qwen3: без рассуждений
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def batch(self, conversations: list[list[dict]], max_tokens: int = 700) -> list[tuple[str, int, int]]:
        """Пакетная генерация: [(ответ, токены промпта, токены ответа)]."""
        texts = [self._render(m) for m in conversations]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                             max_length=self.max_input_tokens).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        res = []
        width = enc["input_ids"].shape[1]
        for i in range(len(texts)):
            gen = out[i, width:]
            n_gen = int((gen != (self.tokenizer.pad_token_id or -1)).sum())
            res.append((self.tokenizer.decode(gen, skip_special_tokens=True).strip(), int(enc["attention_mask"][i].sum()), n_gen))
        return res

    def _create(self, model: str, messages: list[dict], temperature: float = 0, max_tokens: int = 700, **_):
        text, n_in, n_out = self.batch([messages], max_tokens)[0]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=n_in, completion_tokens=n_out))
