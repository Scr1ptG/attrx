"""Судья — открытый аналог JEV: «подтверждает ли текст карточки, что <характеристика> = <значение>?».

Ответы: подтверждает / противоречит / не указано, с откалиброванной вероятностью. Модель не генерирует текст,
поэтому выдумать значение не может: она только проверяет кандидатов от других источников.

Основа (backbone), сравниваются две открытые модели:
- Laya-multilingual (Convai Innovations, Apache-2.0): mmBERT-base, 307M, модель «типизированных решений»;
- FRIDA (ai-forever, MIT): энкодер FRED-T5, сильный в русском.

Разметка бесплатная, из каталога, без ручной работы. Окна — те же, что судья увидит в каскаде: вокруг найденного
в тексте значения или упоминания характеристики. Примеры, где ответ «подстрока есть в окне» неверен, — обязательны:
- подтверждает: эталонное значение в тексте, причём число — рядом с названием или единицей; да/нет — с учётом
  отрицания («без NFC» подтверждает «NFC: Нет»);
- противоречит: то же окно, другое значение характеристики из справочника; значение соседней характеристики
  («8/256 ГБ»: встроенная память не 8 ГБ); «Да» при «без NFC»;
- не указано: характеристика упомянута, но значения в тексте нет; значение из справочника встречается в тексте
  случайно («2 года гарантии» не говорит «Вес: 2 кг»).
Обучение — на train, калибровка температурой — на val, оценка — на test (группы не пересекаются).

В каскаде судья спрашивается только о значениях без подтверждения в тексте (аналоги, похожие товары, LLM)
и только если в тексте найдено значение или упоминание характеристики — дешёвый фильтр перед моделью.
Текстовое подтверждение (источник judge) значение получает, только если оно само найдено в тексте и в контексте.
"""

from __future__ import annotations

import math
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from attrx.lexicon import FLAG_STOP, NEGATIONS, Lexicon, Slot
from attrx.values import _SPLIT, LINE_BREAK, Token, canon, card_text, key, number, tokens

LABELS = ("подтверждает", "противоречит", "не указано")
CONFIRM, CONTRA, NONE = 0, 1, 2
WINDOW = 200  # символов текста по обе стороны от упоминания
MAX_LEN = 192  # токенов на пару «гипотеза + окно»
MAX_HYPOTHESIS = 160  # символов гипотезы: остальное место — тексту карточки
JUDGED_SOURCES = frozenset({"analog", "similar", "llm"})  # значения без текстового подтверждения
MIN_CONFIRM = 0.5  # с какой вероятности «подтверждает» значение получает текстовое подтверждение
VETO = 0.5  # с какой вероятности «противоречит» значение LLM отклоняется (подбирается на val, 05_judge)
NEAR_BEFORE, NEAR_AFTER = 6, 3  # «рядом с названием»: токенов до и после числа
_PARAGRAPH = re.compile(r"(?:\s*¶\s*)+")
_CLAUSE = re.compile(r"[.!?;¶]")  # граница предложения: контекст и отрицание через неё не переносятся
_AFTER_NAME = re.compile(r"\s*[:\-–—=]?\s*")  # «NFC: нет», «NFC — отсутствует»


# ------------------------------------------------------------------ текст: где упомянуто значение
def card_full_text(title: str | None, description: str | None) -> tuple[str, int]:
    """Текст карточки для судьи и длина части «название» (чтобы понять, в каком поле найдено упоминание)."""
    t = card_text(title)
    return t + LINE_BREAK + card_text(description), len(t)


def _find_seq(norms: list[str], seq: list[str]) -> tuple[int, int] | None:
    n = len(seq)
    for i, t in enumerate(norms):
        if t == seq[0] and norms[i:i + n] == seq:
            return i, i + n
    return None


def _flag_name(attribute: str) -> list[str]:
    name = key(attribute).split()
    return [t for t in name if t not in FLAG_STOP] or name


def _value_tokens(norms: list[str], attribute: str, kind: str, value: str) -> tuple[tuple[int, int] | None, bool]:
    """Токены первой найденной части значения и найдены ли все части; для да/нет — название характеристики."""
    if kind == "bool":
        span = _find_seq(norms, _flag_name(attribute))
        return span, span is not None
    first, found_all = None, True
    for part in _SPLIT.split(value or ""):
        seq = key(part).split()
        if not seq:
            continue
        span = _find_seq(norms, seq)
        if span is None:
            found_all = False
        elif first is None:
            first = span
    return first, first is not None and found_all


def _mention_tokens(tk: list[Token], slot: Slot) -> tuple[int, int] | None:
    """Упоминание характеристики: все значимые слова названия подряд (как lexicon._number_after_name)."""
    stems = slot.stems
    if not stems:
        return None
    for j, t in enumerate(tk):
        if not t.norm.startswith(stems[0]):
            continue
        end = j
        if len(stems) > 1:
            win = tk[j + 1:j + 2 + len(stems)]
            hits = [[k for k, w in enumerate(win) if w.norm.startswith(s)] for s in stems[1:]]
            if not all(hits):
                continue
            end = j + 1 + max(h[0] for h in hits)
        return j, end + 1
    return None


def _negated(tk: list[Token], text: str, span: tuple[int, int]) -> bool:
    """«без NFC», «не поддерживает Wi-Fi», «NFC: нет», «Bluetooth отсутствует» — в пределах одной фразы."""
    i0, i1 = span
    for j in range(max(0, i0 - 2), i0):
        if tk[j].norm in NEGATIONS and not re.search(r"[.!?;,¶]", text[tk[j].end:tk[i0].start]):
            return True
    if i1 < len(tk) and (tk[i1].norm == "нет" or tk[i1].norm.startswith("отсутств")):
        return bool(_AFTER_NAME.fullmatch(text[tk[i1 - 1].end:tk[i1].start]))
    return False


def _in_context(tk: list[Token], text: str, span: tuple[int, int], slot: Slot | None, kind: str, value: str,
                unit: str | None) -> bool:
    """Число подтверждает характеристику, только если в той же фразе рядом её название или единица:
    «2 года» ≠ «Вес: 2 кг», и «Гарантия 2 года. Вес: 5 кг» вес 2 не подтверждает."""
    parts = key(value).split()
    if kind != "num" and not all(p.replace(".", "").isdigit() for p in parts):
        return True  # слова («черный», «Samsung») случайно не совпадают
    if kind == "num" and any(not p.replace(".", "").isdigit() for p in parts):
        return True  # единица — часть самого значения («256 ГБ»), она уже найдена рядом
    i0, i1 = span
    u = key(unit).split()
    if u and i1 < len(tk) and not _CLAUSE.search(text[tk[i1 - 1].end:tk[i1].start]):
        nxt, u0 = tk[i1].norm, u[0]
        if nxt == u0 or (len(u0) >= 3 and nxt.startswith(u0[:3])):  # «г» не равно «года», «дюйм» ~ «дюйма»
            return True
    if slot is None:
        return False
    near = [t for t in tk[max(0, i0 - NEAR_BEFORE):i0] if not _CLAUSE.search(text[t.end:tk[i0].start])]
    # после числа — до запятой: «для 2 моделей, вес …» вес 2 не подтверждает; до числа запятая допустима («Вес, г: 2»)
    near += [t for t in tk[i1:i1 + NEAR_AFTER] if not re.search(r"[.!?;,¶]", text[tk[i1 - 1].end:t.start])]
    return any(t.norm.startswith(s) for t in near for s in slot.stems)


def _span_in_context(tk: list[Token], text: str, norms: list[str], slot: Slot | None, kind: str, value: str,
                     unit: str | None) -> tuple[int, int] | None:
    """Первое вхождение значения, которое стоит в контексте характеристики (а не только первое вхождение)."""
    seq = next((key(p).split() for p in _SPLIT.split(value or "") if key(p)), None)
    if not seq:
        return None
    n = len(seq)
    for i, t in enumerate(norms):
        if t == seq[0] and norms[i:i + n] == seq and _in_context(tk, text, (i, i + n), slot, kind, value, unit):
            return i, i + n
    return None


def _char_span(tk: list[Token], span: tuple[int, int]) -> tuple[int, int]:
    return tk[span[0]].start, tk[span[1] - 1].end


def locate(tk: list[Token], attribute: str, kind: str, value: str) -> tuple[int, int] | None:
    """Позиция (символы) значения в тексте; для да/нет — позиция названия характеристики."""
    span, _ = _value_tokens([t.norm for t in tk], attribute, kind, value)
    return _char_span(tk, span) if span else None


def mention(tk: list[Token], slot: Slot) -> tuple[int, int] | None:
    """Позиция (символы) упоминания характеристики."""
    span = _mention_tokens(tk, slot)
    return _char_span(tk, span) if span else None


def window(text: str, span: tuple[int, int], width: int = WINDOW) -> str:
    s, e = span
    a, b = max(0, s - width), min(len(text), e + width)
    out = text[a:b]
    if a > 0 and " " in out:
        out = out.split(" ", 1)[1]
    if b < len(text) and " " in out:
        out = out.rsplit(" ", 1)[0]
    return _PARAGRAPH.sub(" / ", out).strip(" /")


def hypothesis(attribute: str, value: str, unit: str | None = None) -> str:
    return f"{attribute}: {value}{' ' + unit if unit else ''}"


def _evidence(text: str, n_title: int, span: tuple[int, int], pad: int = 40) -> tuple[str, str]:
    """Поле и точный фрагмент его текста вокруг значения (для подсветки в интерфейсе модератора)."""
    s, e = span
    if s < n_title:
        field_text, s0, e0, name = text[:n_title], s, min(e, n_title), "title"
    else:
        offset = n_title + len(LINE_BREAK)
        field_text, s0, e0, name = text[offset:], max(0, s - offset), max(0, e - offset), "description"
    a, b = max(0, s0 - pad), min(len(field_text), e0 + pad)
    return name, field_text[a:b].strip()


# ------------------------------------------------------------------ разметка из каталога
EXAMPLE_SCHEMA = {"sku_id": pl.String, "category_id": pl.String, "attribute": pl.String, "kind": pl.String,
                  "hypothesis": pl.String, "window": pl.String, "label": pl.Int8, "recipe": pl.String}


def build_examples(cards: pl.DataFrame, gold: pl.DataFrame, schema: pl.DataFrame, reference: pl.DataFrame,
                   lexicon: Lexicon, supported: pl.DataFrame, per_card: int = 2, seed: int = 13) -> pl.DataFrame:
    """Примеры для судьи из карточек одной части каталога. supported — metrics.supported_flags для этих карточек.

    recipe — как получен пример: located / swap / cross / negation / mentioned / coincidence."""
    rng = random.Random(seed)
    pairs = (gold.join(cards.select("sku_id", "category_id"), on="sku_id")
             .join(schema.select("category_id", "attribute", "kind"), on=["category_id", "attribute"])
             .join(supported, on=["sku_id", "attribute"], how="left").with_columns(pl.col("supported").fill_null(False))
             .sort("sku_id", "attribute"))
    vocab: dict[tuple[str, str], list[tuple[str, str | None]]] = {}
    ref = reference.join(schema.select("category_id", "attribute"), on=["category_id", "attribute"])
    for c, a, v, u in ref.sort("count", descending=True).select("category_id", "attribute", "value", "unit").iter_rows():
        lst = vocab.setdefault((c, a), [])
        if len(lst) < 30:
            lst.append((v, u))
    seqs = {ca: [(v, u, key(v).split()) for v, u in lst if key(v)] for ca, lst in vocab.items()}
    texts = {s: card_full_text(t, d) for s, t, d in cards.select("sku_id", "title", "description").iter_rows()}
    out: list[tuple] = []
    for (sku,), rows in pairs.group_by("sku_id", maintain_order=True):
        text, _ = texts[sku]
        tk = tokens(text)
        norms = [t.norm for t in tk]
        normset = set(norms)
        cat = rows["category_id"][0]
        slots = lexicon.slots.get(cat, {})
        confirmed, unstated, golds = [], [], {}
        for attr, kind, value, unit, sup in rows.select("attribute", "kind", "value", "unit", "supported").iter_rows():
            slot = slots.get(attr)
            if slot is None:
                continue
            golds[attr] = value
            span, all_found = _value_tokens(norms, attr, kind, value)
            if kind == "bool":
                # однозначно: «Да» без отрицания рядом или «Нет» с отрицанием («без NFC»)
                if span is not None and (key(value) == "да") != _negated(tk, text, span):
                    confirmed.append((attr, kind, value, unit, span))
            elif sup and all_found and (ctx := _span_in_context(tk, text, norms, slot, kind, value, unit)) is not None:
                confirmed.append((attr, kind, value, unit, ctx))
            elif not sup and (m := _mention_tokens(tk, slot)) is not None:
                unstated.append((attr, kind, value, unit, m))

        def add(attr, kind, hyp, span, label, recipe):
            out.append((sku, cat, attr, kind, hyp, window(text, _char_span(tk, span)), label, recipe))

        rng.shuffle(confirmed)
        rng.shuffle(unstated)
        for attr, kind, value, unit, span in confirmed[:per_card]:
            add(attr, kind, hypothesis(attr, value, unit), span, CONFIRM, "negation" if kind == "bool" and key(value) != "да" else "located")
            other = _contrast(value, kind, vocab.get((cat, attr), []), rng)
            if other is not None:
                add(attr, kind, hypothesis(attr, *other), span, CONTRA, "negation" if kind == "bool" else "swap")
        # значение соседней характеристики в том же окне: «8/256 ГБ» не значит «Встроенная память: 8 ГБ»
        cross = [(a, b) for a in confirmed for b in confirmed
                 if a[0] != b[0] and a[1] == b[1] and a[1] in ("num", "enum") and canon(a[2]) != canon(b[2])
                 and abs(tk[a[4][0]].start - tk[b[4][0]].start) <= WINDOW]
        if cross:
            a, b = rng.choice(cross)
            add(b[0], b[1], hypothesis(b[0], a[2], b[3]), b[4], CONTRA, "cross")
        # характеристика упомянута, но значения нет («регулировка мощности» без числа)
        for attr, kind, value, unit, span in unstated[:max(1, per_card // 2)]:
            add(attr, kind, hypothesis(attr, value, unit), span, NONE, "mentioned")
        # значение из справочника встретилось в тексте случайно, характеристика не упомянута
        coincidences = []
        for attr, slot in slots.items():
            if slot.kind not in ("num", "enum") or attr not in golds or _mention_tokens(tk, slot) is not None:
                continue
            g = canon(golds[attr])
            for v, u, seq in seqs.get((cat, attr), ()):
                if seq[0] in normset and canon(v) != g and (span := _find_seq(norms, seq)) is not None:
                    coincidences.append((attr, slot.kind, v, u, span))
                    break
        rng.shuffle(coincidences)
        for attr, kind, v, u, span in coincidences[:max(1, per_card // 2)]:
            add(attr, kind, hypothesis(attr, v, u), span, NONE, "coincidence")
    return pl.DataFrame(out, schema=EXAMPLE_SCHEMA, orient="row")


def _contrast(value: str, kind: str, vocab: list[tuple[str, str | None]], rng: random.Random):
    """Другое значение той же характеристики: для противоречия."""
    if kind == "bool":
        return ("Нет" if key(value) == "да" else "Да"), None
    gold_canon, gold_num = canon(value), number(value)
    options = [(v, u) for v, u in vocab if canon(v) != gold_canon and (kind != "num" or number(v) != gold_num)]
    return rng.choice(options) if options else None


def trivial_rule(examples: pl.DataFrame) -> np.ndarray:
    """Базовая линия без модели: значение в окне — подтверждает; иначе — противоречит. С ней сравниваем судью."""
    out = []
    for hyp, win in examples.select("hypothesis", "window").iter_rows():
        value = hyp.split(": ", 1)[1] if ": " in hyp else hyp
        out.append(CONFIRM if key(value) and f" {key(value)} " in f" {key(win)} " else CONTRA)
    return np.asarray(out)


# ------------------------------------------------------------------ модель
@dataclass
class Backbone:
    name: str  # laya | frida
    path: Path
    prefix: str = ""  # FRIDA обучена с префиксами задач


def load_backbone(name: str, path: Path):
    """Энкодер, токенизатор и размер скрытого слоя. Только transformers, без сторонних пакетов моделей."""
    from transformers import AutoTokenizer

    path = Path(path)
    if name == "laya":
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(path / "encoder")
        enc = AutoModel.from_config(cfg)
        state = load_file(str(path / "model.safetensors"))
        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        enc.load_state_dict(enc_state, strict=True)  # все веса энкодера должны быть в файле
        return enc, AutoTokenizer.from_pretrained(path / "tokenizer"), cfg.hidden_size, Backbone(name, path)
    if name == "frida":
        from transformers import T5EncoderModel

        enc = T5EncoderModel.from_pretrained(path)
        return enc, AutoTokenizer.from_pretrained(path), enc.config.d_model, Backbone(name, path, "categorize_entailment: ")
    raise ValueError(name)


def _blocks(encoder):
    """Список блоков энкодера: ModernBERT (Laya) — layers, T5 (FRIDA) — encoder.block."""
    if hasattr(encoder, "layers"):
        return encoder.layers
    return encoder.encoder.block


def make_net(encoder, hidden: int):
    import torch
    from torch import nn

    class JudgeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, len(LABELS)))

        def forward(self, input_ids, attention_mask):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)  # среднее по токенам
            with torch.autocast(device_type=pooled.device.type, enabled=False):  # голова в fp32: пороги стабильны
                return self.head(pooled)

    return JudgeNet().to(torch.float32)


@dataclass
class Judge:
    net: object
    tokenizer: object
    backbone: Backbone
    temperature: float = 1.0
    device: str = "cpu"
    max_len: int = MAX_LEN
    batch: int = 64
    report: dict = field(default_factory=dict)
    min_confirm: float = MIN_CONFIRM
    veto: float = VETO

    def _encode(self, hyps: list[str], windows: list[str]):
        # длинная гипотеза («Основные характеристики: …») режется вместе с окном, а не роняет токенизатор
        return self.tokenizer([self.backbone.prefix + h[:MAX_HYPOTHESIS] for h in hyps], windows, truncation="longest_first",
                              max_length=self.max_len, padding=True, return_tensors="pt")

    def logits(self, hyps: list[str], windows: list[str]) -> np.ndarray:
        import torch

        self.net.eval()
        order = np.argsort([len(h) + len(w) for h, w in zip(hyps, windows)], kind="stable")  # меньше паддинга
        out = np.zeros((len(hyps), len(LABELS)))
        with torch.no_grad(), torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16, enabled=self.device != "cpu"):
            for i in range(0, len(order), self.batch):
                idx = order[i:i + self.batch]
                enc = self._encode([hyps[k] for k in idx], [windows[k] for k in idx]).to(self.device)
                out[idx] = self.net(enc["input_ids"], enc["attention_mask"]).float().cpu().numpy()
        return out

    def probs(self, hyps: list[str], windows: list[str]) -> np.ndarray:
        z = self.logits(hyps, windows) / self.temperature
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    # -------------------------------------------------------------- обучение
    def fit(self, train: pl.DataFrame, val: pl.DataFrame, epochs: int = 1, lr: float = 3e-5, batch: int = 32,
            grad_accum: int = 1, log_every: int = 200, seed: int = 13, freeze_embeddings: bool = True,
            train_top: int | None = None, optim8bit: bool = False, log_path: Path | None = None) -> Judge:
        """Дообучение на примерах из каталога. Под GPU 8 ГБ: словарь эмбеддингов заморожен; у большой основы
        (FRIDA, 823M) обучаются только верхние train_top блоков, нижние хранятся в bf16, оптимизатор 8-битный."""
        import torch

        torch.manual_seed(seed)
        self.report.pop("id", None)  # новый судья — новый id: верификатор старого с ним не сработает
        enc = self.net.encoder
        half = self.device != "cpu"  # на CPU нет autocast: смешанные типы весов там не работают
        if freeze_embeddings:
            enc.get_input_embeddings().requires_grad_(False)
            if half and train_top is not None:
                enc.get_input_embeddings().to(torch.bfloat16)
        if train_top is not None:
            blocks = list(_blocks(enc))
            if not 0 < train_top <= len(blocks):
                raise ValueError(f"train_top={train_top}, а блоков {len(blocks)}")
            for b in blocks[:len(blocks) - train_top]:
                b.requires_grad_(False)
                if half:
                    b.to(torch.bfloat16)
        if hasattr(enc, "gradient_checkpointing_enable"):
            enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        params = [p for p in self.net.parameters() if p.requires_grad]
        self.report["trainable_params_m"] = round(sum(p.numel() for p in params) / 1e6, 1)
        if optim8bit:
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.01)
        else:
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        n_batches = math.ceil(train.height / batch)
        steps = math.ceil(n_batches / grad_accum) * epochs
        warm = max(1.0, 0.06 * steps)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps))
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        rng = np.random.default_rng(seed)
        t0, step, hist = time.time(), 0, []

        def update():
            nonlocal step
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0:
                line = f"шаг {step}/{steps}  loss {np.mean(hist[-log_every * grad_accum:]):.4f}  {time.time() - t0:.0f} c"
                print(line, flush=True)
                if log_path is not None:  # ход обучения виден снаружи ноутбука
                    with open(log_path, "a", encoding="utf-8") as f:
                        print(line, file=f)

        self.net.train()
        for _ in range(epochs):
            order = rng.permutation(train.height)
            for b in range(n_batches):
                part = train[order[b * batch:(b + 1) * batch]]
                enc_in = self._encode(part["hypothesis"].to_list(), part["window"].to_list()).to(self.device)
                y = torch.tensor(part["label"].to_numpy(), dtype=torch.long, device=self.device)
                with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16, enabled=half):
                    loss = loss_fn(self.net(enc_in["input_ids"], enc_in["attention_mask"]).float(), y)
                (loss / grad_accum).backward()
                hist.append(float(loss))
                if (b + 1) % grad_accum == 0 or b + 1 == n_batches:  # хвост эпохи не переносится в следующую
                    update()
        self.report["train"] = {"examples": train.height, "steps": step, "seconds": round(time.time() - t0, 1),
                                "final_loss": round(float(np.mean(hist[-200:])), 4) if hist else None, "seed": seed}
        self.fit_temperature(val)
        return self

    def fit_temperature(self, val: pl.DataFrame) -> float:
        """Одна температура по val (минимум log loss): Laya и другие модели решений из коробки самоуверенны."""
        z = self.logits(val["hypothesis"].to_list(), val["window"].to_list())
        y = val["label"].to_numpy()
        best = (math.inf, 1.0)
        for t in np.exp(np.linspace(np.log(0.3), np.log(5.0), 60)):
            p = np.exp(z / t - (z / t).max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            nll = -np.log(p[np.arange(len(y)), y] + 1e-9).mean()
            best = min(best, (nll, float(t)))
        self.temperature = best[1]
        self.report["temperature"] = round(self.temperature, 3)
        return self.temperature

    def evaluate(self, data: pl.DataFrame, p: np.ndarray | None = None) -> dict:
        p = self.probs(data["hypothesis"].to_list(), data["window"].to_list()) if p is None else p
        return evaluate_probs(p, data["label"].to_numpy(), self.min_confirm, self.veto)

    # -------------------------------------------------------------- каскад
    def annotate(self, frame: pl.DataFrame, cards: pl.DataFrame, lexicon: Lexicon) -> pl.DataFrame:
        """Проверяет значения без текстового подтверждения (аналоги, похожие товары, LLM).

        Судью спрашиваем, только если в тексте есть значение или упоминание характеристики. Добавляет колонки
        judge_confirm / judge_contra (null — не спрашивали). Источник judge и фрагмент текста как подтверждение
        значение получает, только если само найдено в тексте (все части, число — в контексте) и судья подтвердил.
        Решение о статусе всё равно принимает верификатор с порогом val."""
        texts = {s: card_full_text(t, d) for s, t, d in cards.select("sku_id", "title", "description").iter_rows()}
        cur, tk, norms = None, [], []  # строки каскада идут по карточкам: храним токены только текущей
        idx, hyps, wins, spans, grantable = [], [], [], [], []
        rows = frame.select("sku_id", "category_id", "attribute", "kind", "value", "unit", "source").iter_rows()
        for i, (sku, cat, attr, kind, value, unit, source) in enumerate(rows):
            if not set(source.split("+")) <= JUDGED_SOURCES or sku not in texts:
                continue
            text, _ = texts[sku]
            if sku != cur:
                cur, tk = sku, tokens(text)
                norms = [t.norm for t in tk]
            slot = lexicon.slots.get(cat, {}).get(attr)
            span, all_found = _value_tokens(norms, attr, kind, value)
            ok = False
            if span is not None and kind == "bool":
                ok = (key(value) == "да") != _negated(tk, text, span)
            elif span is not None and all_found:
                ctx = _span_in_context(tk, text, norms, slot, kind, value, unit)
                ok, span = ctx is not None, ctx or span
            if span is None and slot is not None:
                span = _mention_tokens(tk, slot)
            if span is None:
                continue
            char = _char_span(tk, span)
            idx.append(i)
            hyps.append(hypothesis(attr, value, unit))
            wins.append(window(text, char))
            spans.append(char)
            grantable.append(ok)
        confirm = np.full(frame.height, np.nan)
        contra = np.full(frame.height, np.nan)
        if idx:
            p = self.probs(hyps, wins)
            confirm[idx], contra[idx] = p[:, CONFIRM], p[:, CONTRA]
        out = frame.with_columns(pl.Series("judge_confirm", confirm).fill_nan(None),
                                 pl.Series("judge_contra", contra).fill_nan(None))
        granted = 0
        if idx:
            # подтверждённым — текстовое подтверждение: источник judge и точный фрагмент поля карточки
            src, n_src = out["source"].to_list(), out["n_sources"].to_list()
            fld, ev = out["evidence_field"].to_list(), out["evidence_text"].to_list()
            skus = out["sku_id"].to_list()
            for i, span, ok in zip(idx, spans, grantable):
                if not ok or confirm[i] < self.min_confirm or "llm" in src[i].split("+"):
                    continue  # значение LLM судья только проверяет (вето в Stack), права на авто-публикацию не даёт
                text, n_title = texts[skus[i]]
                src[i], n_src[i] = src[i] + "+judge", n_src[i] + 1
                fld[i], ev[i] = _evidence(text, n_title, span)
                granted += 1
            out = out.with_columns(pl.Series("source", src), pl.Series("n_sources", n_src, dtype=out["n_sources"].dtype),
                                   pl.Series("evidence_field", fld), pl.Series("evidence_text", ev))
        self.report["last_annotate"] = {"rows": frame.height, "judged": len(idx), "granted": granted}
        return out

    # -------------------------------------------------------------- сохранение
    def save(self, path: Path) -> Path:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {}
        for k, v in self.net.state_dict().items():
            v = v.detach().cpu()
            half = v.to(torch.float16) if v.is_floating_point() else v
            state[k] = half if not v.is_floating_point() or torch.isfinite(half).all() else v.float()  # вне fp16 — fp32
        self.report.setdefault("id", uuid.uuid4().hex[:12])  # связывает судью с верификатором, обученным на его признаках
        tmp = path.with_suffix(".tmp")
        torch.save({"state": state, "backbone": self.backbone.name, "temperature": self.temperature, "max_len": self.max_len,
                    "min_confirm": self.min_confirm, "veto": self.veto, "report": self.report}, tmp)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path, backbone_path: Path, device: str = "cpu") -> Judge:
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=True)
        enc, tok, hidden, bb = load_backbone(blob["backbone"], backbone_path)
        net = make_net(enc, hidden)
        net.load_state_dict(blob["state"])  # copy_ приводит fp16 к fp32 по тензору, без полной копии в памяти
        return cls(net.to(device), tok, bb, blob["temperature"], device, blob["max_len"], report=blob.get("report", {}),
                   min_confirm=blob.get("min_confirm", MIN_CONFIRM), veto=blob.get("veto", VETO))

    @classmethod
    def new(cls, backbone: str, path: Path, device: str = "cuda", seed: int = 13) -> Judge:
        import torch

        enc, tok, hidden, bb = load_backbone(backbone, path)
        torch.manual_seed(seed)  # инициализация головы воспроизводима
        return cls(make_net(enc, hidden).to(device), tok, bb, device=device)


def evaluate_probs(p: np.ndarray, y: np.ndarray, min_confirm: float = MIN_CONFIRM, veto: float = VETO) -> dict:
    """Точность, macro-F1, log loss, ECE, точность по меткам и на рабочих точках каскада."""
    pred = p.argmax(1)
    conf = p.max(1)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs((pred[bins == b] == y[bins == b]).mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    per = {LABELS[c]: {"precision": round(float((y[pred == c] == c).mean()), 4) if (pred == c).any() else None,
                       "recall": round(float((pred[y == c] == c).mean()), 4) if (y == c).any() else None}
           for c in range(len(LABELS))}
    f1 = [2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"]) if v["precision"] and v["recall"] else 0.0
          for v in per.values()]
    ops = {}
    for c, name, t in ((CONFIRM, "confirm", min_confirm), (CONTRA, "veto", veto)):
        take = p[:, c] >= t
        ops[name] = {"share": round(float(take.mean()), 4),
                     "precision": round(float((y[take] == c).mean()), 4) if take.any() else None}
    return {"examples": int(len(y)), "accuracy": round(float((pred == y).mean()), 4), "macro_f1": round(float(np.mean(f1)), 4),
            "nll": round(float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean()), 4), "ece": round(float(ece), 4),
            "per_label": per, "operating": ops}


__all__ = ["Judge", "LABELS", "CONFIRM", "CONTRA", "NONE", "build_examples", "trivial_rule", "evaluate_probs", "locate",
           "mention", "window", "hypothesis", "card_full_text", "load_backbone", "JUDGED_SOURCES", "EXAMPLE_SCHEMA"]
