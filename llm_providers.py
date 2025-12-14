from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from pdf2image import convert_from_bytes  # optional, used for PDF->image vision
except Exception:
    convert_from_bytes = None  # type: ignore

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


UNIVERSAL_TABLE_PROMPT = """You are an expert data extraction AI. I will provide you with a table-based quote sheet,
possibly in Markdown, CSV, plain text, or other formats.

Your job is to:
- Parse the entire input regardless of format.
- Identify all quotes labeled â€œQuote 1â€, â€œQuote 2â€, etc. OR if no labels exist, treat each distinct table block as a separate quote.
- For each quote, extract every single field shown in every row (including header rows and subsequent rows).
- Do NOT invent keys or groupings â€” extract exactly whatâ€™s there.
- If multiple rows exist for a quote, combine them into one dictionary per quote (flatten; no nesting).

Output rules:
- Output ONLY valid JSON (no markdown, no code blocks, no commentary).
- Return a JSON ARRAY of objects: [ { ... }, { ... } ].
- Booleans must be true/false, numbers must be numbers, strings preserve casing.
"""

VISION_TO_MARKDOWN_PROMPT = (
    "Extract ALL tables from this document. Output ONLY GitHub-flavored Markdown tables. "
    "Preserve headers and cell values exactly. If multiple tables exist, output them separated by a blank line. "
    "If NO tables are found, output exactly: NO_TABLES. "
    "No commentary."
)


def _extract_first_json_object(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    start = raw.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : i + 1]
    return ""


def _extract_first_json_array(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    start = raw.find("[")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return raw[start : i + 1]
    return ""


def _parse_quotes_from_raw(raw: str) -> List[Dict[str, Any]]:
    """
    Accepts either:
      - {"quotes":[{...},{...}]}
      - [{...},{...}]
    And also tolerates extra text by extracting the first JSON object/array.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty_llm_response")

    # Prefer object form if present
    obj_str = _extract_first_json_object(raw)
    if obj_str:
        obj = json.loads(obj_str)
        if isinstance(obj, dict) and isinstance(obj.get("quotes"), list):
            return [x for x in obj["quotes"] if isinstance(x, dict)]

    arr_str = _extract_first_json_array(raw)
    if arr_str:
        arr = json.loads(arr_str)
        if isinstance(arr, list):
            return [x for x in arr if isinstance(x, dict)]

    # Last attempt: direct parse
    parsed = json.loads(raw)
    if isinstance(parsed, dict) and isinstance(parsed.get("quotes"), list):
        return [x for x in parsed["quotes"] if isinstance(x, dict)]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]

    raise ValueError("no_quotes_json_found")


@dataclass
class Providers:
    openai_enabled: bool
    openai_text_model: str
    openai_vision_model: str

    hf_token: Optional[str]
    hf_vision_model: str

    openai_vision_input_per_m: float
    openai_vision_output_per_m: float
    openai_text_input_per_m: float
    openai_text_output_per_m: float
    hf_local_vl_model: Optional[str]
    hf_local_vl_device: Optional[str]

    # Lazy cache for local HF vision model (processor, model, device)
    _hf_local_vl_cache: Optional[Tuple[Any, Any, str]] = None

    @property
    def vision_provider_name(self) -> str:
        if self.openai_enabled:
            return f"openai:{self.openai_vision_model}"
        return f"hf:{self.hf_vision_model}"

    @property
    def text_provider_name(self) -> str:
        if self.openai_enabled:
            return f"openai:{self.openai_text_model}"
        return "none"

    @property
    def hf_enabled(self) -> bool:
        return bool(self.hf_token or self.hf_local_vl_model)

    @staticmethod
    def from_env() -> "Providers":
        openai_enabled = os.getenv("OPENAI_ENABLED", "0") == "1"
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            hf_token = hf_token.strip()
        hf_model = os.getenv("HF_VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct:novita")
        if hf_model:
            hf_model = hf_model.strip()
        return Providers(
            openai_enabled=openai_enabled,
            openai_text_model=os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"),
            openai_vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o"),
            hf_token=hf_token,
            hf_vision_model=hf_model,
            openai_vision_input_per_m=float(os.getenv("OPENAI_VISION_INPUT_PER_M", "0.25")),
            openai_vision_output_per_m=float(os.getenv("OPENAI_VISION_OUTPUT_PER_M", "2.00")),
            openai_text_input_per_m=float(os.getenv("OPENAI_TEXT_INPUT_PER_M", "0.25")),
            openai_text_output_per_m=float(os.getenv("OPENAI_TEXT_OUTPUT_PER_M", "2.00")),
            hf_local_vl_model=os.getenv("HF_LOCAL_VL_MODEL"),
            hf_local_vl_device=os.getenv("HF_LOCAL_VL_DEVICE"),
        )

    def is_reasoning_model(self, model: str) -> bool:
        lower_model = model.lower()
        return 'o1' in lower_model or 'gpt-5' in lower_model

    def accumulate_usage(self, agg: Dict[str, Any], usage: Optional[Dict[str, Any]]) -> None:
        if not usage:
            return
        # Handle nested/extra usage entries (e.g., OpenAI attempt + fallback provider)
        extras = usage.get("extra_usages") if isinstance(usage, dict) else None
        if isinstance(extras, list):
            for u in extras:
                if isinstance(u, dict):
                    self.accumulate_usage(agg, u)
        prov = usage.get("provider")
        if prov == "openai":
            agg["openai"]["calls"] += 1
            input_tokens = int(usage.get("input_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))
            agg["openai"]["input_tokens"] += input_tokens
            agg["openai"]["output_tokens"] += output_tokens

            # Determine which model was used and apply correct pricing
            model = usage.get("model", "")
            if model == self.openai_vision_model:
                # Vision model pricing
                in_cost = (input_tokens / 1_000_000.0) * self.openai_vision_input_per_m
                out_cost = (output_tokens / 1_000_000.0) * self.openai_vision_output_per_m
            else:
                # Text model pricing (default)
                in_cost = (input_tokens / 1_000_000.0) * self.openai_text_input_per_m
                out_cost = (output_tokens / 1_000_000.0) * self.openai_text_output_per_m

            # Accumulate cost
            current_cost = agg["openai"].get("estimated_cost_usd", 0.0)
            agg["openai"]["estimated_cost_usd"] = round(current_cost + in_cost + out_cost, 6)
        elif prov == "hf":
            agg["hf"]["calls"] += 1

    def _openai_client(self) -> OpenAI:
        if OpenAI is None:
            raise RuntimeError("openai package not installed")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY missing")
        return OpenAI()


    # Text model: universal parse
    def universal_parse_rows(self, text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Avoid paying for / failing on empty inputs (common when vision returns no tables).
        if not (text or "").strip():
            return [], {"provider": "none", "warning": "empty_input"}
        if self.openai_enabled:
            return self._openai_universal_parse(text)
        raise RuntimeError("openai_text_disabled_no_fallback_configured")

    def _openai_universal_parse(self, text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        client = self._openai_client()
        is_reasoning = self.is_reasoning_model(self.openai_text_model)

        # Adjust prompt for reasoning models
        if is_reasoning:
            prompt = (
                UNIVERSAL_TABLE_PROMPT
                + "\n\nIMPORTANT: Output ONLY a valid JSON array of quote objects. No other text.\n"
                + "Format: [ { ... }, { ... } ]\n\nINPUT:\n"
                + text
            )
        else:
            prompt = (
                UNIVERSAL_TABLE_PROMPT
                + "\n\nIMPORTANT: Return a JSON OBJECT with one key: quotes.\n"
                + "Format: {\"quotes\": [ { ... }, { ... } ] }\n\nINPUT:\n"
                + text
            )

        def usage_of(resp) -> Dict[str, Any]:
            usage = resp.usage or None
            return {
                "provider": "openai",
                "model": self.openai_text_model,
                "input_tokens": int(usage.prompt_tokens if usage else 0),
                "output_tokens": int(usage.completion_tokens if usage else 0),
            }

        max_param = 'max_completion_tokens' if is_reasoning else 'max_tokens'
        kwargs: Dict[str, Any] = {max_param: 2400}
        if not is_reasoning:
            kwargs['response_format'] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=self.openai_text_model,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt + "\n\nReturn VALID JSON only."}]}],
                    **kwargs
                )
                raw = resp.choices[0].message.content or ""
                rows = _parse_quotes_from_raw(raw)
                return rows, usage_of(resp)
            except Exception as e:
                last_err = e
                attempt += 1

        raise last_err or RuntimeError("openai_universal_parse_failed")


    # Vision model: document -> markdown tables
    def vision_to_markdown_from_pdf(self, pdf_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Render PDF pages to images, then reuse vision_to_markdown_from_image per page.
        Useful when OpenAI vision is unavailable (reasoning model) and HF vision is the fallback.
        """
        if convert_from_bytes is None:
            raise RuntimeError("pdf2image not installed or poppler unavailable for PDF rendering")

        max_pages = int(os.getenv("PDF_VISION_PAGES", "2"))
        dpi = int(os.getenv("PDF_VISION_DPI", "220"))
        pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=max_pages)

        mds: List[str] = []
        primary_usage: Optional[Dict[str, Any]] = None

        for img in pages:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            md, usage = self.vision_to_markdown_from_image(buf.getvalue())
            if md.strip():
                mds.append(md.strip())

            if usage:
                if primary_usage is None:
                    primary_usage = usage
                else:
                    extras = primary_usage.setdefault("extra_usages", [])
                    if isinstance(extras, list):
                        extras.append(usage)

        combined_md = "\n\n".join(mds).strip()
        return combined_md, primary_usage or {"provider": "hf", "model": self.hf_vision_model}

    def vision_to_markdown_from_image(self, image_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Preferred flow: try OpenAI vision first (if supported), then fall back to HF if OpenAI returns empty or raises.
        This avoids dropping to text-only parsing with empty vision output.
        """
        openai_usage: Optional[Dict[str, Any]] = None

        # --- OpenAI first (if enabled) ---
        if self.openai_enabled and not self.is_reasoning_model(self.openai_vision_model):
            try:
                md, openai_usage = self._openai_image_to_md(image_bytes)
                if (md or "").strip():
                    return md, openai_usage
            except Exception as exc:
                openai_usage = {
                    "provider": "openai",
                    "model": self.openai_vision_model,
                    "error": str(exc),
                }

            # Fallback on empty/error is enabled by default
            if os.getenv("VISION_FALLBACK_ON_EMPTY", "1") != "1":
                return "", (openai_usage or {"provider": "openai", "model": self.openai_vision_model})

        # --- Fallback to HF (local or hosted) ---
        try:
            if self.hf_local_vl_model:
                md, usage = self._hf_local_image_to_md(image_bytes)
            else:
                md = self._hf_image_to_md(image_bytes)
                usage = {"provider": "hf", "model": self.hf_vision_model}
            if openai_usage:
                usage["extra_usages"] = [openai_usage]
            return md, usage
        except Exception as exc2:
            u = openai_usage or {"provider": "hf", "model": self.hf_vision_model}
            u["fallback_error"] = str(exc2)
            return "", u

    def _openai_image_to_md(self, image_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
        client = self._openai_client()
        is_reasoning = self.is_reasoning_model(self.openai_vision_model)
        if is_reasoning:
            raise RuntimeError("Reasoning models like o1/gpt-5 do not support vision tasks")
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        max_param = 'max_completion_tokens' if is_reasoning else 'max_tokens'
        resp = client.chat.completions.create(
            model=self.openai_vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_TO_MARKDOWN_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            **{max_param: 2600},
        )
        usage = resp.usage or None
        u = {
            "provider": "openai",
            "model": self.openai_vision_model,
            "input_tokens": int(usage.prompt_tokens if usage else 0),
            "output_tokens": int(usage.completion_tokens if usage else 0),
        }
        content = resp.choices[0].message.content or ""
        # Fallback: if OpenAI vision returns empty, try local HF VL model if configured
        if not content.strip() and self.hf_local_vl_model:
            try:
                fallback_md, fallback_usage = self._hf_local_image_to_md(image_bytes)
                fallback_usage["extra_usages"] = [u]
                return fallback_md, fallback_usage
            except Exception as exc:  # best-effort fallback; keep original empty if it fails
                u["fallback_error"] = str(exc)[:200]
        return content, u

    def _hf_local_image_to_md(self, image_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Local fallback using a Hugging Face vision-language model (e.g., Qwen2.5-VL-7B-Instruct).
        Only used when OpenAI returns an empty response.
        """
        if not self.hf_local_vl_model:
            raise RuntimeError("hf_local_vl_model_not_configured")
        try:
            import io
            from PIL import Image
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except Exception as exc:
            raise RuntimeError("transformers/torch/Pillow missing; pip install transformers torch Pillow") from exc

        if self._hf_local_vl_cache is None:
            device = self.hf_local_vl_device or ("cuda" if torch.cuda.is_available() else "cpu")
            processor = AutoProcessor.from_pretrained(self.hf_local_vl_model)
            model = AutoModelForVision2Seq.from_pretrained(self.hf_local_vl_model)
            model.to(device)
            self._hf_local_vl_cache = (processor, model, device)

        processor, model, device = self._hf_local_vl_cache
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": VISION_TO_MARKDOWN_PROMPT},
            ],
        }]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(**inputs, max_new_tokens=2600)
        text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]).strip()

        return text, {"provider": "hf_local", "model": self.hf_local_vl_model, "device": device}

    def _hf_image_to_md(self, image_bytes: bytes) -> str:
        if not self.hf_token:
            raise RuntimeError("HF_TOKEN required for VISION_PROVIDER=hf")
        try:
            from huggingface_hub import InferenceClient
        except Exception as exc:
            raise RuntimeError("huggingface_hub missing; pip install huggingface_hub") from exc

        client = InferenceClient(api_key=self.hf_token)

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = client.chat.completions.create(
            model=self.hf_vision_model,  # Supports "Qwen/Qwen3-VL-8B-Instruct" format
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_TO_MARKDOWN_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            max_tokens=2400,
        )
        return (resp.choices[0].message.content or "").strip()