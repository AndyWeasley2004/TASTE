"""Evidence route: annotate one curated work with a local Qwen3.6 agent that searches and reads the web through the Jina MCP server."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pyarrow as pa

from taste.annotation.curated.common import CURATED_ROOT, read_parquet_rows, write_json, write_parquet

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]

DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_PROMPT = HERE / "prompt.txt"
ATTRIBUTES = json.loads((REPO / "attributes.json").read_text())
MOOD_IMAGERY_FEWSHOT = """Few-shot mood-imagery style examples (style only; never treat them as target evidence):
- calm -> Gazing out the window at the gentle afternoon sunlight in the classroom, chalk dust floating softly while distant voices blur into a warm quiet.
- sad -> Returning to the old house after leaving a few decades ago, touching the worn banister while late light falls across familiar rooms and half-remembered voices.
- happy -> Triumph after a great success, bright city lights reflecting on glass doors as cheers and laughter fill the street.
- anxious -> Waiting alone in a train station before an important departure, watching the clock hands jump forward while footsteps and announcements blur into nervous motion.
- angry -> Walking through rain-bright streets after a bitter argument, jaw tight and hands clenched as traffic lights flash against wet pavement.
Do not copy their openings, settings, actions, or wording. Rewrite any sentiment
that shares five consecutive words with an example; construct distinct imagery
from TARGET evidence."""
TRAINING_SCHEMA = pa.schema([
    pa.field("performance_id", pa.string()),
    pa.field("work_id", pa.string()),
    pa.field("midi_path", pa.string()),
    pa.field("prompt", pa.string()),
    pa.field("split", pa.string()),
    pa.field("source", pa.string()),
    pa.field("has_pedal", pa.bool_()),
    pa.field("duration_sec", pa.float64()),
])
NORMALIZATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "emotion": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "value": {"type": ["string", "null"], "enum": [*ATTRIBUTES["emotion"], None]},
                "status": {"type": "string", "enum": ["accepted", "abstain"]},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                "evidence_urls": {"type": "array", "items": {"type": "string"}},
                "support_excerpts": {"type": "array", "items": {"type": "string"}},
                "inference_scope": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["value", "status", "evidence_ids", "evidence_urls", "support_excerpts", "inference_scope", "rationale"],
        },
        "genre": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "value": {"type": ["string", "null"], "enum": [*ATTRIBUTES["genre"], None]},
                "status": {"type": "string", "enum": ["accepted", "abstain"]},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                "evidence_urls": {"type": "array", "items": {"type": "string"}},
                "support_excerpts": {"type": "array", "items": {"type": "string"}},
                "inference_scope": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["value", "status", "evidence_ids", "evidence_urls", "support_excerpts", "inference_scope", "rationale"],
        },
        "accompaniment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "value": {"type": ["string", "null"], "enum": [*ATTRIBUTES["accompaniment"], None]},
                "status": {"type": "string", "enum": ["accepted", "abstain"]},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                "evidence_urls": {"type": "array", "items": {"type": "string"}},
                "support_excerpts": {"type": "array", "items": {"type": "string"}},
                "inference_scope": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["value", "status", "evidence_ids", "evidence_urls", "support_excerpts", "inference_scope", "rationale"],
        },
        "performance": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "value": {"type": ["string", "null"], "enum": [*ATTRIBUTES["performance"], None]},
                "status": {"type": "string", "enum": ["accepted", "abstain"]},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                "evidence_urls": {"type": "array", "items": {"type": "string"}},
                "support_excerpts": {"type": "array", "items": {"type": "string"}},
                "inference_scope": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["value", "status", "evidence_ids", "evidence_urls", "support_excerpts", "inference_scope", "rationale"],
        },
        "sentiments": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "value": {
                    "type": ["string", "null"],
                    "minLength": 12,
                    "maxLength": 320,
                    "description": (
                        "One original target-specific 15-35 word mood-imagery sentence, "
                        "not a category, adjective list, or analysis summary."
                    ),
                },
                "status": {"type": "string", "enum": ["accepted", "abstain"]},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                "evidence_urls": {"type": "array", "items": {"type": "string"}},
                "support_excerpts": {"type": "array", "items": {"type": "string"}},
                "inference_scope": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["value", "status", "evidence_ids", "evidence_urls", "support_excerpts", "inference_scope", "rationale"],
        },
    },
    "required": ["emotion", "genre", "accompaniment", "performance", "sentiments"],
}


class ToolBudgetReached(RuntimeError):
    """Stop the research loop and finalize from evidence already collected."""


class JinaUsageExceeded(RuntimeError):
    """Stop annotation when Jina reports account-level quota exhaustion."""


def target_payload(work: dict[str, Any]) -> dict[str, Any]:
    source_records = []
    for record in work.get("source_records", []):
        metadata = record.get("metadata") or {}
        source_records.append({
            "source": record.get("source", ""),
            "source_id": record.get("source_id", ""),
            "metadata": {
                key: metadata[key]
                for key in ("name", "title", "artist", "composer", "composition", "movement", "album", "year")
                if metadata.get(key)
            },
        })
    payload = {
        "work_id": work["work_id"],
        "composer": work.get("composer", ""),
        "composition": work.get("composition", ""),
        "movement": work.get("movement", ""),
        "artist": work.get("artist", ""),
        "title": work.get("title", ""),
        "dataset_genre": work.get("genre", ""),
        "dataset_emotions": work.get("observed_emotions", []),
        "source_records": source_records,
    }
    if payload["composer"] and payload["composition"]:
        payload["search_hints"] = {
            "exact_target": " ".join(filter(None, (
                payload["composer"], payload["composition"], payload["movement"],
            ))),
            "parent_work": f'{payload["composer"]} {payload["composition"]}',
            "reputation": f'{payload["composer"]} {payload["composition"]} significance reception',
        }
    elif payload["artist"] and payload["title"]:
        payload["search_hints"] = {
            "exact_target": f'{payload["artist"]} {payload["title"]}',
            "lyrics": f'{payload["artist"]} {payload["title"]} lyrics',
            "reputation": f'{payload["artist"]} {payload["title"]} reception impact',
        }
    return payload


def load_work(root: Path, work_id: str | None) -> dict[str, Any]:
    if work_id is None:
        work_id = read_parquet_rows(root / "manifests/works.parquet")[0]["work_id"]
    path = root / "pieces" / work_id / "meta_archive" / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"unknown work: {work_id}")
    return json.loads(path.read_text())


def mcp_config(config: dict[str, Any]) -> dict[str, Any]:
    server: dict[str, Any] = {
        "type": "streamable-http",
        "url": config["jina_mcp_url"],
        "sse_read_timeout": config["timeout_seconds"],
    }
    jina_key = os.environ.get("JINA_API_KEY")
    if jina_key:
        server["headers"] = {"Authorization": f"Bearer {jina_key}"}
    return {"mcpServers": {"jina": server}}


def llm_config(config: dict[str, Any]) -> dict[str, Any]:
    # Qwen-Agent injects the MCP tool schema after applying max_input_tokens.
    # Reserve enough room for that schema and the requested completion.
    max_input_tokens = min(
        12000,
        int(config["context_length"]) - int(config["max_output_tokens"]) - 1600,
    )
    return {
        "model": config["model"],
        "model_server": config["base_url"],
        "api_key": config["api_key"],
        "generate_cfg": {
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
            "max_tokens": config["max_output_tokens"],
            "max_input_tokens": max_input_tokens,
            "fncall_prompt_type": "nous",
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": config["enable_thinking"],
                },
            },
        },
    }


def evidence_keywords(target: dict[str, Any]) -> list[str]:
    values = [target.get("movement", ""), target.get("title", "")]
    movement = str(target.get("movement", "")).lower()
    ordinal = {
        "i.": "first movement", "ii.": "second movement", "iii.": "third movement",
        "iv.": "fourth movement", "v.": "fifth movement", "vi.": "sixth movement",
    }
    values.extend(text for prefix, text in ordinal.items() if movement.startswith(prefix))
    generic = {"movement", "piano", "sonata", "major", "minor", "opus"}
    values.extend(
        token for token in re.findall(r"[\w\u4e00-\u9fff]{3,}", " ".join(values))
        if token.lower() not in generic
    )
    return list(dict.fromkeys(value.lower() for value in values if value))


def compact_tool_result(result: Any, kind: str, target: dict[str, Any], limit: int) -> Any:
    if not isinstance(result, str) or len(result) <= limit:
        return result
    if kind == "search":
        return result[:limit] + "\n[SEARCH_RESULTS_TRUNCATED_BY_HARNESS]"

    # Reader pages often contain an entire parent work: keep the page head plus
    # windows around the target movement/song.
    head_size = min(2500, limit // 3)
    chunks = [result[:head_size]]
    remaining = limit - head_size
    lowered = result.lower()
    positions: list[int] = []
    for keyword in evidence_keywords(target):
        start = 0
        while len(positions) < 8 and (index := lowered.find(keyword, start)) >= 0:
            positions.append(index)
            start = index + len(keyword)
    # `positions` is ordered by keyword specificity, and excerpts are taken in that order.
    for index in dict.fromkeys(positions):
        if remaining <= 200:
            break
        radius = min(1600, remaining // 2)
        chunk = result[max(0, index - radius):min(len(result), index + radius)]
        if chunk in chunks:
            continue
        chunks.append(chunk)
        remaining -= len(chunk)
    compacted = "\n\n[RELEVANT_EXCERPT]\n".join(chunks)
    return compacted[:limit] + "\n[PAGE_COMPACTED_BY_HARNESS]"


def tool_result_title(result: Any) -> str:
    if not isinstance(result, str):
        return ""
    for line in result.splitlines()[:20]:
        text = line.strip().lstrip("#").strip()
        if text.lower().startswith("title:"):
            return text.split(":", 1)[1].strip()[:300]
        if line.strip().startswith("#") and text:
            return text[:300]
    return ""


def identity_cautions(target: dict[str, Any], evidence_title: str) -> list[str]:
    """Flag unresolved identifier differences for the model and keep the evidence."""
    target_title = " ".join(str(target.get(key) or "") for key in ("composition", "movement", "title"))
    catalog_pattern = re.compile(r"\b(op|k|kv|wwv|bwv|d|sz)\.?\s*(\d+)\b", re.IGNORECASE)
    target_catalogs = {(prefix.lower(), number) for prefix, number in catalog_pattern.findall(target_title)}
    evidence_catalogs = {(prefix.lower(), number) for prefix, number in catalog_pattern.findall(evidence_title)}
    cautions = []
    for prefix, number in sorted(target_catalogs - evidence_catalogs):
        cautions.append(
            f"TARGET catalog {prefix.upper()} {number} is not confirmed by this record title; "
            "do not substitute another catalog with the same number."
        )

    number_pattern = re.compile(r"\bno\.?\s*(\d+)\b", re.IGNORECASE)
    target_numbers = set(number_pattern.findall(target_title))
    evidence_numbers = set(number_pattern.findall(evidence_title))
    if target_numbers and evidence_numbers and not target_numbers <= evidence_numbers:
        cautions.append(
            f"TARGET No. value(s) {sorted(target_numbers)} conflict with record-title No. value(s) "
            f"{sorted(evidence_numbers)}; verify their roles before using this evidence."
        )
    return cautions


def serialize_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return {
        key: value for key in ("role", "name", "content", "reasoning_content", "extra")
        if (value := getattr(message, key, None)) is not None
    }


def clean_sentence(value: Any, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[:limit]


def accepted_item(raw: dict[str, Any], field: str) -> str:
    if field == "sentiments":
        item = raw.get(field)
        if isinstance(item, str):
            return clean_sentence(item)
        if isinstance(item, list) and item and all(isinstance(x, str) for x in item):
            return clean_sentence(item[0])
        if isinstance(item, dict) and item.get("status") == "accepted":
            return clean_sentence(item.get("value"))
        return ""
    item = raw.get(field) or {}
    value = item.get("value")
    if item.get("status") != "accepted":
        return ""
    return value if value in ATTRIBUTES[field] else ""


def render_prompt(fields: dict[str, str]) -> str:
    lines = []
    for name in ("emotion", "genre", "accompaniment", "performance", "imagery"):
        value = fields.get(name)
        if value:
            lines.append(f"<{name}> {value}")
    return "\n".join(lines) + "\n\nMIDI:\n"


def normalize_annotation(raw: dict[str, Any] | None, work: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    raw = raw or {}
    accepted_fields: dict[str, str] = {}
    field_sources: dict[str, str] = {}
    if work.get("genre") in ATTRIBUTES["genre"]:
        accepted_fields["genre"] = work["genre"]
        field_sources["genre"] = "dataset_prior"
    observed_emotions = work.get("observed_emotions") or []
    if len(observed_emotions) == 1 and observed_emotions[0] in ATTRIBUTES["emotion"]:
        accepted_fields["emotion"] = observed_emotions[0]
        field_sources["emotion"] = "dataset_prior"
    # The model answers under the key "sentiments"; the label file stores it as "imagery".
    for field in ("emotion", "genre", "accompaniment", "performance", "sentiments"):
        label_field = "imagery" if field == "sentiments" else field
        if label_field not in accepted_fields:
            value = accepted_item(raw, field)
            if value:
                accepted_fields[label_field] = value
                field_sources[label_field] = "model_evidence"
    return {
        "work_id": work["work_id"],
        "status": "complete" if accepted_fields else "abstained",
        "model": config["model"],
        "revision": config["revision"],
        "accepted_fields": accepted_fields,
        "field_sources": field_sources,
        "prompt": render_prompt(accepted_fields),
        "model_output": raw,
    }


def raise_for_jina_usage_exceeded(messages: list[Any]) -> None:
    """Raise only for account-level exhaustion, not an individual page paywall."""
    usage_errors = (
        "quota exceeded", "exceeded your quota", "usage limit exceeded",
        "api usage exceeded", "insufficient credits", "insufficient balance",
        "credit balance exhausted", "token balance exhausted",
    )
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "function":
            continue
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        content = str(content or "").lower()
        if any(error in content for error in usage_errors):
            raise JinaUsageExceeded("Jina API usage quota exceeded")


def rebuild_training_manifest(root: Path) -> dict[str, int]:
    performances = read_parquet_rows(root / "manifests/performances.parquet")
    prompts = {}
    for work_dir in (root / "pieces").iterdir():
        path = work_dir / "label.json"
        if not path.exists():
            continue
        annotation = json.loads(path.read_text())
        if annotation.get("status") == "complete" and annotation.get("prompt"):
            prompts[work_dir.name] = annotation["prompt"]
    rows = [{
        "performance_id": row["performance_id"],
        "work_id": row["work_id"],
        "midi_path": row["midi_path"],
        "prompt": prompts[row["work_id"]],
        "split": row["split"],
        "source": row["source"],
        "has_pedal": row["has_pedal"],
        "duration_sec": row["duration_sec"],
    } for row in performances
        if row["work_id"] in prompts and row["midi_path"] and Path(row["midi_path"]).exists()]
    write_parquet(rows, root / "manifests/training_rows.parquet", TRAINING_SCHEMA)
    return {"annotated_works": len(prompts), "training_rows": len(rows)}


def build_agent(config: dict[str, Any], prompt: str, target: dict[str, Any]):
    try:
        from qwen_agent.agents import Assistant
        from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt
        from qwen_agent.log import logger as qwen_agent_logger
    except ImportError as exc:
        raise RuntimeError('install Qwen-Agent with: pip install -U "qwen-agent[mcp]"') from exc

    class RepairingNousFnCallPrompt(NousFnCallPrompt):
        """Canonicalize slightly malformed Qwen3 XML calls before parsing."""

        @staticmethod
        def _repair_block(match: re.Match[str]) -> str:
            body = match.group(1).strip()
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                    return f"<tool_call>\n{json.dumps(parsed, ensure_ascii=False)}\n</tool_call>"
            except json.JSONDecodeError:
                pass
            name_match = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)', body)
            if not name_match:
                return match.group(0)
            arguments: dict[str, Any] = {}
            args_match = re.search(r'["\']arguments["\']\s*:', body)
            if args_match:
                tail = body[args_match.end():].lstrip()
                brace = tail.find("{")
                if brace >= 0:
                    try:
                        candidate, _ = json.JSONDecoder().raw_decode(tail[brace:])
                        if isinstance(candidate, dict):
                            arguments = candidate
                    except json.JSONDecodeError:
                        pass
            if not arguments:
                for key in ("query", "url", "href"):
                    value = re.search(rf'["\']{key}["\']\s*:\s*["\']([^"\']+)', body)
                    if value:
                        arguments[key] = value.group(1)
                        break
            payload = {"name": name_match.group(1), "arguments": arguments}
            return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"

        def postprocess_fncall_messages(self, messages, **kwargs):
            messages = copy.deepcopy(messages)
            pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
            for message in messages:
                for item in message.content or []:
                    if getattr(item, "text", None):
                        item.text = pattern.sub(self._repair_block, item.text)
            return super().postprocess_fncall_messages(messages, **kwargs)
    class BudgetedAssistant(Assistant):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tool_counts = {"search": 0, "read": 0, "expand": 0}
            self.budget_redirects = {"search": 0, "read": 0, "expand": 0}
            self.tool_cache_hits = 0
            self._tool_cache: dict[tuple[str, str], Any] = {}
            self.evidence_records: list[dict[str, Any]] = []

        def _call_tool(self, tool_name, tool_args="{}", **kwargs):
            # Qwen occasionally leaves a trailing XML tag or brace around an
            # otherwise valid argument object. Canonicalize it once so the MCP
            # call and every later prompt turn receive strict JSON.
            if isinstance(tool_args, str):
                candidate = tool_args.strip().removeprefix("```json").removesuffix("```").strip()
                decoder = json.JSONDecoder()
                try:
                    arguments, _ = decoder.raw_decode(candidate[candidate.find("{"):])
                except (json.JSONDecodeError, ValueError):
                    arguments = None
                if not isinstance(arguments, dict):
                    arguments = {}
                    for key in ("query", "url", "href"):
                        match = re.search(rf'["\']{key}["\']\s*:\s*["\']([^"\']+)', candidate)
                        if match:
                            arguments[key] = match.group(1)
                            break
                tool_args = json.dumps(arguments, ensure_ascii=False)
                for message in reversed(kwargs.get("messages") or []):
                    function_call = getattr(message, "function_call", None)
                    if function_call and function_call.name == tool_name:
                        function_call.arguments = tool_args
                        break
            kind = "expand" if tool_name.endswith("expand_query") else "read" if tool_name.endswith("read_url") else "search"
            limits = {
                "search": config["max_search_calls"],
                "read": config["max_read_calls"],
                "expand": config["max_query_expansions"],
            }
            cache_key = (tool_name, tool_args)
            if cache_key in self._tool_cache:
                self.tool_cache_hits += 1
                return self._tool_cache[cache_key]
            if self.tool_counts[kind] >= limits[kind]:
                if self.budget_redirects[kind] == 0:
                    self.budget_redirects[kind] += 1
                    if kind == "search":
                        return (
                            "SEARCH_BUDGET_REACHED. Do not search again. Read one of the strongest URLs "
                            "already found if nuanced evidence is needed; otherwise return the final JSON."
                        )
                    return f"{kind.upper()}_BUDGET_REACHED. Return the final JSON from collected evidence."
                raise ToolBudgetReached(kind)
            self.tool_counts[kind] += 1
            result = super()._call_tool(tool_name, tool_args, **kwargs)
            result = compact_tool_result(
                result, kind, target, int(config["max_tool_result_chars"]),
            )
            if kind in {"search", "read"}:
                evidence_id = f"ev_jina_{len(self.evidence_records) + 1:02d}"
                try:
                    arguments = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                except json.JSONDecodeError:
                    arguments = {"raw": str(tool_args)}
                url = ""
                if isinstance(arguments, dict):
                    url = str(arguments.get("url") or arguments.get("href") or "")
                query = str(arguments.get("query") or "") if isinstance(arguments, dict) else ""
                record_title = tool_result_title(result) or (
                    f"Search results for: {query}" if query else "Jina search results"
                )
                record = {
                    "evidence_id": evidence_id,
                    "source_type": f"jina_{kind}",
                    "title": record_title,
                    "url": url,
                    "tool_name": tool_name,
                    "tool_arguments": arguments,
                }
                if cautions := identity_cautions(target, record_title):
                    record["identity_cautions"] = cautions
                self.evidence_records.append(record)
                result = (
                    "EVIDENCE_RECORD\n"
                    + json.dumps(record, ensure_ascii=False, indent=2)
                    + "\nIDENTITY_POLICY\nCandidate evidence only. Match TARGET creator, work, "
                    "catalog/opus, and movement or track number; a similar title is insufficient.\n"
                    + "\nCONTENT\n"
                    + str(result)
                )
            self._tool_cache[cache_key] = result
            return result

    agent = BudgetedAssistant(
        llm=llm_config(config),
        system_message=prompt,
        function_list=[mcp_config(config)],
    )
    qwen_agent_logger.setLevel("WARNING")
    agent.llm.fncall_prompt = RepairingNousFnCallPrompt()
    return agent


def extract_final_json(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            continue
        content = message["content"].strip()
        fenced = re.findall(r"```json\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
        if fenced:
            for block in reversed(fenced):
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    pass
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            for index in reversed([i for i, char in enumerate(content) if char == "{"]):
                try:
                    candidate = json.loads(content[index:].strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    return candidate
    return None


def trace_for_normalization(messages: list[dict[str, Any]], max_chars: int = 12000) -> str:
    # Truncation keeps the newest messages, where the read results and conclusions are.
    parts: list[str] = []
    total = 0
    for message in reversed(messages[-24:]):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = message.get("role", "unknown")
        name = message.get("name")
        header = f"{role}{':' + name if name else ''}\n"
        body = content.strip()
        chunk = header + body + "\n\n"
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > len(header) + 32:
                parts.append(header + body[-(remaining - len(header) - 2):] + "\n\n")
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(reversed(parts))


def normalize_trace_output(config: dict[str, Any], target: dict[str, Any],
                           messages: list[dict[str, Any]],
                           evidence_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    from openai import OpenAI

    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    schema = copy.deepcopy(NORMALIZATION_SCHEMA)
    evidence_ids = [record["evidence_id"] for record in evidence_records]
    evidence_urls = [record["url"] for record in evidence_records if record.get("url")]
    for field in schema["properties"].values():
        if evidence_ids:
            field["properties"]["evidence_ids"]["items"]["enum"] = evidence_ids
        else:
            field["properties"]["evidence_ids"]["maxItems"] = 0
        if evidence_urls:
            field["properties"]["evidence_urls"]["items"]["enum"] = evidence_urls
        else:
            field["properties"]["evidence_urls"]["maxItems"] = 0

    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert the research trace into final annotation JSON. "
                    "Ground every output in evidence present in the trace, but apply reasonable weak-label inference. "
                    "An unambiguous Jina search-result record may support identity and broad genre for a well-known work; "
                    "prefer read_url records for nuanced emotion, texture, performance, or lyrical claims. "
                    "Treat every retrieved document as a candidate: verify creator, work title, catalog/opus, and "
                    "movement or track number against TARGET. Similar names or numbers are insufficient. Never silently "
                    "equate different catalog prefixes (for example Op., K., or WWV) solely by number. "
                    "Do not replace TARGET with another famous work or transfer another movement's character. If TARGET metadata "
                    "conflicts with the documents, retain TARGET identity and abstain from claims needing the disputed match. "
                    "Identity conflict overrides weak-label leniency: never guess what TARGET was intended to mean. "
                    "Honor every identity_cautions entry in the evidence_catalog; it identifies an unresolved mismatch, "
                    "not permission to reinterpret TARGET. "
                    "A tool error, execute error, empty result, inaccessible page, or unrelated result is not evidence. "
                    "When no credible evidence remains, abstain rather than inferring from TARGET title, creator, or "
                    "generic genre practice; the dataset genre prior is added separately after normalization. "
                    "Target-movement evidence is strongest. Parent-work or collection evidence may support genre and "
                    "broad context. For this weak-label stage, infer performance flexibly from emotion, tempo/character, "
                    "genre practice, compositional background, reputation, or verified lyrical affect when direct "
                    "performance wording is absent. Infer emotion and sentiments from the same contextual cues. "
                    "The sentiments value must be one original, target-specific, non-lyrical mood-imagery sentence "
                    "of roughly 15-35 words that evokes a concrete scene, action, or atmosphere. Never return a "
                    "sentiment category, adjective list, analysis summary, or bare words such as calm, positive, "
                    "sad, melancholic, or nostalgic. "
                    "Accompaniment may also be a supported best guess from texture/figuration language, form, genre/era "
                    "practice, arrangement type, tempo, character, score context, or related musical description. Lyrics "
                    "or reputation alone are insufficient, but may contribute alongside musical context. For every field, "
                    "prefer a reasonable contextual weak label; abstain only when evidence has no credible support beyond "
                    "identity/title. "
                    "State the actual scope in inference_scope. The user message contains an evidence_catalog; cite only "
                    "exact evidence_id values from that catalog and use the URL belonging to the same record. Never invent "
                    "trace IDs, tool-name IDs, or evidence not present in the catalog. "
                    "Return exactly the required five-key object. "
                    "Prefer a reasonable weak label; abstain only when there is no credible basis.\n\n"
                    + MOOD_IMAGERY_FEWSHOT
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "target": target,
                    "allowed_values": ATTRIBUTES,
                    "evidence_catalog": evidence_records,
                    "trace": trace_for_normalization(messages),
                }, ensure_ascii=False, indent=2),
            },
        ],
        temperature=0.0,
        max_tokens=2000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "curated_annotation", "strict": True, "schema": schema},
        },
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response.choices[0].message.content
    if not content:
        return None
    return extract_final_json([{"role": "assistant", "content": content}])


def annotate_agentic_work(root: Path, config: dict[str, Any], prompt: str,
                          work_id: str, overwrite: bool = False) -> Path:
    """Research and annotate one work through Qwen-Agent and Jina MCP."""
    work = load_work(root, work_id)
    if work.get("labeling_route") == "multimodal_required":
        raise RuntimeError(f"{work_id} is routed to the audio route")
    target = target_payload(work)
    work_dir = root / "pieces" / work_id
    output_path = work_dir / "meta_archive" / "agent_trace.json"
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    agent = build_agent(config, prompt, target)
    response = []
    try:
        for response in agent.run(messages=[{
            "role": "user",
            "content": "TARGET\n" + json.dumps(target, ensure_ascii=False, indent=2),
        }]):
            pass
    except ToolBudgetReached:
        pass
    serialized = [serialize_message(message) for message in response]
    raise_for_jina_usage_exceeded(serialized)
    final_json = extract_final_json(serialized)
    if final_json is None:
        final_json = normalize_trace_output(config, target, serialized, agent.evidence_records)
    if final_json is None:
        raise RuntimeError("agent did not produce an annotation object")
    write_json(output_path, {
        "revision": config["revision"], "model": config["model"],
        "mode": "jina_mcp_agentic", "target": target,
        "elapsed_seconds": round(time.time() - started, 3),
        "tool_counts": agent.tool_counts, "budget_redirects": agent.budget_redirects,
        "tool_cache_hits": agent.tool_cache_hits,
        "evidence_records": agent.evidence_records,
        "messages": serialized, "annotation": final_json,
    })
    write_json(work_dir / "label.json", normalize_annotation(final_json, work, config))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CURATED_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--base-url", help="Override the configured local model server URL")
    parser.add_argument("--work-id")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-training-rows", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if args.base_url:
        config["base_url"] = args.base_url
    prompt = args.prompt_file.read_text()
    if args.rebuild_training_rows:
        print(json.dumps(rebuild_training_manifest(args.root), ensure_ascii=False, indent=2))
        return

    work = load_work(args.root, args.work_id)
    target = target_payload(work)
    mcp = mcp_config(config)
    if "headers" in mcp["mcpServers"]["jina"]:
        mcp["mcpServers"]["jina"]["headers"] = {"Authorization": "Bearer <redacted>"}
    preview = {
        "config": {
            key: value for key, value in config.items()
            if key != "api_key"
        },
        "mcp": mcp,
        "system_prompt": prompt,
        "target": target,
    }
    if args.preview:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    if work.get("labeling_route") == "multimodal_required":
        raise RuntimeError(
            f"{work['work_id']} lacks searchable identity metadata and is routed to the audio route"
        )

    print(annotate_agentic_work(
        args.root, config, prompt, work["work_id"], overwrite=args.overwrite,
    ))


if __name__ == "__main__":
    main()
