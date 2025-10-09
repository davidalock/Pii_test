"""
Ollama Integration for Presidio PII Analysis
This module provides functionality to use local Ollama models (like Mistral)
for enhanced entity recognition with Microsoft Presidio.
"""

import logging
import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import OrderedDict
import threading
import time
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger("presidio-ollama-integration")

class OllamaEntityExtractor:
    """
    A class to extract entities from text using Ollama models.
    
    This class connects to a local Ollama instance to use models like Mistral
    for extracting entities from text through prompt engineering.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2:latest",
        api_url: str = "http://localhost:11434/api/generate",
        prompt_template: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        temperature: float = 0.0,
        # Back-compat: max_tokens maps to Ollama options.num_predict
        max_tokens: int = 192,
        # New low-level runtime options
        num_thread: int = 6,
        num_ctx: int = 1024,
        num_predict: Optional[int] = None,
        http_pool_maxsize: int = 8,
        request_timeout_seconds: int = 60,
        keep_alive: str = "30m",
    max_concurrent_requests: int = 4,
        # New tuning knobs
        retry_on_parse_fail: bool = True,
        autopull_model: bool = False,
        warm_up_on_init: bool = False,
        # Simple auto-tuning: drop threads from 6 to 4 if latency repeatedly high
        adapt_threads_on_wobble: bool = True,
        wobble_high_ms: int = 6000,
    ):
        """Initialize the Ollama entity extractor."""

        # Allow environment overrides for quick tuning
        env_model = os.getenv("OLLAMA_MODEL")
        env_temp = os.getenv("OLLAMA_TEMPERATURE")
        env_tokens = os.getenv("OLLAMA_MAX_TOKENS")  # back-compat alias for num_predict
        env_predict = os.getenv("OLLAMA_NUM_PREDICT")
        env_keep = os.getenv("OLLAMA_KEEP_ALIVE")
        env_conc = os.getenv("OLLAMA_MAX_CONCURRENCY")
        env_debug = os.getenv("OLLAMA_DEBUG")
        env_threads = os.getenv("OLLAMA_NUM_THREAD")
        env_ctx = os.getenv("OLLAMA_NUM_CTX")
        env_adapt = os.getenv("OLLAMA_ADAPT_THREADS")

        self.model_name = env_model or model_name
        self.api_url = api_url
        self.temperature = float(env_temp) if env_temp is not None else temperature
        # Determine num_predict (priority: explicit env -> explicit arg -> legacy env -> legacy arg)
        if env_predict is not None:
            self.num_predict = int(env_predict)
        elif num_predict is not None:
            self.num_predict = int(num_predict)
        elif env_tokens is not None:
            self.num_predict = int(env_tokens)
        else:
            self.num_predict = int(max_tokens)
        # Retain for backward compatibility (not used directly in payload)
        self.max_tokens = int(self.num_predict)
        self.keep_alive = env_keep or keep_alive
        self.request_timeout_seconds = request_timeout_seconds
        self.retry_on_parse_fail = retry_on_parse_fail
        self.autopull_model = autopull_model
        self._debug_enabled = (str(env_debug).lower() in ("1", "true", "yes")) if env_debug is not None else False
        self.num_thread = int(env_threads) if env_threads is not None else int(num_thread)
        self.num_ctx = int(env_ctx) if env_ctx is not None else int(num_ctx)
        self._adapt_threads_on_wobble = (str(env_adapt).lower() in ("1","true","yes")) if env_adapt is not None else bool(adapt_threads_on_wobble)
        self._wobble_high_ms = int(wobble_high_ms)
        self._wobble_high_count = 0
        self._threads_dropped = False

        # Apply concurrency override if set via env
        if env_conc is not None:
            try:
                max_concurrent_requests = int(env_conc)
            except Exception:
                pass

        # Create a persistent session for HTTP connection reuse
        self.session = requests.Session()
        # Configure connection pooling and retries for concurrency and resilience
        retries = Retry(total=2, backoff_factor=0.2, status_forcelist=(502, 503, 504))
        adapter = HTTPAdapter(
            pool_connections=http_pool_maxsize,
            pool_maxsize=http_pool_maxsize,
            max_retries=retries,
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        # Tiny LRU cache for repeated texts (helps UI scenarios)
        self._cache_max = 128
        self._cache = OrderedDict()

        # Optional concurrency limiter for heavy local models
        self._sem = threading.Semaphore(max(1, int(max_concurrent_requests)))
        self._max_concurrency = max(1, int(max_concurrent_requests))
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._req_seq = 0
        self._seq_lock = threading.Lock()

        # Optionally ensure model exists (auto-pull) and warm up
        if self.autopull_model:
            try:
                self._ensure_model_available()
            except Exception as e:
                logger.warning(f"Ollama autopull skipped/failed: {e}")
        if warm_up_on_init:
            try:
                self.warm_up()
            except Exception as e:
                logger.warning(f"Ollama warm-up skipped/failed: {e}")

        # Set default entity types if not provided
        if entity_types is None:
            self.entity_types = [
                "PERSON", "ORGANIZATION", "LOCATION", "DATE_TIME",
                "EMAIL_ADDRESS", "PHONE_NUMBER", "ADDRESS", "CREDIT_CARD",
                "BANK_ACCOUNT", "PASSPORT_NUMBER", "ID_NUMBER", "UK_NI_NUMBER",
                "UK_NHS_NUMBER",
            ]
        else:
            self.entity_types = entity_types

        # Set default prompt template if not provided
        if prompt_template is None:
            self.prompt_template = (
                """
You are an expert PII entity extractor.
Extract only these entity types: {entity_types}

Return your entire answer as a single valid JSON array (and nothing else). Do not include explanations or code fences.
Each array item must be an object with keys: entity_type, text, start, end, confidence.
- entity_type: one of the requested types
- text: exact substring from the input
- start: integer start character offset (0-based)
- end: integer end character offset (exclusive)
- confidence: number in [0,1]

If no entities are present, return []

Input:
{text}
"""
            )
        else:
            self.prompt_template = prompt_template
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from the given text using the Ollama model.
        
        Args:
            text: The text to analyze
            
        Returns:
            A list of extracted entities with their details
        """
        # Format the prompt
        entity_types_str = ", ".join(self.entity_types)
        
        # Debug logging for UI mode
        if hasattr(self, '_debug_ui_mode') and self._debug_ui_mode:
            logger.error(f"UI DEBUG - Entity types string: {entity_types_str}")
            logger.error(f"UI DEBUG - Template: {repr(self.prompt_template[:200])}")
            
        # Important: Do NOT use str.format on the full template, as the JSON example
        # within the template contains braces that will be treated as placeholders.
        # Instead, perform targeted replacements for the two supported tokens.
        try:
            prompt = (
                self.prompt_template
                .replace("{entity_types}", entity_types_str)
                .replace("{text}", text)
            )
        except Exception as e:
            logger.error(f"UI DEBUG - Error during prompt assembly: {e}")
            logger.error(f"UI DEBUG - Template content: {repr(self.prompt_template)}")
            raise
        
        # Return from cache if seen recently
        try:
            cached = self._cache.get(text)
            if cached is not None:
                # Move to end to mark as recently used
                self._cache.move_to_end(text)
                return cached
        except Exception:
            pass

        # Prepare the request payload for generate API
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
            "format": "json",
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": self.num_predict,
                "num_thread": self.num_thread,
                "num_ctx": self.num_ctx,
            },
        }

        try:
            # Concurrency diagnostics
            t_queue_start = time.perf_counter()
            with self._seq_lock:
                self._req_seq += 1
                req_id = self._req_seq

            thread = threading.current_thread()
            thread_info = f"{thread.name}:{thread.ident}"
            logger.info(f"Ollama CALL QUEUED req={req_id} thread={thread_info} max_conc={self._max_concurrency}")

            # Make the API request using persistent session with concurrency cap
            with self._sem:
                t_acquired = time.perf_counter()
                queued_ms = int((t_acquired - t_queue_start) * 1000)
                with self._inflight_lock:
                    self._inflight += 1
                    inflight_now = self._inflight
                logger.info(f"Ollama CALL START req={req_id} thread={thread_info} queued_ms={queued_ms} inflight={inflight_now}/{self._max_concurrency}")

                t_req_start = t_acquired
                status_code = None
                ok = False
                entities: List[Dict[str, Any]] = []
                response_text = ""
                try:
                    response = self.session.post(self.api_url, json=payload, timeout=self.request_timeout_seconds)
                    status_code = response.status_code if hasattr(response, 'status_code') else None
                    response.raise_for_status()
                    # Parse the response
                    result = response.json()
                    response_text = result.get("response", "")
                    entities = self._parse_json_response(response_text)
                    # Optional one-shot retry with stricter prompt if parse returned nothing
                    if self.retry_on_parse_fail and not entities:
                        strict_prompt = (
                            "Return only a valid JSON array (no markdown or explanations). If none, return [].\n\n"
                            + prompt
                        )
                        retry_payload = dict(payload)
                        retry_payload["prompt"] = strict_prompt
                        try:
                            r2 = self.session.post(self.api_url, json=retry_payload, timeout=self.request_timeout_seconds)
                            status2 = r2.status_code if hasattr(r2, 'status_code') else None
                            r2.raise_for_status()
                            res2 = r2.json()
                            txt2 = res2.get("response", "")
                            ent2 = self._parse_json_response(txt2)
                            if ent2:
                                entities = ent2
                                status_code = status2
                        except Exception as _:
                            pass
                    ok = True
                    logger.info(f"Extracted {len(entities)} entities using Ollama model")
                finally:
                    t_end = time.perf_counter()
                    req_ms = int((t_end - t_req_start) * 1000)
                    # Simple adaptive threads: if requests are repeatedly slow, drop to 4 once
                    try:
                        if self._adapt_threads_on_wobble and not self._threads_dropped:
                            if req_ms >= self._wobble_high_ms:
                                self._wobble_high_count += 1
                            else:
                                # decay slowly
                                self._wobble_high_count = max(0, self._wobble_high_count - 1)
                            if self._wobble_high_count >= 2 and self.num_thread > 4:
                                self.num_thread = 4
                                self._threads_dropped = True
                                logger.info(f"Ollama adaptive threads engaged: num_thread -> {self.num_thread}")
                    except Exception:
                        pass
                    with self._inflight_lock:
                        self._inflight -= 1
                        inflight_after = self._inflight
                    logger.info(f"Ollama CALL END req={req_id} thread={thread_info} ok={ok} status={status_code} req_ms={req_ms} inflight_now={inflight_after}/{self._max_concurrency}")

            # Store in cache (bounded LRU)
            try:
                self._cache[text] = entities
                self._cache.move_to_end(text)
                if len(self._cache) > self._cache_max:
                    self._cache.popitem(last=False)
            except Exception:
                pass

            return entities

        except requests.RequestException as e:
            logger.error(f"Error connecting to Ollama API: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Ollama response as JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during Ollama entity extraction: {e}")
            return []

    def _ensure_model_available(self) -> None:
        """Check tags and try to auto-pull the model via Ollama HTTP API if missing."""
        try:
            # Ollama tags endpoint
            tags_url = self.api_url.replace("/api/generate", "/api/tags")
            r = self.session.get(tags_url, timeout=15)
            r.raise_for_status()
            data = r.json() or {}
            models = data.get("models", [])
            names = {m.get("name") for m in models if isinstance(m, dict)}
            if self.model_name in names:
                return
        except Exception:
            # If tags fetch fails, attempt pull anyway
            pass
        # Attempt pull
        pull_url = self.api_url.replace("/api/generate", "/api/pull")
        try:
            logger.info(f"Ollama auto-pulling model: {self.model_name}")
            pr = self.session.post(pull_url, json={"name": self.model_name}, timeout=60)
            pr.raise_for_status()
        except Exception as e:
            logger.warning(f"Ollama pull request failed: {e}")

    def warm_up(self) -> None:
        """Send a tiny request to keep the model loaded and connections warm."""
        tiny_prompt = "Return [].\nInput:\n."
        payload = {
            "model": self.model_name,
            "prompt": tiny_prompt,
            "temperature": 0.0,
            "stream": False,
            "format": "json",
            "keep_alive": self.keep_alive,
            "options": {"num_predict": 8},
        }
        try:
            self.session.post(self.api_url, json=payload, timeout=10)
        except Exception:
            pass
    
    def _parse_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the JSON response from the Ollama model.
        
        The model might return additional text before or after the JSON array,
        so this method attempts to extract just the JSON part.
        
        Args:
            response_text: The raw text response from the model
            
        Returns:
            The parsed list of entities
        """
        logger.debug(f"Ollama raw response: {response_text[:500]}...")  # Log first 500 chars
        
        # Add detailed UI debug logging
        if hasattr(self, '_debug_ui_mode'):
            logger.error(f"UI DEBUG - Full Ollama response: {repr(response_text)}")
        
        # Normalize common wrappers like markdown code fences
        if response_text.strip().startswith("```"):
            # Strip leading and trailing code fences if present
            try:
                stripped = response_text.strip().strip("`")
                # Remove possible language tag like json\n
                first_newline = stripped.find("\n")
                if first_newline != -1 and stripped[:first_newline].lower().startswith("json"):
                    response_text = stripped[first_newline+1:]
                else:
                    response_text = stripped
            except Exception:
                pass

        # Try to find JSON array in the response
        try:
            # First, try to parse the entire response as JSON
            parsed = json.loads(response_text)
            logger.debug(f"Successfully parsed entire response as JSON: {type(parsed)}")
            if isinstance(parsed, list):
                return parsed
            # Accept object-shaped responses like {"entities": [...]}
            if isinstance(parsed, dict):
                for key in ("entities", "data", "results"):
                    if isinstance(parsed.get(key), list):
                        return parsed.get(key)
            return []
        except json.JSONDecodeError as e:
            logger.debug(f"Could not parse entire response as JSON: {e}")
            # If that fails, try to extract the JSON array part
            try:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    logger.debug(f"Extracted JSON string: {json_str[:200]}...")
                    parsed = json.loads(json_str)
                    logger.debug(f"Successfully parsed extracted JSON: {type(parsed)}")
                    return parsed if isinstance(parsed, list) else []
                else:
                    # As a final fallback, try to parse one or more consecutive JSON objects
                    # commonly produced by some models instead of a single array.
                    decoder = json.JSONDecoder()
                    idx = 0
                    objs = []
                    text_len = len(response_text)
                    while idx < text_len:
                        # Find the next JSON value start
                        next_brace = response_text.find('{', idx)
                        next_bracket = response_text.find('[', idx)
                        # Choose the earliest valid start
                        candidates = [p for p in (next_brace, next_bracket) if p != -1]
                        if not candidates:
                            break
                        idx = min(candidates)
                        try:
                            obj, end = decoder.raw_decode(response_text, idx)
                            objs.append(obj)
                            idx = end
                        except json.JSONDecodeError:
                            # Move forward one char and try again
                            idx += 1
                            continue
                    # Consolidate results
                    if objs:
                        # If any object has an 'entities' list, merge those
                        entities_agg = []
                        for o in objs:
                            if isinstance(o, dict) and isinstance(o.get('entities'), list):
                                entities_agg.extend(o.get('entities'))
                        if entities_agg:
                            return entities_agg
                        # Otherwise if the objects themselves look like entity dicts, return them
                        if all(isinstance(o, dict) and ('text' in o or 'value' in o or 'entity_type' in o or 'type' in o) for o in objs):
                            return objs  # treat as list of entity objects
                    logger.error("Could not find JSON array in Ollama response")
                    logger.debug(f"Response text: {response_text}")
                    return []
            except Exception as e:
                logger.error(f"Error extracting JSON from Ollama response: {e}")
                logger.debug(f"Response text: {response_text}")
                return []

    
    def close(self):
        """Close the HTTP session to free resources"""
        if hasattr(self, 'session'):
            self.session.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.close()


def convert_ollama_to_presidio(text: str, ollama_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert entities extracted by Ollama to the format used by Presidio.
    
    Args:
        text: The original text that was analyzed
        ollama_entities: Entities extracted by the Ollama model
        
    Returns:
        Entities in Presidio format
    """
    import logging
    logger = logging.getLogger('presidio-ollama-integration')
    
    import re

    presidio_entities: List[Dict[str, Any]] = []
    seen_spans = set()  # (entity_type, start, end)

    logger.error(f"UI DEBUG - Converting {len(ollama_entities)} entities")
    logger.error(f"UI DEBUG - Original text: {repr(text)}")

    # Precompute lowercase for case-insensitive searches
    text_lower = text.lower()

    # Lightweight helpers
    def _luhn_check(num: str) -> bool:
        digits = [int(c) for c in num if c.isdigit()]
        if len(digits) < 13:
            return False
        checksum = 0
        parity = (len(digits) - 2) % 2
        for i, d in enumerate(digits[::-1]):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
        return checksum % 10 == 0

    def _looks_like_credit_card(val: str) -> bool:
        # 13-19 digits after stripping spaces and hyphens and passes Luhn
        digits = ''.join(ch for ch in val if ch.isdigit())
        if not (13 <= len(digits) <= 19):
            return False
        return _luhn_check(digits)

    def _looks_like_phone(val: str) -> bool:
        import re as _re
        return _re.search(r"\b(?:\+?44\s?\d{9,11}|0\d{3,4}[\s-]?\d{3}[\s-]?\d{3,4})\b", val) is not None

    def _normalize_entity_type(e: Dict[str, Any], value_text: str) -> str:
        """Get entity type from common keys (with typo tolerance) or infer from value."""
        # 1) Try standard keys
        for k in ("entity_type", "entityType", "type", "label", "category"):
            v = e.get(k)
            if isinstance(v, str) and v.strip():
                et = v.strip()
                # Heuristic: override obviously mislabeled credit cards
                if _looks_like_credit_card(value_text) and not _looks_like_phone(value_text):
                    return "CREDIT_CARD"
                return et
        # 2) Case-insensitive / typo tolerant: any key containing 'type'
        for k, v in e.items():
            if isinstance(k, str) and "type" in k.lower() and isinstance(v, str) and v.strip():
                et = v.strip()
                if _looks_like_credit_card(value_text) and not _looks_like_phone(value_text):
                    return "CREDIT_CARD"
                return et
        # 3) Heuristic inference from text value when not provided
        t = value_text or ""
        # Email
        if re.search(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b", t, flags=re.IGNORECASE):
            return "EMAIL_ADDRESS"
        # Phone (very permissive UK-ish)
        if re.search(r"\b(?:\+?44\s?\d{9,11}|0\d{3,4}[\s-]?\d{3}[\s-]?\d{3,4})\b", t):
            return "PHONE_NUMBER"
        # Credit card
        if _looks_like_credit_card(t) and not _looks_like_phone(t):
            return "CREDIT_CARD"
        # Date (dd/mm/yyyy or dd-mm-yyyy)
        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", t):
            return "DATE_TIME"
        return "UNKNOWN"

    def _get_model_name(e: Dict[str, Any]) -> str:
        return (e.get('model_name') or e.get('model') or 'ollama').strip()

    # moved and enhanced above with Luhn

    for entity in ollama_entities:
        # Extract entity details
        # Try both 'text' and 'value' keys since different models may use different formats
        raw_entity_text = entity.get("text", entity.get("value", ""))
        entity_text = (raw_entity_text or "").strip()
        entity_type = _normalize_entity_type(entity, entity_text)
        if isinstance(entity_type, str):
            entity_type = entity_type.strip().upper()
        # Try both 'text' and 'value' keys since different models may use different formats
        confidence = entity.get("confidence", 0.5)
        model_name = _get_model_name(entity)

        logger.error(f"UI DEBUG - Processing entity: {entity}")
        logger.error(f"UI DEBUG - Extracted text: {repr(entity_text)}")

        if not entity_text:
            logger.error("UI DEBUG - Skipping entity with no text")
            continue

        # Strategy 1: Prefer case-insensitive search to align with UI input casing
        matches = list(re.finditer(re.escape(entity_text), text, flags=re.IGNORECASE))

        # Strategy 2: If model provided start/end look plausible, use them only when direct matches are unavailable
        hinted_added = False
        try:
            hint_start = int(entity.get("start")) if entity.get("start") is not None else None
            hint_end = int(entity.get("end")) if entity.get("end") is not None else None
        except Exception:
            hint_start = hint_end = None

        # Strategy 3: Prefer direct case-insensitive matches first for stability
        matched_added = False
        if matches:
            for m in matches:
                start, end = m.start(), m.end()
                span_key = (entity_type, start, end)
                if span_key in seen_spans:
                    continue
                presidio_entity = {
                    "entity_type": entity_type,
                    "start": start,
                    "end": end,
                    "text": text[start:end],  # preserve original casing from source text
                    "confidence": confidence,
                    "source": f"ollama-{model_name}"
                }
                logger.error(f"UI DEBUG - Created presidio entity (ci match): {presidio_entity}")
                presidio_entities.append(presidio_entity)
                seen_spans.add(span_key)
                matched_added = True

        # If no direct matches were found, fall back to hinted offsets and try to align
        if not matched_added and hint_start is not None and hint_end is not None:
            if 0 <= hint_start < hint_end <= len(text):
                hinted_text = text[hint_start:hint_end]
                # Try to align hinted span exactly to entity_text if it's a super/sub-string
                if entity_text:
                    ht_low = hinted_text.lower()
                    et_low = entity_text.lower()
                    if ht_low == et_low:
                        adj_start, adj_end = hint_start, hint_end
                    elif et_low in ht_low:
                        inner_idx = ht_low.find(et_low)
                        adj_start = hint_start + inner_idx
                        adj_end = adj_start + len(entity_text)
                    else:
                        # Try a local search around hinted region, then whole text
                        search_lo = max(0, hint_start - 20)
                        search_hi = min(len(text), hint_end + 20)
                        local_idx = text.lower().find(et_low, search_lo, search_hi)
                        if local_idx == -1:
                            local_idx = text.lower().find(et_low)
                        if local_idx != -1:
                            adj_start = local_idx
                            adj_end = local_idx + len(entity_text)
                        else:
                            # As a last resort, keep the hinted span but it's less reliable
                            adj_start, adj_end = hint_start, hint_end
                else:
                    adj_start, adj_end = hint_start, hint_end

                span_key = (entity_type, adj_start, adj_end)
                if span_key not in seen_spans:
                    item = {
                        "entity_type": entity_type,
                        "start": adj_start,
                        "end": adj_end,
                        "text": text[adj_start:adj_end],  # preserve original casing
                        "confidence": confidence,
                        "source": f"ollama-{model_name}"
                    }
                    logger.error(f"UI DEBUG - Used hinted offsets: {(adj_start, adj_end)} -> {repr(item['text'])}")
                    presidio_entities.append(item)
                    seen_spans.add(span_key)
                    hinted_added = True

        if not matches and not hinted_added:
            # Strategy 4: As a last resort, try lowercased index lookup
            idx = text_lower.find(entity_text.lower())
            logger.error(f"UI DEBUG - Fallback lower() search for {repr(entity_text)} found at: {idx}")
            if idx != -1:
                end = idx + len(entity_text)
                span_key = (entity_type, idx, end)
                if span_key not in seen_spans:
                    presidio_entities.append({
                        "entity_type": entity_type,
                        "start": idx,
                        "end": end,
                        "text": text[idx:end],
                        "confidence": confidence,
                        "source": f"ollama-{model_name}"
                    })
                    seen_spans.add(span_key)
            else:
                logger.error(f"UI DEBUG - Could not find {repr(entity_text)} in {repr(text)}")

    # Post-filter for obvious type/shape mismatches to improve consistency
    filtered: List[Dict[str, Any]] = []
    import re as _re_cc
    for e in presidio_entities:
        et = e.get("entity_type")
        txt = e.get("text", "")
        # Promote PHONE_NUMBER to CREDIT_CARD if it looks like a card and not like a phone
        group_of4 = _re_cc.search(r"(?:\d{4}[ -]?){3}\d{4}", txt) is not None
        digits_only = ''.join(ch for ch in txt if ch.isdigit())
        near_card_word = False
        try:
            s, nd = int(e.get("start", 0)), int(e.get("end", 0))
            window_lo = max(0, s - 10)
            ctx = text[window_lo:s].lower()
            near_card_word = "card" in ctx
        except Exception:
            pass
        if et == "PHONE_NUMBER" and (_looks_like_credit_card(txt) or (group_of4 and (len(digits_only) in (15,16) or near_card_word))):
            e = {**e, "entity_type": "CREDIT_CARD"}
            et = "CREDIT_CARD"
        if et == "CREDIT_CARD" and not (_looks_like_credit_card(txt) or group_of4):
            # Drop spurious CC labels like "PG", "BR", etc.
            continue
        filtered.append(e)

    logger.error(f"UI DEBUG - Final conversion result: {filtered}")
    return filtered

# Example function to analyze text with Ollama
def analyze_text_with_ollama(
    text: str,
    model_name: str = "mistral:7b-instruct",
    prompt_template: Optional[str] = None,
    entity_types: Optional[List[str]] = None,
    _ui_debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Analyze text for entities using an Ollama model.
    
    Args:
        text: The text to analyze
        model_name: The Ollama model to use
        prompt_template: Custom prompt template for entity extraction
        entity_types: List of entity types to extract
        _ui_debug: Enable enhanced debug logging for UI issues
        
    Returns:
        A list of extracted entities in Presidio format
    """
    # Create an Ollama entity extractor
    extractor = OllamaEntityExtractor(
        model_name=model_name,
        prompt_template=prompt_template,
        entity_types=entity_types
    )
    
    # Enable UI debug mode if requested
    if _ui_debug:
        extractor._debug_ui_mode = True
        logger.error(f"UI DEBUG - Analyzing text: {repr(text)}")
        logger.error(f"UI DEBUG - Model: {model_name}")
    
    def _regex_fallback_presidio(t: str) -> List[Dict[str, Any]]:
        import re as _re
        results: List[Dict[str, Any]] = []
        # Email
        for m in _re.finditer(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b", t):
            results.append({
                "entity_type": "EMAIL_ADDRESS",
                "start": m.start(),
                "end": m.end(),
                "text": t[m.start():m.end()],
                "confidence": 0.8,
                "source": "regex"
            })
        # UK-ish Phone numbers (simple)
        for m in _re.finditer(r"\b(?:\+?44\s?\d{9,11}|0\d{3,4}[\s-]?\d{3}[\s-]?\d{3,4})\b", t):
            results.append({
                "entity_type": "PHONE_NUMBER",
                "start": m.start(),
                "end": m.end(),
                "text": t[m.start():m.end()],
                "confidence": 0.8,
                "source": "regex"
            })
        # Credit cards: simple 13-19 digit sequences with Luhn check
        def _digits(s: str) -> str:
            return ''.join(ch for ch in s if ch.isdigit())
        for m in _re.finditer(r"(?:\b(?:\d[ -]?){13,19}\b)", t):
            seg = t[m.start():m.end()]
            d = _digits(seg)
            if 13 <= len(d) <= 19:
                # Luhn check
                ld = [int(c) for c in d]
                checksum = 0
                dbl = False
                for x in reversed(ld):
                    y = x * 2 if dbl else x
                    if y > 9:
                        y -= 9
                    checksum += y
                    dbl = not dbl
                if checksum % 10 == 0:
                    results.append({
                        "entity_type": "CREDIT_CARD",
                        "start": m.start(),
                        "end": m.end(),
                        "text": seg,
                        "confidence": 0.85,
                        "source": "regex"
                    })
        # Dates dd/mm/yyyy or dd-mm-yyyy
        for m in _re.finditer(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", t):
            results.append({
                "entity_type": "DATE_TIME",
                "start": m.start(),
                "end": m.end(),
                "text": t[m.start():m.end()],
                "confidence": 0.7,
                "source": "regex"
            })
        return results

    try:
        # Extract entities
        ollama_entities = extractor.extract_entities(text)

        if _ui_debug:
            logger.error(f"UI DEBUG - Ollama entities returned: {ollama_entities}")

        # Convert to Presidio format
        presidio_entities = convert_ollama_to_presidio(
            text=text,
            ollama_entities=ollama_entities
        )

        # Optional deterministic fallback for consistency
        use_fallback = os.getenv("OLLAMA_FALLBACK_REGEX")
        fallback_on = (str(use_fallback).lower() not in ("0", "false", "no")) if use_fallback is not None else True
        if fallback_on:
            if not presidio_entities:
                presidio_entities = _regex_fallback_presidio(text)
            else:
                # Merge, avoiding duplicates by span/type
                fallback = _regex_fallback_presidio(text)
                seen = {(e["entity_type"], e["start"], e["end"]) for e in presidio_entities}
                for e in fallback:
                    key = (e["entity_type"], e["start"], e["end"])
                    if key not in seen:
                        presidio_entities.append(e)
                        seen.add(key)

        if _ui_debug:
            logger.error(f"UI DEBUG - Final Presidio format: {presidio_entities}")

        return presidio_entities
        
    except Exception as e:
        if _ui_debug:
            logger.error(f"UI DEBUG - Exception in analyze_text_with_ollama: {e}")
            import traceback
            logger.error(f"UI DEBUG - Traceback: {traceback.format_exc()}")
        raise
