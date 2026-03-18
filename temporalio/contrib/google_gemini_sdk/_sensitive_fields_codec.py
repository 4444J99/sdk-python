"""Targeted field-level encryption for Temporal payloads.

Encrypts specific keys within specific fields of specific Pydantic models,
leaving everything else fully readable in the Temporal UI without a Codec Server.

This module provides three components that work together:

1. **TypeTaggingPydanticJSONConverter** — a Pydantic JSON converter that writes
   the Python type name into payload metadata so the codec can identify which
   model produced a payload.

2. **SensitiveFieldsCodec** — a PayloadCodec that reads the type tag, looks up
   which dict-field keys are sensitive, and encrypts/decrypts just those values.

3. **make_sensitive_fields_data_converter** — a factory that wires both into a
   single ``DataConverter`` ready to pass to ``Client.connect()`` and ``Worker()``.

See ``plans/sensitive_fields_codec.md`` for the full design document.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Sequence

from cryptography.fernet import Fernet
from pydantic import BaseModel

import temporalio.api.common.v1
from temporalio.contrib.pydantic import (
    PydanticJSONPlainPayloadConverter,
    PydanticPayloadConverter,
    pydantic_data_converter,
)
from temporalio.converter import (
    DataConverter,
    PayloadCodec,
)

# ── Constants ────────────────────────────────────────────────────────────────

SENSITIVE_FIELDS_ENCODING = b"json/sensitive-fields"
"""Encoding metadata value written to payloads that have had fields encrypted."""

SENTINEL_PREFIX = "__enc__"
"""Prefix on encrypted values — a defensive guard against double-decryption."""

TYPE_TAG_METADATA_KEY = "x-python-type"
"""Metadata key (str) carrying the Python ``__qualname__`` of the serialized value."""


# ── Converter ────────────────────────────────────────────────────────────────


class TypeTaggingPydanticJSONConverter(PydanticJSONPlainPayloadConverter):
    """Pydantic JSON converter that tags payloads with the Python type name.

    For registered (watched) types, writes ``type(value).__qualname__`` into
    ``payload.metadata[x-python-type]`` after standard Pydantic serialization.
    This tag is read by :class:`SensitiveFieldsCodec` to route encryption.
    """

    def __init__(
        self,
        watched_types: frozenset[type],
        to_json_options: Any = None,
    ) -> None:
        super().__init__(to_json_options)
        self._watched = watched_types

    def to_payload(
        self, value: Any
    ) -> temporalio.api.common.v1.Payload | None:
        payload = super().to_payload(value)
        if payload is not None and isinstance(value, tuple(self._watched)):
            payload.metadata[TYPE_TAG_METADATA_KEY] = (
                type(value).__qualname__.encode("utf-8")
            )
        return payload


# ── Codec ────────────────────────────────────────────────────────────────────


class SensitiveFieldsCodec(PayloadCodec):
    """PayloadCodec that encrypts specific keys within dict fields of registered models.

    For each registered Pydantic model type, the config specifies which dict
    fields contain sensitive keys and which keys within those dicts to encrypt.

    Example config::

        model_configs = {
            HttpRequestData: {"headers": {"x-goog-api-key", "authorization"}},
        }

    This encrypts ``headers["x-goog-api-key"]`` and ``headers["authorization"]``
    within ``HttpRequestData`` payloads while leaving all other headers, the URL,
    method, body, etc. fully readable.
    """

    def __init__(
        self,
        model_configs: dict[type[BaseModel], dict[str, set[str]]],
        encryption_key: bytes,
    ) -> None:
        """Initialize the codec.

        Args:
            model_configs: Mapping of model type → dict field name → set of
                sensitive keys within that dict.
            encryption_key: A Fernet key (32 bytes, URL-safe base64).  Generate
                one with ``cryptography.fernet.Fernet.generate_key()``.
        """
        self._configs: dict[str, dict[str, set[str]]] = {
            t.__qualname__: fields_config
            for t, fields_config in model_configs.items()
        }
        self._fernet = Fernet(encryption_key)

    async def encode(
        self, payloads: Sequence[temporalio.api.common.v1.Payload]
    ) -> list[temporalio.api.common.v1.Payload]:
        """Encrypt sensitive dict-field keys in matching payloads."""
        return [self._encode_one(p) for p in payloads]

    async def decode(
        self, payloads: Sequence[temporalio.api.common.v1.Payload]
    ) -> list[temporalio.api.common.v1.Payload]:
        """Decrypt sensitive dict-field keys in matching payloads."""
        return [self._decode_one(p) for p in payloads]

    def _encode_one(
        self, p: temporalio.api.common.v1.Payload
    ) -> temporalio.api.common.v1.Payload:
        # Gate: type tag must be present and registered.
        type_name = p.metadata.get(TYPE_TAG_METADATA_KEY, b"").decode("utf-8")
        if not type_name or type_name not in self._configs:
            return p

        config = self._configs[type_name]
        data: dict[str, Any] = json.loads(p.data)
        changed = False

        for dict_field, sensitive_keys in config.items():
            field_val = data.get(dict_field)
            if not isinstance(field_val, dict):
                continue
            for key in sensitive_keys:
                if key in field_val and isinstance(field_val[key], str):
                    plaintext = field_val[key].encode("utf-8")
                    ciphertext = self._fernet.encrypt(plaintext).decode("utf-8")
                    field_val[key] = SENTINEL_PREFIX + ciphertext
                    changed = True

        if not changed:
            return p

        new_metadata = dict(p.metadata)
        new_metadata["encoding"] = SENSITIVE_FIELDS_ENCODING  # bytes
        return temporalio.api.common.v1.Payload(
            metadata=new_metadata,
            data=json.dumps(data).encode("utf-8"),
        )

    def _decode_one(
        self, p: temporalio.api.common.v1.Payload
    ) -> temporalio.api.common.v1.Payload:
        # Gate 1: encoding must match.
        if p.metadata.get("encoding") != SENSITIVE_FIELDS_ENCODING:
            return p

        # Gate 2: type tag must be present.
        type_name = p.metadata.get(TYPE_TAG_METADATA_KEY, b"").decode("utf-8")
        if not type_name:
            return p

        # Gate 3: type must be registered.
        config = self._configs.get(type_name)
        if config is None:
            return p

        data: dict[str, Any] = json.loads(p.data)

        for dict_field, sensitive_keys in config.items():
            field_val = data.get(dict_field)
            if not isinstance(field_val, dict):
                continue
            for key in sensitive_keys:
                val = field_val.get(key)
                if isinstance(val, str) and val.startswith(SENTINEL_PREFIX):
                    ciphertext = val[len(SENTINEL_PREFIX) :].encode("utf-8")
                    field_val[key] = self._fernet.decrypt(ciphertext).decode("utf-8")

        new_metadata = dict(p.metadata)
        new_metadata["encoding"] = b"json/plain"
        return temporalio.api.common.v1.Payload(
            metadata=new_metadata,
            data=json.dumps(data).encode("utf-8"),
        )


# ── Factory ──────────────────────────────────────────────────────────────────


def make_sensitive_fields_data_converter(
    model_configs: dict[type[BaseModel], dict[str, set[str]]],
    encryption_key: bytes,
) -> DataConverter:
    """Create a ``DataConverter`` with targeted field-level encryption.

    Args:
        model_configs: Mapping of model type → dict field name → set of
            sensitive keys within that dict.
        encryption_key: A Fernet key.

    Returns:
        A ``DataConverter`` drop-in replacement for ``pydantic_data_converter``.
    """
    watched_types = frozenset(model_configs.keys())

    # Create a zero-arg converter class that closes over watched_types.
    # DataConverter instantiates this class with no arguments in __post_init__.
    class _TaggingConverter(PydanticPayloadConverter):
        def __init__(self, to_json_options: Any = None) -> None:
            # Let PydanticPayloadConverter set up all builtin converters normally,
            # then swap the JSON converter with our type-tagging subclass.
            super().__init__(to_json_options)
            tagging_json = TypeTaggingPydanticJSONConverter(
                watched_types, to_json_options
            )
            self.converters = {
                k: tagging_json if k == b"json/plain" else v
                for k, v in self.converters.items()
            }

    return dataclasses.replace(
        pydantic_data_converter,
        payload_converter_class=_TaggingConverter,
        payload_codec=SensitiveFieldsCodec(model_configs, encryption_key),
    )
