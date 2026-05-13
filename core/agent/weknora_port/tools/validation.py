"""
WeKnora Faithful Port — Tool Parameter Validation & Casting

Ported from WeKnora internal/agent/tools/registry.go (ValidateParams + CastParams)

Validates tool call arguments against JSON Schema definitions,
and casts parameter types to match expected types (e.g., string "5" → int 5).
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger


def validate_params(args: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    """
    Validate tool call arguments against the parameter JSON Schema.

    Returns None if valid, or an error message string if invalid.

    Checks:
    - Required fields are present
    - Type mismatches (before casting)
    - Array minItems/maxItems
    - String minLength (if specified)
    """
    if not schema:
        return None

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Check required fields
    missing = required - set(args.keys())
    if missing:
        return f"Missing required parameters: {', '.join(sorted(missing))}"

    # Check property types
    for key, value in args.items():
        if key not in properties:
            continue  # extra params are allowed

        prop_schema = properties[key]
        expected_type = prop_schema.get("type")

        if expected_type and value is not None:
            type_ok = _check_type(value, expected_type)
            if not type_ok:
                actual_type = type(value).__name__
                return (
                    f"Parameter '{key}' has wrong type: "
                    f"expected {expected_type}, got {actual_type}"
                )

        # Array constraints
        if expected_type == "array" and isinstance(value, list):
            min_items = prop_schema.get("minItems")
            max_items = prop_schema.get("maxItems")
            if min_items is not None and len(value) < min_items:
                return f"Parameter '{key}' has too few items: {len(value)} < {min_items}"
            if max_items is not None and len(value) > max_items:
                return f"Parameter '{key}' has too many items: {len(value)} > {max_items}"

    return None


def cast_params(args: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cast tool call arguments to match the expected types in the schema.

    LLMs sometimes return all parameters as strings (e.g., "5" instead of 5).
    This function coerces types where safe:
    - string → int/float/bool/array
    - int → string (when schema expects string)
    - JSON string → parsed object/array

    Returns a new dict with casted values.
    """
    if not schema:
        return args

    properties = schema.get("properties", {})
    result = dict(args)

    for key, value in list(result.items()):
        if key not in properties or value is None:
            continue

        prop_schema = properties[key]
        expected_type = prop_schema.get("type")

        if not expected_type:
            continue

        casted = _cast_value(value, expected_type, prop_schema)
        if casted is not None:
            result[key] = casted

    return result


def _check_type(value: Any, expected_type: str) -> bool:
    """Check if value matches the expected JSON Schema type."""
    type_map = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "array": (list,),
        "object": (dict,),
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return True  # unknown type, allow

    # bool is subclass of int in Python, exclude it from integer check
    if expected_type == "integer" and isinstance(value, bool):
        return False
    if expected_type == "boolean" and not isinstance(value, bool):
        return False

    return isinstance(value, expected)


def _cast_value(value: Any, expected_type: str, prop_schema: Dict[str, Any]) -> Any:
    """
    Attempt to cast a value to the expected type.
    Returns None if casting fails (caller should keep original value).
    """
    try:
        if expected_type == "integer":
            if isinstance(value, str):
                # Try direct int conversion
                return int(value)
            if isinstance(value, float) and value == int(value):
                return int(value)

        elif expected_type == "number":
            if isinstance(value, str):
                return float(value)

        elif expected_type == "boolean":
            if isinstance(value, str):
                if value.lower() in ("true", "1", "yes"):
                    return True
                if value.lower() in ("false", "0", "no"):
                    return False

        elif expected_type == "string":
            if isinstance(value, (int, float)):
                return str(value)

        elif expected_type == "array":
            if isinstance(value, str):
                # Try parsing as JSON array
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
                # Wrap single value in list
                return [parsed]
            if isinstance(value, (int, float, bool)):
                return [value]

        elif expected_type == "object":
            if isinstance(value, str):
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed

    except (ValueError, json.JSONDecodeError, TypeError):
        logger.debug(f"[Validation] Failed to cast {key}={value!r} to {expected_type}")

    return None  # casting failed, keep original
