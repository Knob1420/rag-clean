"""
WeKnora Faithful Port — Constants

Ported from WeKnora internal/agent/const.go
"""

# ── Agent defaults ──────────────────────────────────────────
DEFAULT_AGENT_TEMPERATURE = 0.7
DEFAULT_AGENT_MAX_ITERATIONS = 20
DEFAULT_LLM_CALL_TIMEOUT = 120  # seconds
DEFAULT_TOOL_EXEC_TIMEOUT = 60  # seconds

# ── Retry thresholds ───────────────────────────────────────
MAX_LLM_RETRIES = 2
MAX_EMPTY_RESPONSE_RETRIES = 2
MAX_REPEATED_RESPONSE_ROUNDS = 2

# ── Tool output ─────────────────────────────────────────────
DEFAULT_MAX_TOOL_OUTPUT = 16000  # chars (WeKnora default)

# ── Transient error markers (for retry detection) ──────────
TRANSIENT_ERROR_MARKERS = [
    "429",
    "500",
    "502",
    "503",
    "timeout",
    "rate_limit",
    "overloaded",
    "connection reset",
    "context deadline",
]

# ── Context management ──────────────────────────────────────
DEFAULT_CONTEXT_TOKENS = 32768  # DeepSeek context window
DEFAULT_CONSOLIDATION_THRESHOLD = 0.5  # trigger consolidation at 50% of context
DEFAULT_CONTEXT_THRESHOLD_RATIO = 0.8  # trigger compression at 80%
CONSOLIDATION_TIMEOUT = 60  # seconds
MAX_CONSOLIDATION_ATTEMPTS = 3
