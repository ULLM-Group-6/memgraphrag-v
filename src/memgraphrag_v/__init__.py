"""Shared interfaces for MemGraphRAG-V (task [01]).

Every pipeline stage reads and writes the records defined here. The
human-readable contract is docs/interfaces.md; bump SCHEMA_VERSION whenever a
record changes shape.
"""

SCHEMA_VERSION = "1"

__all__ = ["SCHEMA_VERSION"]
