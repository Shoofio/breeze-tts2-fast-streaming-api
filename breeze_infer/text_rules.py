"""Text rules shared by the HTTP boundary and the voice file reader.

A module of its own because both `http_fields` (request fields) and `voice_file` (a saved
voice's `ref_text`, read back from disk) apply them, and `http_fields` already imports
`voice_file`: putting them in either would make the two import each other.
"""

from __future__ import annotations

import unicodedata


def has_control_characters(value: str) -> bool:
    """BC-46: any Unicode general-category `Cc` (control) character other than tab, CR
    and LF. `Cc` is precisely C0 (`\\x00`-`\\x1f`, e.g. NUL and ESC), DEL (`\\x7f`) and the
    C1 controls (`\\x80`-`\\x9f`, e.g. NEL `\\x85`) -- exactly the set the contract means by
    "control characters", so this is checked via `unicodedata.category` rather than a fixed
    codepoint list.
    """
    return any(ch not in "\t\r\n" and unicodedata.category(ch) == "Cc" for ch in value)


def is_utf8_encodable(value: str) -> bool:
    """False for a string holding a lone surrogate (`\\ud800`): valid in JSON and in a
    Python `str`, but it can't be encoded as UTF-8, so any response carrying it fails
    with UnicodeEncodeError."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True
