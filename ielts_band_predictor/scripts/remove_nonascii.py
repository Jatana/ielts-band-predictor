import re
from typing import Optional

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")


def strip_non_ascii(text: str, *, ratio: float, absolute: int) -> Optional[str]:
    n_total = len(text)
    cleaned = _NON_ASCII_RE.sub("", text)
    removed = n_total - len(cleaned)

    if removed > absolute or removed / max(n_total, 1) > ratio:
        return None
    return cleaned
