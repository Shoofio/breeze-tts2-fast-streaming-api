"""Canonicalizing a bare ``scheme://host[:port]`` origin (CORS; contracts/http-api.md "CORS").

Dependency-free: only the standard library. Neither ``breeze_infer.settings`` nor
``breeze_infer.cors`` is imported here -- both of *them* import ``canonical_origin`` from this
module, so importing either back would be a cycle.

Used from two different places, which is why every message here is neutral about *why* a value
is being canonicalized and never mentions ``--cors``: ``settings.py`` calls it to validate and
normalize each ``--cors`` allowlist entry at startup (where ``parser.error()`` already prefixes
the flag), while ``cors.py`` calls it on every request's incoming ``Origin`` header, where a
``ValueError`` is never shown to anyone -- a malformed header is simply treated as not allowed.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from urllib.parse import urlsplit

_DEFAULT_PORTS = {"http": 80, "https": 443}
# What's left of a lowercase, ASCII-only hostname -- letters, digits, '.', '_' (docker-compose
# service names) and '-' are the only characters a real DNS label can contain.
_HOST_CHARS_RE = re.compile(r"[a-z0-9_.-]+")
_DOTTED_QUAD_RE = re.compile(r"[0-9.]+")
# `*` (not `+`): the bare label "0x" itself is just as much a smuggled-IP trick as "0x7f000001"
# (review-agent final pass, issue 2).
_HEX_LABEL_RE = re.compile(r"0x[0-9a-f]*")


def canonical_origin(value: str) -> str:
    """Canonicalize one ``scheme://host[:port]`` origin to the exact form a browser's ``Origin``
    header carries, so an allowlist entry and an incoming header that only differ cosmetically
    (case, a redundant default port, an expanded IPv6 literal, ...) compare equal.

    Raises ``ValueError`` naming ``value`` on anything that isn't a bare origin a real browser
    could send; callers decide what that means (see the module docstring).

    - No control character or whitespace may appear anywhere in ``value`` -- checked before
      ``urlsplit``, which otherwise silently strips a tab, CR or LF rather than rejecting them.
    - Scheme must be ``http`` or ``https`` (lowercased).
    - Host is lowercased. An IP literal (IPv4 dotted-quad, or IPv6 in brackets) is canonicalized
      with ``ipaddress``: ``127.1`` (which browsers never send; ``ipaddress`` requires all four
      octets) and an IPv4-mapped IPv6 address (``::ffff:a.b.c.d``, which no browser sends either)
      are both rejected, an IPv6 zone id is rejected, an IPvFuture literal (``[v1....]``) is
      rejected, and ``[0:0:0:0:0:0:0:1]`` canonicalizes to ``[::1]``. A non-ASCII hostname is
      rejected -- browsers send punycode, so the caller must too, rather than this module guessing
      at an IDNA conversion. A hostname of ``*``, one containing any character outside
      ``[a-z0-9_.-]``, or one whose last label is all-digit or hex-looking (``0x...``) -- a
      classic trick for smuggling an IP address past a hostname allowlist -- is also rejected.
    - Port must be plain digits, 1-65535 (browsers never send port ``0``); the scheme's own
      default port (80 for http, 443 for https) is dropped, since browsers omit it.
    - Userinfo, a path (including a lone trailing ``/``), a query or a fragment are all rejected,
      including an empty ``?`` or ``#`` -- none of those can ever appear in an ``Origin`` header.
    """
    _reject_unsafe_characters(value)

    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ValueError(f"origin {value!r} is invalid: {exc}") from exc

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"origin {value!r} must start with http:// or https://")

    if "@" in parsed.netloc:
        raise ValueError(
            f"origin {value!r} must be a bare scheme://host[:port] origin (no userinfo)"
        )

    # `parsed.query`/`parsed.fragment` are empty strings both when the header is absent and when
    # it's present but empty (a bare trailing '?' or '#'), so the raw value is checked instead.
    has_query_or_fragment = "?" in value or "#" in value
    if parsed.path == "/" and not has_query_or_fragment:
        without_slash = value[: value.rindex("/")]
        raise ValueError(
            f"origin {value!r} must not have a trailing slash; use {without_slash!r}"
        )
    if parsed.path or has_query_or_fragment:
        raise ValueError(
            f"origin {value!r} must be a bare scheme://host[:port] origin, "
            "with no path, query or fragment"
        )

    # `SplitResult.port` is an `int`: it can't tell a genuinely absent port from an *empty* one
    # (`"host:"`, `.port` gives `None` either way), and it silently drops a leading zero
    # (`"080"` -> `80`) -- neither of which a real browser's `Origin` header would ever produce.
    # The raw text is checked before `.port` gets a chance to launder either away (review-agent
    # final pass, issue 5).
    raw_port = _raw_port(parsed.netloc)
    if raw_port == "":
        raise ValueError(f"origin {value!r} has an empty port")
    if raw_port is not None and raw_port.isdigit() and len(raw_port) > 1 and raw_port[0] == "0":
        raise ValueError(f"origin {value!r} port must not have a leading zero")

    try:
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"origin {value!r} has an invalid port") from exc

    if not host:
        raise ValueError(f"origin {value!r} must be a bare scheme://host[:port] origin")
    if port == 0:
        raise ValueError(f"origin {value!r} port must be between 1 and 65535")

    host = _canonical_host(value, host, bracketed=parsed.netloc.startswith("["))

    port_suffix = "" if port is None or port == _DEFAULT_PORTS[scheme] else f":{port}"
    return f"{scheme}://{host}{port_suffix}"


def _reject_unsafe_characters(value: str) -> None:
    """``urlsplit`` silently strips a tab, CR or LF before parsing rather than rejecting them;
    reject those, every other control character, and any whitespace up front instead of relying
    on it to catch just those three.

    ``unicodedata.category(ch) == "Cc"`` is every Unicode control character -- both C0 (the
    former ord-range check only covered ``0x00-0x1F`` plus DEL ``0x7F``) and C1 (``0x80-0x9F``,
    review-agent final pass issue 10) -- alongside ``ch.isspace()`` for whitespace itself, which
    ``Cc`` does not include.
    """
    for ch in value:
        if ch.isspace() or unicodedata.category(ch) == "Cc":
            raise ValueError(
                f"origin {value!r} must not contain control characters or whitespace"
            )


def _raw_port(netloc: str) -> str | None:
    """The raw text after the port-introducing ``':'`` in ``netloc`` (userinfo, if any, is
    already rejected by the caller before this runs), or ``None`` if there is no ``':'`` at all.

    ``SplitResult.port`` is an ``int``, so it can't distinguish an *empty* port
    (``"host:"`` -- gives ``None``, same as no port) from a genuinely absent one, and it silently
    drops a leading zero (``"080"`` -> ``80``). Working from the original text lets the caller
    reject both.
    """
    rest = netloc
    if rest.startswith("["):
        end = rest.find("]")
        if end == -1:
            return None  # unterminated bracket; urlsplit itself already rejects this
        rest = rest[end + 1 :]
    if ":" not in rest:
        return None
    return rest.split(":", 1)[1]


def _canonical_host(value: str, host: str, *, bracketed: bool) -> str:
    """``host`` is ``urlsplit(value).hostname``: already lowercased and brackets stripped.

    ``bracketed`` says whether the *original* netloc wrapped the host in ``[...]`` -- checked
    from the netloc text itself, not by looking for a leftover ``':'`` in ``host``: an IPvFuture
    literal like ``"[v1.example]"`` has no ``':'`` once its brackets are stripped, so a check
    keyed on ``':' in host`` let it fall through and be treated as the ordinary hostname
    ``"v1.example"`` instead of being validated as the IP literal its brackets promised (review-
    agent final pass, issue 1). Anything bracketed is validated as IPv6 or rejected outright, with
    no fallback to the hostname branch below.
    """
    if bracketed:
        return _canonical_ipv6_literal(value, host)

    if _DOTTED_QUAD_RE.fullmatch(host):
        # All digits and dots: this was clearly meant as an IPv4 literal, so a failure here (e.g.
        # '127.1', which ipaddress rejects since it isn't all four dotted-quad octets) is reported
        # as a bad IPv4 address rather than falling through to the hostname checks below.
        try:
            return str(ipaddress.IPv4Address(host))
        except ValueError as exc:
            raise ValueError(
                f"origin {value!r} host must be dotted-quad IPv4 (got {host!r}): {exc}"
            ) from exc

    if not host.isascii():
        raise ValueError(f"origin {value!r} host must be ASCII; use the punycode (xn--) form")
    if "*" in host:
        raise ValueError(f"origin {value!r}: wildcard hosts aren't supported; list each origin")
    if not _HOST_CHARS_RE.fullmatch(host):
        raise ValueError(
            f"origin {value!r} host must contain only letters, digits, '.', '_' or '-' "
            f"(got {host!r})"
        )

    # A trailing '.' is FQDN absolute-name syntax ("example.com."); stripped before splitting on
    # '.' so it doesn't leave an *empty* last label, which neither `isdigit()` nor the hex regex
    # would ever match -- silently bypassing the check below for e.g. "evil.example.2130706433."
    # (review-agent final pass, issue 2). The returned `host` keeps the dot as written; only the
    # label check ignores it.
    without_trailing_dot = host.removesuffix(".")
    last_label = without_trailing_dot.rsplit(".", 1)[-1]
    if last_label.isdigit() or _HEX_LABEL_RE.fullmatch(last_label):
        # A trailing all-numeric or hex-looking label (e.g. "evil.example.2130706433" or
        # "evil.example.0x7f000001") is a classic trick for smuggling an IP address past a
        # hostname allowlist; a real DNS label is never purely numeric or hex-prefixed.
        raise ValueError(
            f"origin {value!r} host's last label {last_label!r} looks numeric or hex-looking, "
            "not a hostname"
        )
    return host


def _canonical_ipv6_literal(value: str, host: str) -> str:
    """``host`` came from a bracketed netloc (``[...]``); validate it as IPv6, or reject it
    outright -- there is no fallback to treating it as an ordinary hostname.
    """
    if host[:1] == "v":
        # RFC 3986 IPvFuture (e.g. "[v1.fe80::1]", or "[v1.example]" with no ':' at all). No real
        # IPv6 address starts with 'v' -- hex digits are 0-9/a-f only -- so this is an
        # unambiguous signal, not a guess.
        raise ValueError(f"origin {value!r}: IPvFuture literals aren't supported")
    if "%" in host:
        # A percent inside brackets means an IPv6 zone id (`fe80::1%eth0`), meaningful only on
        # the machine that assigned the zone, so it can never appear in a browser's Origin
        # header. Outside brackets a stray '%' is just an invalid character, caught by the
        # hostname branch instead of being mislabeled as a zone id here.
        raise ValueError(f"origin {value!r} must not include an IPv6 zone id")
    try:
        address = ipaddress.IPv6Address(host)
    except ValueError as exc:
        raise ValueError(f"origin {value!r} has an invalid IPv6 host: {exc}") from exc
    if address.ipv4_mapped is not None:
        # e.g. "::ffff:127.0.0.1" -- valid IPv6 syntax, but no browser ever sends this form.
        raise ValueError(
            f"origin {value!r} is an IPv4-mapped IPv6 address; use plain IPv4 instead"
        )
    return f"[{address.compressed}]"
