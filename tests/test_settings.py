"""Tests for breeze_infer/settings.py: the launch-configuration parser.

Covers build_parser()/settings_from_args() -- the new home for command-line
parsing (T016). Tests for runtime behavior (ApiSettings, configure_compile_cache,
_load_app, FastBreezeStreamingRuntime properties) stay in test_runtime_flags.py;
none of those exercise a parser, so nothing moved.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from breeze_infer.settings import build_parser, settings_from_args


def test_defaults(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path)])

    assert settings.model_path == tmp_path
    assert settings.host == "127.0.0.1"
    assert settings.port == 8080
    assert settings.ws_port == 8081
    assert settings.cors == ()
    assert settings.split_chars == 600
    assert settings.chunk_first == 1
    assert settings.chunk_max == 25
    assert settings.voices_dir == Path("voices")
    assert settings.fast_all is None
    assert settings.fast_text_encoder is False
    assert settings.fast_backbone_prefill is False
    assert settings.fast_backbone_decode is False
    assert settings.fast_depth_decoder is False
    assert settings.fast_codec is False
    assert settings.attn_implementation == "eager"
    assert settings.compile_cache_dir is None


def test_settings_is_frozen(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path)])
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.port = 9090  # type: ignore[misc]


def test_host_and_port_overrides(tmp_path: Path) -> None:
    settings = settings_from_args(
        [str(tmp_path), "--host", "0.0.0.0", "--port", "9000"]
    )
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    # ws_port still derives from the overridden port.
    assert settings.ws_port == 9001


def test_ws_port_explicit(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path), "--ws-port", "9500"])
    assert settings.ws_port == 9500


def test_ws_port_disabled(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path), "--ws-port", "disabled"])
    assert settings.ws_port is None


def test_ws_port_garbage_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--ws-port", "not-a-port"])
    assert "ws-port" in capsys.readouterr().err


def test_ws_port_equal_to_port_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--port", "9000", "--ws-port", "9000"])
    assert "differ" in capsys.readouterr().err


@pytest.mark.parametrize("port", [0, -1, 65536, 100000])
def test_port_out_of_range_rejected(
    tmp_path: Path, port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--port", str(port)])
    assert "--port" in capsys.readouterr().err


@pytest.mark.parametrize("ws_port", [0, -1, 65536])
def test_ws_port_out_of_range_rejected(
    tmp_path: Path, ws_port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--ws-port", str(ws_port)])
    assert "--ws-port" in capsys.readouterr().err


def test_cors_absent_is_off(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path)])
    assert settings.cors == ()


def test_cors_bare_flag_allows_any_origin(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path), "--cors"])
    assert settings.cors == ("*",)


def test_cors_allowlist_is_trimmed(tmp_path: Path) -> None:
    settings = settings_from_args(
        [str(tmp_path), "--cors", " https://a.example , https://b.example "]
    )
    assert settings.cors == ("https://a.example", "https://b.example")


def test_cors_wildcard_mixed_with_others_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "*,https://a.example"])
    assert "*" in capsys.readouterr().err


def test_split_chars_default_and_override(tmp_path: Path) -> None:
    assert settings_from_args([str(tmp_path)]).split_chars == 600
    settings = settings_from_args([str(tmp_path), "--split-chars", "0"])
    assert settings.split_chars == 0


def test_negative_split_chars_rejected_not_clamped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--split-chars", "-1"])
    assert "--split-chars" in capsys.readouterr().err


def test_chunk_first_clamped_to_chunk_max(tmp_path: Path) -> None:
    settings = settings_from_args(
        [str(tmp_path), "--chunk-first", "100", "--chunk-max", "25"]
    )
    assert settings.chunk_first == 25
    assert settings.chunk_max == 25


@pytest.mark.parametrize("flag,value", [("--chunk-first", "0"), ("--chunk-max", "0")])
def test_chunk_sizes_below_one_rejected(
    tmp_path: Path, flag: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), flag, value])
    assert flag in capsys.readouterr().err


def test_voices_dir_override(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path), "--voices-dir", "my-voices"])
    assert settings.voices_dir == Path("my-voices")


def test_fast_all_tri_state(tmp_path: Path) -> None:
    assert settings_from_args([str(tmp_path)]).fast_all is None
    assert settings_from_args([str(tmp_path), "--fast-all"]).fast_all is True
    assert settings_from_args([str(tmp_path), "--no-fast-all"]).fast_all is False


def test_individual_fast_flags(tmp_path: Path) -> None:
    settings = settings_from_args(
        [
            str(tmp_path),
            "--fast-text-encoder",
            "--fast-backbone-prefill",
            "--fast-backbone-decode",
            "--fast-depth-decoder",
            "--fast-codec",
        ]
    )
    assert settings.fast_text_encoder is True
    assert settings.fast_backbone_prefill is True
    assert settings.fast_backbone_decode is True
    assert settings.fast_depth_decoder is True
    assert settings.fast_codec is True


def test_attn_implementation_choices(tmp_path: Path) -> None:
    assert settings_from_args([str(tmp_path)]).attn_implementation == "eager"
    settings = settings_from_args(
        [str(tmp_path), "--attn-implementation", "sdpa"]
    )
    assert settings.attn_implementation == "sdpa"


def test_attn_implementation_rejects_flash_attention_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args(
            [str(tmp_path), "--attn-implementation", "flash_attention_2"]
        )
    assert "--attn-implementation" in capsys.readouterr().err


def test_compile_cache_dir_override(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cc"
    settings = settings_from_args(
        [str(tmp_path), "--compile-cache-dir", str(cache_dir)]
    )
    assert settings.compile_cache_dir == cache_dir


def test_model_path_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([])
    assert "model_path" in capsys.readouterr().err


def test_build_parser_returns_argument_parser() -> None:
    import argparse

    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_settings_never_reads_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Principle III: env resolution (e.g. TORCHINDUCTOR_CACHE_DIR) is the
    composition root's job, not the parser's -- so an unset compile cache dir
    must stay None regardless of what's in the environment."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/should/not/be/read")
    settings = settings_from_args([str(tmp_path)])
    assert settings.compile_cache_dir is None


@pytest.mark.parametrize("value", ["", ",", " , "])
def test_cors_with_an_empty_allowlist_is_rejected(
    tmp_path: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", value])
    assert "empty" in capsys.readouterr().err


def test_cors_wildcard_repeated_collapses_to_single_star(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path), "--cors", "*,*"])
    assert settings.cors == ("*",)


def test_cors_entries_are_deduped_preserving_order(tmp_path: Path) -> None:
    settings = settings_from_args(
        [
            str(tmp_path),
            "--cors",
            "https://b.example,https://a.example,https://b.example",
        ]
    )
    assert settings.cors == ("https://b.example", "https://a.example")


def test_cors_scheme_and_host_are_lowercased(tmp_path: Path) -> None:
    settings = settings_from_args(
        [str(tmp_path), "--cors", "HTTPS://Example.COM:8443"]
    )
    assert settings.cors == ("https://example.com:8443",)


@pytest.mark.parametrize(
    "entry",
    [
        "https://a.example/",
        "https://a.example/path",
        "https://a.example?x=1",
        "https://a.example#frag",
    ],
)
def test_cors_origin_with_path_query_or_fragment_rejected(
    tmp_path: Path, entry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", entry])
    # The message names the offending entry so an operator can find it in a
    # long allowlist.
    assert entry in capsys.readouterr().err


def test_cors_origin_trailing_slash_suggests_the_value_without_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "https://a.example/"])
    assert "https://a.example'" in capsys.readouterr().err


@pytest.mark.parametrize("entry", ["a.example", "ftp://a.example", "//a.example"])
def test_cors_origin_missing_or_wrong_scheme_rejected(
    tmp_path: Path, entry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", entry])
    assert entry in capsys.readouterr().err


def test_model_path_must_be_an_existing_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(SystemExit):
        settings_from_args([str(missing)])
    assert "model_path" in capsys.readouterr().err


def test_model_path_that_is_a_file_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a_file = tmp_path / "not-a-directory"
    a_file.write_text("x")
    with pytest.raises(SystemExit):
        settings_from_args([str(a_file)])
    assert "model_path" in capsys.readouterr().err


def test_ws_port_derived_out_of_range_names_the_derived_port(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--port", "65535"])
    err = capsys.readouterr().err
    assert "derived" in err
    assert "--ws-port" in err


@pytest.mark.parametrize(
    "value", ["+8080", "8080.0", " 8080", "8080 ", "0x1f90", "1e3", "-1"]
)
def test_port_rejects_non_strict_digit_forms(
    tmp_path: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--port", value])
    assert "--port" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["+9000", "9000.0", " 9000", "0x2328"])
def test_ws_port_rejects_non_strict_digit_forms(
    tmp_path: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--ws-port", value])
    assert "--ws-port" in capsys.readouterr().err


def test_cors_ipv6_origin_keeps_its_brackets(tmp_path):
    settings = settings_from_args([str(tmp_path), "--cors", "http://[::1]:8000"])
    assert settings.cors == ("http://[::1]:8000",)


def test_cors_origin_with_userinfo_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "https://user:pw@a.example"])
    assert "userinfo" in capsys.readouterr().err


@pytest.mark.parametrize("entry", ["https://a.example:0", "https://a.example:99999"])
def test_cors_origin_bad_or_out_of_range_port_rejected(
    tmp_path: Path, entry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", entry])
    assert entry in capsys.readouterr().err


@pytest.mark.parametrize(
    ("entry", "expected"),
    [("http://a.example:80", "http://a.example"), ("https://a.example:443", "https://a.example")],
)
def test_cors_default_port_is_dropped(tmp_path: Path, entry: str, expected: str) -> None:
    settings = settings_from_args([str(tmp_path), "--cors", entry])
    assert settings.cors == (expected,)


def test_cors_idn_host_rejected_suggests_punycode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """c1: a browser's `Origin` header is always punycode, never raw Unicode, so this module
    doesn't guess at an IDNA conversion -- the operator must already pass the punycode form."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "https://café.example"])
    assert "punycode" in capsys.readouterr().err


def test_cors_punycode_idn_host_accepted(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path), "--cors", "https://xn--caf-dma.example"])
    assert settings.cors == ("https://xn--caf-dma.example",)


def test_cors_underscore_in_hostname_allowed(tmp_path: Path) -> None:
    """c2: docker-compose service names (e.g. `backend_service`) are common as `--cors` hosts
    and aren't valid DNS labels, but must not be rejected."""
    settings = settings_from_args(
        [str(tmp_path), "--cors", "http://backend_service.local:8000"]
    )
    assert settings.cors == ("http://backend_service.local:8000",)


def test_cors_ipv4_mapped_ipv6_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """c3: `::ffff:a.b.c.d` is valid IPv6 syntax, but no browser ever sends this form -- an
    allowlist entry using it could never match a real `Origin` header."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "http://[::ffff:127.0.0.1]"])
    assert "IPv4-mapped" in capsys.readouterr().err


@pytest.mark.parametrize(
    "entry",
    ["https://a.example\t.evil.example", "https://a\n.example", "https://a .example"],
)
def test_cors_control_character_or_whitespace_rejected(
    tmp_path: Path, entry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """c4: `urlsplit` silently strips a tab/CR/LF rather than rejecting them, which could let
    `"https://a.example\\t.evil.example"` parse as the (allowed) `a.example` while actually
    meaning something else entirely; every control character and whitespace must be rejected
    up front instead."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", entry])
    assert "control characters or whitespace" in capsys.readouterr().err


@pytest.mark.parametrize("entry", ["https://evil.example.2130706433", "https://evil.example.0x7f000001"])
def test_cors_numeric_or_hex_last_label_rejected(
    tmp_path: Path, entry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """c6: some legacy URL parsers treat an all-numeric or hex-looking trailing label as an
    encoded IP address, a known trick for smuggling a real destination past a hostname
    allowlist; a real DNS label is never purely numeric or hex-prefixed."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", entry])
    assert "numeric" in capsys.readouterr().err


def test_cors_ipvfuture_literal_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """c6: `[v1....]` (RFC 3986 IPvFuture) is bracketed literal syntax no browser ever sends."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "http://[v1.fe80::1]"])
    assert "IPvFuture" in capsys.readouterr().err


def test_cors_bare_percent_in_hostname_is_not_reported_as_a_zone_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """c7: "zone id" only makes sense for a '%' *inside brackets* (an IPv6 literal); a stray '%'
    in a plain hostname is just an invalid character, not a zone id."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "http://a%25b.example"])
    err = capsys.readouterr().err
    assert "zone id" not in err
    assert "letters, digits" in err


def test_cors_ipv6_uppercase_and_expanded_form_canonicalizes(tmp_path: Path) -> None:
    settings = settings_from_args(
        [
            str(tmp_path),
            "--cors",
            "http://[0:0:0:0:0:0:0:1]:8000,http://[FE80::1]:8000",
        ]
    )
    assert settings.cors == ("http://[::1]:8000", "http://[fe80::1]:8000")


def test_cors_ipv4_non_dotted_quad_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "http://127.1"])
    assert "dotted-quad" in capsys.readouterr().err


def test_cors_ipv6_zone_id_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "http://[fe80::1%eth0]"])
    assert "zone id" in capsys.readouterr().err


def test_cors_wildcard_host_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "https://*.example"])
    assert "wildcard" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["８０８０", "٨٠٨٠"])
def test_port_rejects_unicode_digits(
    tmp_path: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--port", value])
    assert "--port" in capsys.readouterr().err


def test_cors_incomplete_ipv6_bracket_error_names_the_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "http://[::1"])
    assert "http://[::1" in capsys.readouterr().err


@pytest.mark.parametrize("entry", ["https://a.example?", "https://a.example#"])
def test_cors_origin_with_empty_query_or_fragment_rejected(
    tmp_path: Path, entry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", entry])
    assert entry in capsys.readouterr().err


# --------------------------------------------------- review-agent final pass (fuzzed 1M inputs)


def test_cors_bracketed_ipvfuture_without_a_colon_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """issue 1: `[v1.example]` has no ':' left once brackets are stripped (unlike
    `[v1.fe80::1]`), so a check keyed on "is there a ':' in the stripped host" let it slip
    through as the plain hostname `v1.example`. Anything inside brackets must be validated as an
    IP literal, never reinterpreted as an ordinary hostname."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "http://[v1.example]"])
    assert "IPvFuture" in capsys.readouterr().err


@pytest.mark.parametrize("entry", ["http://evil.0x", "http://0x"])
def test_cors_bare_0x_last_label_rejected(
    tmp_path: Path, entry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """issue 2: the hex check required at least one digit after `"0x"`, letting the bare label
    `"0x"` itself -- just as much a smuggled-IP trick as `"0x7f000001"` -- through."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", entry])
    assert "numeric" in capsys.readouterr().err


@pytest.mark.parametrize(
    "entry", ["https://evil.example.2130706433.", "http://0x7f000001."]
)
def test_cors_numeric_or_hex_last_label_with_trailing_dot_rejected(
    tmp_path: Path, entry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """issue 2 (first review-agent pass): a trailing '.' (FQDN absolute-name syntax) left the
    last label empty after `rsplit(".", 1)`, silently bypassing the numeric/hex check entirely.
    Superseded by the general empty-label rejection (second-to-last pass, issue 1,
    `test_cors_empty_label_rejected`), which now catches the trailing dot itself, before the
    numeric/hex check ever runs -- this only confirms the entry is still rejected."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", entry])
    assert "empty label" in capsys.readouterr().err


def test_cors_origin_error_is_prefixed_with_the_flag_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """issue 3: `canonical_origin`'s own messages are deliberately flag-agnostic (it's also used
    on the request path, where "--cors" would be meaningless); `settings.py` is the one place
    that knows this value came from `--cors`, so it's the one that must say so."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "ftp://a.example"])
    err = capsys.readouterr().err
    assert "--cors: " in err
    assert "ftp://a.example" in err


@pytest.mark.parametrize("entry", ["http://a.example:", "http://[::1]:"])
def test_cors_origin_with_empty_port_rejected(
    tmp_path: Path, entry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """issue 5: `urlsplit(...).port` silently returns `None` for a trailing ':' with nothing
    after it -- indistinguishable from no port at all -- rather than raising."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", entry])
    assert "port" in capsys.readouterr().err


@pytest.mark.parametrize("entry", ["http://a.example:0080", "http://[::1]:0080"])
def test_cors_origin_port_with_leading_zero_rejected(
    tmp_path: Path, entry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """issue 5: `urlsplit(...).port` is an `int`, so `"0080"` and `"80"` are indistinguishable by
    the time this module would see them -- silently normalizing away a leading zero that a real
    browser's `Origin` header would never carry in the first place."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", entry])
    assert "leading zero" in capsys.readouterr().err


def test_cors_c1_control_character_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """issue 10: the control-character check only ever covered C0 (0x00-0x1F) and DEL (0x7F);
    a C1 control (0x80-0x9F, also Unicode category "Cc") must be rejected the same way."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "https://a\x80b.example"])
    assert "control characters or whitespace" in capsys.readouterr().err


# ------------------------------------------------- review-agent second-to-last pass (800k fuzz)


@pytest.mark.parametrize(
    "entry",
    [
        "http://.example.com",
        "http://example..com",
        "http://example.com.",
        "http://example.com..",
    ],
)
def test_cors_empty_label_rejected(
    tmp_path: Path, entry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """issue 1: a leading '.', '..', or a trailing '.' (even just one -- the FQDN absolute-name
    form, real DNS syntax, but no browser ever sends it in an `Origin` header) all produce an
    empty label. `_HOST_CHARS_RE` allows '.' as a character but says nothing about *where*, so
    all of these previously slipped through -- and a single trailing '.' had silently defeated
    the numeric/hex last-label check from the previous pass (issue 2 there)."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", entry])
    assert "empty label" in capsys.readouterr().err


def test_cors_kelvin_sign_host_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """issue 5: U+212A KELVIN SIGN lowercases to the plain ASCII "k" -- `SplitResult.hostname`
    lowercases internally, so checking `str.isascii()` only *after* that (as `_canonical_host`
    alone did) would miss exactly this: a non-ASCII origin quietly laundering itself into one
    that looks perfectly ordinary once lowercased."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "https://aKxample.com"])
    err = capsys.readouterr().err
    assert "ASCII" in err
    assert "punycode" in err


def test_cors_non_ascii_digit_port_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """issue 6: `str.isdigit()` (used for the leading-zero check) is also true for non-ASCII
    decimal digits (e.g. full-width '０８０'); `int()`/`SplitResult.port` parse them
    the same as their ASCII equivalents, so a port spelled entirely in full-width digits would
    silently canonicalize to an ordinary-looking port number -- a form no browser's `Origin`
    header would ever carry."""
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "http://a.example:０８０"])
    assert "ASCII" in capsys.readouterr().err
