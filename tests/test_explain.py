"""Tests for the LLM explanation layer (sahu65/services/explain.py).

The interesting assertions here are the safety ones: that the narrator cannot change a
verdict, that no image bytes can reach a provider, and that an unconfigured or failing
provider degrades to None instead of breaking a detection.
"""
import json

import pytest

from sahu65.services import explain as ex


def make_request(**overrides):
    base = dict(
        verdict="likely_ai",
        ai_probability=0.93,
        confidence="high",
        calibrated=True,
        warnings=[],
        signals={"classifier_probability": 0.93},
        thresholds={"T_lo": 0.35, "T_hi": 0.65},
        model_version="2026.10.0",
    )
    base.update(overrides)
    return ex.ExplanationRequest(**base)


class FakeProvider(ex.BaseProvider):
    """Stands in for a network provider and records what it was asked to say."""

    name = "fake"
    reply = "The detector found signs this image was made by AI."
    last_instance = None

    def __init__(self, **kwargs):
        super().__init__(api_key=kwargs.get("api_key", "k"),
                         model=kwargs.get("model", "fake-model"))
        self.seen_system = None
        self.seen_user = None
        FakeProvider.last_instance = self

    def complete(self, system, user):
        self.seen_system = system
        self.seen_user = user
        if type(self).fail_with:
            raise ex.ExplanationError(type(self).fail_with)
        return type(self).reply


FakeProvider.fail_with = None


@pytest.fixture
def fake_registered():
    """Registers FakeProvider as a test-only provider for the duration of a test."""
    FakeProvider.fail_with = None
    FakeProvider.last_instance = None
    ex.PROVIDERS["__test__"] = FakeProvider
    yield FakeProvider
    ex.PROVIDERS.pop("__test__", None)


def test_returns_none_when_no_provider_configured():
    assert ex.explain(make_request(), env={}) is None


def test_returns_none_when_chosen_provider_has_no_key():
    assert ex.explain(make_request(), provider="gemini", env={}) is None


def test_provider_failure_degrades_to_none(fake_registered):
    fake_registered.fail_with = "upstream 500"
    result = ex.explain(make_request(), provider="__test__", api_key="k", env={})
    assert result is None


def test_sanitize_replaces_certainty_claims_on_inconclusive():
    req = make_request(verdict="inconclusive", confidence="low")
    text = ex._sanitize("This is definitely AI generated.", req)
    assert "definitely" not in text.lower()
    assert "could not determine" in text.lower()


def test_sanitize_keeps_consistent_text_on_inconclusive():
    req = make_request(verdict="inconclusive", confidence="low")
    text = ex._sanitize("The detector could not tell either way.", req)
    assert "could not tell" in text.lower()


def test_sanitize_strips_markdown_scaffolding():
    req = make_request()
    text = ex._sanitize("- point one\n- point two", req)
    assert not text.startswith("-")


def test_build_facts_excludes_uncalibrated_thresholds():
    req = make_request(calibrated=False)
    facts = ex.build_facts(req)
    assert facts["thresholds_used"] is None
    assert facts["score_is_calibrated"] is False


def test_build_facts_includes_thresholds_when_calibrated():
    facts = ex.build_facts(make_request(calibrated=True))
    assert facts["thresholds_used"]["likely_ai_above"] == 0.65


def test_prompt_never_carries_image_data(fake_registered):
    out = ex.explain(make_request(), provider="__test__", api_key="k", env={})
    assert out["text"] == fake_registered.reply
    # The prompt is the entire contract with the provider: numbers only, no pixels.
    payload = fake_registered.last_instance.seen_user.lower()
    assert "verdict" in payload
    for banned in ("image_url", "base64", "pixel", "bytes", "filename", "sha256"):
        assert banned not in payload


def test_explain_result_carries_provider_and_model(fake_registered):
    out = ex.explain(make_request(), provider="__test__", api_key="k", env={})
    assert out["provider"] == "__test__"
    # A provider with no registered default model resolves to an empty string rather than
    # raising; the provider's own complete() then applies its own fallback.
    assert out["model"] == ""


def test_registered_providers_have_default_models():
    assert ex.DEFAULT_MODELS["gemini"]
    assert ex.DEFAULT_MODELS["groq"]
    for name in ex.DEFAULT_MODELS:
        assert issubclass(ex.PROVIDERS[name], ex.BaseProvider)


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        ex.explain(make_request(), provider="nope", api_key="k", env={})


def test_configured_reports_provider_without_key():
    info = ex.configured(env={"GEMINI_API_KEY": "secret-value"})
    assert info["available"] is True
    assert info["provider"] == "gemini"
    assert "secret-value" not in json.dumps(info)


def test_resolve_api_key_prefers_first_nonempty():
    assert ex.resolve_api_key("gemini", env={"GEMINI_API_KEY": "a", "GOOGLE_API_KEY": "b"}) == "a"
    assert ex.resolve_api_key("gemini", env={"GEMINI_API_KEY": "  "}) is None
    assert ex.resolve_api_key("gemini", env={"GOOGLE_API_KEY": "b"}) == "b"