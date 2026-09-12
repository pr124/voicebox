"""Tests for the OpenAI-compatible speech API."""

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import backend.routes.openai_compat as openai_compat


def test_list_models_exposes_openai_compatible_ids():
    result = __import__("asyncio").run(openai_compat.list_models())

    assert result["object"] == "list"
    assert {item["id"] for item in result["data"]} == {
        "tts-1",
        "tts-1-hd",
        "gpt-4o-mini-tts",
    }


def test_apply_speed_changes_sample_count():
    samples = np.arange(10, dtype=np.float32)

    faster = openai_compat._apply_speed(samples, 2.0)
    slower = openai_compat._apply_speed(samples, 0.5)

    assert len(faster) == 5
    assert len(slower) == 20


def test_speech_request_validates_speed():
    with pytest.raises(ValidationError):
        openai_compat.SpeechRequest(model="tts-1", input="Hello", speed=5.0)


@pytest.mark.asyncio
async def test_create_speech_returns_wav(monkeypatch):
    async def fake_resolve_voice_prompt(voice, engine, db):
        assert voice == "alloy"
        assert engine == "kokoro"
        return {"preset_voice_id": "af_alloy"}

    async def fake_generate_audio(**kwargs):
        assert kwargs["text"] == "Hello from Voicebox."
        return np.zeros(240, dtype=np.float32), 24000

    monkeypatch.setattr(
        openai_compat,
        "_resolve_voice_prompt",
        fake_resolve_voice_prompt,
    )
    monkeypatch.setattr(openai_compat, "_generate_audio", fake_generate_audio)

    response = await openai_compat.create_speech(
        openai_compat.SpeechRequest(
            model="tts-1",
            input="Hello from Voicebox.",
        ),
        None,
    )

    assert response.media_type == "audio/wav"
    assert response.body[:4] == b"RIFF"
    assert response.headers["x-audio-sample-rate"] == "24000"


@pytest.mark.asyncio
async def test_create_speech_rejects_unknown_model():
    with pytest.raises(HTTPException) as error:
        await openai_compat.create_speech(
            openai_compat.SpeechRequest(model="unknown", input="Hello"),
            None,
        )

    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_create_speech_rejects_unsupported_format():
    with pytest.raises(HTTPException) as error:
        await openai_compat.create_speech(
            openai_compat.SpeechRequest(
                model="tts-1",
                input="Hello",
                response_format="mp3",
            ),
            None,
        )

    assert error.value.status_code == 400
