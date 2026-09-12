"""OpenAI-compatible speech endpoints."""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

_MODEL_MAP: dict[str, tuple[str, str]] = {
    "tts-1": ("kokoro", "default"),
    "tts-1-hd": ("qwen", "1.7B"),
    "gpt-4o-mini-tts": ("qwen_custom_voice", "0.6B"),
}

_OPENAI_KOKORO_VOICES = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    "shimmer": "af_sky",
}

_OPENAI_QWEN_VOICES = {
    "alloy": "Ryan",
    "echo": "Aiden",
    "fable": "Dylan",
    "onyx": "Uncle_Fu",
    "nova": "Vivian",
    "shimmer": "Serena",
}

_SUPPORTED_RESPONSE_FORMATS = {"wav", "flac", "pcm"}


class SpeechRequest(BaseModel):
    """Request body compatible with the OpenAI speech endpoint."""

    model: str = Field(..., min_length=1, max_length=100)
    input: str = Field(..., min_length=1, max_length=50000)
    voice: str = Field(default="alloy", min_length=1, max_length=100)
    response_format: str = Field(default="wav", min_length=1, max_length=20)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = Field(default=None, max_length=500)
    language: str = Field(
        default="en",
        pattern="^(zh|en|ja|ko|de|fr|ru|pt|es|it|he|ar|da|el|fi|hi|ms|nl|no|pl|sv|sw|tr)$",
    )


@router.post("/v1/audio/speech")
async def create_speech(
    data: SpeechRequest,
    db: Session = Depends(get_db),
) -> Response:
    """Generate speech and return audio bytes in an OpenAI-compatible shape."""
    mapping = _MODEL_MAP.get(data.model)
    if mapping is None:
        supported = ", ".join(_MODEL_MAP)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{data.model}'. Supported models: {supported}.",
        )

    response_format = data.response_format.casefold()
    if response_format not in _SUPPORTED_RESPONSE_FORMATS:
        supported = ", ".join(sorted(_SUPPORTED_RESPONSE_FORMATS))
        raise HTTPException(
            status_code=400,
            detail=(f"Unsupported response_format '{data.response_format}'. Voicebox currently supports: {supported}."),
        )

    engine, model_size = mapping

    try:
        voice_prompt = await _resolve_voice_prompt(data.voice, engine, db)
        audio, sample_rate = await _generate_audio(
            text=data.input,
            voice_prompt=voice_prompt,
            engine=engine,
            model_size=model_size,
            language=data.language,
            instructions=data.instructions,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("OpenAI-compatible speech generation failed")
        raise HTTPException(status_code=500, detail="Voice generation failed.") from exc

    audio = _apply_speed(audio, data.speed)
    media_type, body, extension = _encode_audio(audio, sample_rate, response_format)

    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="speech.{extension}"',
            "X-Audio-Sample-Rate": str(sample_rate),
        },
    )


@router.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List the model IDs understood by this compatibility layer."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "voicebox",
            }
            for model_id in _MODEL_MAP
        ],
    }


async def _resolve_voice_prompt(
    voice: str,
    engine: str,
    db: Session,
) -> dict[str, Any]:
    """Resolve a Voicebox profile or a built-in compatibility voice."""
    from ..services import profiles
    from ..services.profiles import get_profile_orm_by_name_or_id

    profile = get_profile_orm_by_name_or_id(voice, db)
    if profile is not None:
        try:
            return await profiles.create_voice_prompt_for_profile(
                str(profile.id),
                db,
                use_cache=True,
                engine=engine,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized_voice = voice.casefold()

    if engine == "kokoro":
        from ..backends.kokoro_backend import KOKORO_VOICES

        known_voices = {}
        for voice_id, display_name, _gender, _language in KOKORO_VOICES:
            known_voices[voice_id.casefold()] = voice_id
            known_voices[display_name.casefold()] = voice_id

        voice_id = _OPENAI_KOKORO_VOICES.get(normalized_voice) or known_voices.get(normalized_voice)
        if voice_id is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown voice '{voice}'. Use an OpenAI-compatible voice "
                    "name, a Kokoro voice id, or a Voicebox profile name."
                ),
            )
        return {
            "voice_type": "preset",
            "preset_engine": "kokoro",
            "preset_voice_id": voice_id,
        }

    if engine == "qwen_custom_voice":
        from ..backends.qwen_custom_voice_backend import QWEN_CUSTOM_VOICES

        known_voices = {
            speaker.casefold(): speaker for speaker, _name, _gender, _language, _description in QWEN_CUSTOM_VOICES
        }
        speaker = _OPENAI_QWEN_VOICES.get(normalized_voice) or known_voices.get(normalized_voice)
        if speaker is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown voice '{voice}'. Use an OpenAI-compatible voice, "
                    "a Qwen CustomVoice speaker, or a Voicebox preset profile."
                ),
            )
        return {
            "voice_type": "preset",
            "preset_engine": "qwen_custom_voice",
            "preset_voice_id": speaker,
        }

    from ..mcp_server.resolve import resolve_profile

    default_profile = resolve_profile(None, None, db)
    if default_profile is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "tts-1-hd needs a cloned Voicebox profile. Pass its profile "
                "name or id as voice, or configure a default playback voice."
            ),
        )

    try:
        return await profiles.create_voice_prompt_for_profile(
            str(default_profile.id),
            db,
            use_cache=True,
            engine=engine,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _generate_audio(
    *,
    text: str,
    voice_prompt: dict[str, Any],
    engine: str,
    model_size: str,
    language: str,
    instructions: str | None,
) -> tuple[np.ndarray, int]:
    """Run the existing Voicebox chunked generation pipeline in memory."""
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        ensure_model_cached_or_raise,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..utils.audio import normalize_audio
    from ..utils.chunked_tts import generate_chunked

    await ensure_model_cached_or_raise(engine, model_size)
    await load_engine_model(engine, model_size)

    trim_fn = None
    runaway_detector = None
    if engine_needs_trim(engine):
        from ..utils.audio import trim_tts_output

        trim_fn = trim_tts_output
    if engine_retries_runaway(engine):
        from ..utils.audio import has_tts_runaway

        runaway_detector = has_tts_runaway

    tts_model = get_tts_backend_for_engine(engine)
    audio, sample_rate = await generate_chunked(
        tts_model,
        text,
        voice_prompt,
        language=language,
        seed=None,
        instruct=instructions,
        max_chunk_chars=800,
        crossfade_ms=50,
        trim_fn=trim_fn,
        runaway_detector=runaway_detector,
    )
    return normalize_audio(audio), sample_rate


def _apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Adjust playback speed without adding a runtime dependency."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if speed == 1.0 or samples.size < 2:
        return samples

    target_size = max(1, round(samples.size / speed))
    source_positions = np.linspace(0.0, 1.0, samples.size)
    target_positions = np.linspace(0.0, 1.0, target_size)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _encode_audio(
    audio: np.ndarray,
    sample_rate: int,
    response_format: str,
) -> tuple[str, bytes, str]:
    """Encode generated samples into a supported OpenAI response format."""
    if response_format == "pcm":
        samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        body = (samples * 32767.0).astype("<i2").tobytes()
        return "audio/pcm", body, "pcm"

    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.asarray(audio, dtype=np.float32),
        sample_rate,
        format=response_format.upper(),
        subtype="PCM_16",
    )
    return f"audio/{response_format}", buffer.getvalue(), response_format
