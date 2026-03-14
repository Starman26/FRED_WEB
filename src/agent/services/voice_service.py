"""
voice_service.py - ElevenLabs TTS streaming service.
"""
import os
import logging
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # "George"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"

# Accumulate small ElevenLabs fragments (~128-512B) into ~4KB chunks to reduce SSE overhead
CHUNK_ACCUMULATION_TARGET = 4096


class VoiceService:
    """ElevenLabs TTS streaming wrapper."""

    def __init__(
        self,
        api_key: str,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        output_format: Optional[str] = None,
    ):
        from elevenlabs import ElevenLabs

        self._client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id or DEFAULT_VOICE_ID
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.output_format = output_format or DEFAULT_OUTPUT_FORMAT
        self.stability = 0.6
        self.similarity_boost = 0.75
        self.speed = 0.85

        logger.info(
            f"[VoiceService] Initialized: voice={self.voice_id}, "
            f"model={self.model_id}, format={self.output_format}, "
            f"speed={self.speed}, stability={self.stability}"
        )

    def stream_tts(
        self,
        text: str,
        voice_id: Optional[str] = None,
    ) -> Iterator[bytes]:
        """Stream TTS audio as accumulated MP3 byte chunks."""
        vid = voice_id or self.voice_id
        print(f"[VoiceService] stream_tts: text_len={len(text)}, voice={vid}, model={self.model_id}", flush=True)

        try:
            raw_stream = self._client.text_to_speech.stream(
                text=text,
                voice_id=vid,
                model_id=self.model_id,
                output_format=self.output_format,
                voice_settings={
                    "stability": self.stability,
                    "similarity_boost": self.similarity_boost,
                    "speed": self.speed,
                },
            )
            print(f"[VoiceService] .stream() returned: {type(raw_stream).__name__}", flush=True)
        except Exception as e:
            print(f"[VoiceService] .stream() FAILED: {type(e).__name__}: {e}", flush=True)
            raise

        buffer = b""
        fragment_count = 0
        total_bytes = 0
        chunks_yielded = 0

        for fragment in raw_stream:
            if not isinstance(fragment, bytes):
                print(f"[VoiceService] Unexpected fragment type: {type(fragment).__name__}", flush=True)
                if isinstance(fragment, str):
                    fragment = fragment.encode("utf-8")
                else:
                    continue

            fragment_count += 1
            total_bytes += len(fragment)
            buffer += fragment

            if fragment_count == 1:
                print(f"[VoiceService] First fragment: {len(fragment)} bytes", flush=True)

            while len(buffer) >= CHUNK_ACCUMULATION_TARGET:
                yield buffer[:CHUNK_ACCUMULATION_TARGET]
                buffer = buffer[CHUNK_ACCUMULATION_TARGET:]
                chunks_yielded += 1

        if buffer:
            yield buffer
            chunks_yielded += 1

        print(f"[VoiceService] Done: {fragment_count} fragments, {total_bytes} bytes, {chunks_yielded} chunks", flush=True)

        if fragment_count == 0:
            print("[VoiceService] ERROR: ElevenLabs returned 0 fragments!", flush=True)
