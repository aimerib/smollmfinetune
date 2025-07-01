# R6-3: STT Input Pipeline  
Status: **Not Started**  
Ring: R6
Created: 2025-06-30   
Related-Tasks: R4-13, R5-3, R6-1

---

## 1 · Goal  
Enable players to speak to characters. Build a speech-to-text (ASR) pipeline that transcribes microphone input, feeds it into the Narrative Engine, and completes the loop: **Player speech → ASR → Engine → Character TTS**.

---

## 2 · Context  
While the Agentic Loop Framework (R4-13) already supports text messages, spoken interaction is a core immersion feature. Whisper-large-v3 fine-tuned on in-game proper nouns offers strong accuracy with manageable latency when quantised.

---

## 3 · Acceptance Criteria  

| # | Requirement |
|---|-------------|
| **ASR Service** | `narrative_engine/audio/asr.py` wraps **OpenAI Whisper** (or SpeechBrain) with `transcribe(audio_bytes) -> str`. |
| | CLI `asr_server.py` exposes a gRPC / REST endpoint returning JSON `{ "text": "<transcript>", "segments": [...] }`. |
| **UI** | Chat interface gains a “🎤 Hold to Speak” button: press-&-hold records ≤ 15 s; on release sends WAV to ASR service. |
| | Shows interim “transcribing…” spinner; on success, inserts transcript as if the player typed it. |
| **Engine** | `PlatformRuntimeEngine` accepts new input event type `PlayerVoice(text, raw_audio)`; raw audio stored for analytics. |
| **Latency** | End-to-end (record stop → transcript ready) ≤ 800 ms on RTX 4090. |
| **TDD** | `tests/audio/test_asr.py` mocks 1-s sine-wave, asserts transcript == "", latency < 800 ms, error handling. |
| **Loop Demo** | `demo_voice_loop.py` records mic, sends to ASR, forwards transcript to NarrativeLLM, plays TTS reply via R5-8. |

---

## 4 · Implementation Notes  

* Package **VAD** (`webrtcvad`) client-side to trim leading/trailing silence before upload.  
* Whisper inference tips: fp16, 8-bit weight quantisation (`bitsandbytes`), `--condition_on_previous_text false` for speed.  
* Cache results of identical audio SHA-256 to skip duplicate transcriptions in testing.  
* For privacy, discard raw audio after 30 days or on player request.  
* Provide language auto-detect with fallback to English; store `lang` in transcript metadata.

---

## 5 · How to Test?  

1. **Unit** – Pass a prerecorded “test.wav” saying “hello world”; assert transcript matches.  
2. **E2E CI** – GitHub Action runs `demo_voice_loop.py` on sample audio, asserts no exceptions, latency budget met, and returned TTS clip length ≥ 95 % of ground truth.  
3. **Manual Smoke** – Local run; speak a sentence; see it appear in chat and hear the character reply audibly.

---

## 6 · References  
* R4-13 Agentic Loop input event flow 
* Whisper: Radford et al., 2023 – “Robust Speech Recognition via Large-Scale Weak Supervision”  
* webrtcvad – real-time voice-activity detection library
