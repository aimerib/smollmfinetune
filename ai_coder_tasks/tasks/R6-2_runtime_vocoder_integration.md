# R6-2: Runtime Vocoder Integration  
Status: **Not Started**  
Ring: R6
Created: 2025-06-30  
Related-Tasks: R4-10, R6-1, R5-3

---

## 1 · Goal  
Attach a low-latency neural vocoder (default: **HiFi-GAN V1**) to the SpeechHead so the platform can stream audible dialogue in real time.

---

## 2 · Context  
The Production Inference Engine (R4-10) already hits < 200 ms for triple-head text output :contentReference. With SpeechHead active we must stay under **100 ms** from mel frames to audio to keep total latency acceptable for live interaction and the Director’s View “Listen” button (R5-3).

---

## 3 · Acceptance Criteria  

| # | Requirement |
|---|-------------|
| **Code** | `narrative_engine/audio/vocoder.py` exposes `Vocoder.load(model_name)` and `Vocoder.mel_to_audio(mel: Tensor) -> FloatTensor`. |
| | Supports **HiFi-GAN** out-of-the-box and allows config override for UnivNet or RVC. |
| **Integration** | `PlatformRuntimeEngine` converts SpeechHead output to a *streaming* byte iterator (`Iterable[bytes]`). |
| | Response schema gains optional field `audio_url` (presigned S3) **or** WS stream ID. |
| **UI** | Director’s Chair adds “🎧 Listen” button next to each agent utterance; clicking plays the generated clip via HTML5 `<audio>` element. |
| **Performance** | Benchmark script `bench_vocoder.py` shows **< 100 ms** mel→wave for a 2-second sample on RTX 4090. |
| **TDD** | `tests/audio/test_vocoder.py` asserts successful load, correct sample-rate output (24 kHz), and latency budget. |
| **Docs** | Update `docs/audio_pipeline.md` with installation, usage, and troubleshooting.

---

## 4 · Implementation Notes  

* Use `torch.hub.load("snakers4/silero-models", "hifi_gan")` to fetch the pre-trained weights in CI.  
* Default sample rate 24 kHz, 16-bit PCM.  
* Provide `--vocoder_device cuda:0` flag; fall back to CPU if unavailable (expect ~4× slower).  
* Add simple FIFO ring buffer so audio starts streaming after first 15 mel frames (~60 ms).  
* Provide utility `save_to_wav(filename, audio)` for offline debugging.  

---

## 5 · How to Test?  

1. **Unit test** loads vocoder, passes random mel (80×80) and asserts output shape `(1, ≈3840)`.  
2. **Integration** – Launch Inference Engine with `--enable_speech`; send “Hello”; receive `audio_url`; play it; assert HTTP 200.  
3. **Latency** – Run `bench_vocoder.py` in CI nightly; fail if p50 > 100 ms or p99 > 150 ms.

---

## 6 · References  
* R4-10 Inference Engine architecture 
* HiFi-GAN: Kong et al., 2020 – “HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis”
