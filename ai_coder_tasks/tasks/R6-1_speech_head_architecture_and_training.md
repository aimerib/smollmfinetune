# R6-1: Speech Head Architecture & Training  
Status: **Not Started**  
Ring: R6  
Created: 2025-06-30  
Related-Tasks: R4-3, R4-10, R4-11, R6-2

---

## 1 · Goal  
Extend the **Narrative-LLM** with a fourth output head, **SpeechHead**, that produces mel-spectrogram frames conditioned on the same hidden states used by the existing Generation, Control, and Memory heads. The result is a **quad-head** model capable of end-to-end text-to-speech synthesis inside the narrative engine.

---

## 2 · Context  
Our current triple-head architecture was laid out in R4-1 and R4-3 . Adding speech requires three big changes:

1. **Model code** – a new linear/convolutional head and an extended config.  
2. **Data pipeline** – paired text + audio batches and automatic spectrogram generation.  
3. **Loss function** – integrate an L1/L2 *SpectrogramLoss* into what becomes **QuadHeadLoss**.

Because LJ-Speech and similar corpora are small (~24 h) we will begin with single-speaker English and later broaden the dataset in R6.

---

## 3 · Acceptance Criteria  

| # | Requirement |
|---|-------------|
| **Model** | `NarrativeLLM` in `narrative_engine/model.py` has a `speech_head` (e.g. `nn.Conv1d(hidden_dim, n_mel_channels, kernel_size=1)`). |
| | `NarrativeLLMConfig` adds `n_mel_channels`, `mel_loss_weight`, `speech_head_type`. |
| **Training** | `run_sft.py` accepts `--audio_dir` / `--audio_manifest.jsonl`; dataset yields `(input_ids, attention_mask, mel_targets)`. |
| | New `SpectrogramLoss` (Smooth-L1 default) lives in `narrative_engine/loss.py`; incorporated into `QuadHeadLoss(text, control, memory, speech)`. |
| | CLI flag `--enable_speech_head` toggles the new functionality. |
| **Evaluation** | `eval_speech.py` computes mel-cepstral distortion (MCD) and saves one validation sample every epoch for listening tests. |
| **TDD** | Red-Green tests in `tests/narrative_engine/test_speech_head.py`: import, instantiation, forward pass shape `(B, T, n_mel)`, loss ≠ nan, back-prop works. |
| **Docs** | Add section *Quad-Head Architecture* to `docs/architecture.md`. |

---

## 4 · Implementation Notes  

* Start with a lightweight Post-Net stack (5× Conv1d → BatchNorm → Tanh) after the linear projection to refine spectrograms.  
* Use `torchaudio.transforms.MelSpectrogram` with Libri-TTS defaults (n_fft = 1024, hop = 256, win = 1024, n_mels = 80).  
* Gradually warm-up the SpeechHead: freeze it for the first N training steps to let text convergence stabilise.  
* Provide `collate_fn_speech` that right-pads both tokens **and** mel frames and produces `speech_loss_mask`.  
* Loss weighting suggestion:

```yaml
QuadHeadLoss weights:
  text:       1.0
  control:    0.5
  memory:     0.2
  speech:     3.0   # higher because of larger target magnitude
````

* Every epoch, dump `sample_epoch_{N}.pt` containing `(text, pred_mel, gt_mel)` for later MOS studies.

---

## 5 · How to Test?

1. **Unit tests** (see Acceptance Criteria).
2. **Quick-start notebook** `notebooks/speech_head_smoke_test.ipynb` that:

   * Loads `NarrativeLLM` with `--enable_speech_head`.
   * Feeds “Hello world” dummy input, decodes mel → waveform via HiFi-GAN (see R5-8).
   * Plays the audio in Jupyter to confirm audible output.
3. **CI job** runs `pytest -m "speech_head"` on every PR.
4. **Manual** – Train for 1 epoch on LJ-Speech; verify MCD < 8 dB and spectrogram visually resembles ground truth.

---

## 6 · References

* R4-1 Model scaffolding (baseline class layout)&#x20;
* R4-3 Dual-Head Loss design (loss routing pattern)&#x20;
* LJ-Speech dataset (public domain)
* StyleTTS2 paper for small-footprint TTS baseline
