# CLI-First Training Loop Design

## Goal

Refactor this fork toward a CLI-first GPT-SoVITS workflow for Linux training and macOS inference. The first implementation must provide a complete training loop without Gradio/WebUI dependence while preserving a safe fallback path during migration.

The target workflow is:

1. Prepare a speaker dataset from clean vocal audio or a prebuilt annotation list.
2. Optionally call an externally installed `audio-separator` command for vocal separation.
3. Slice audio, run Faster-Whisper ASR when needed, and generate GPT-SoVITS training artifacts.
4. Train SoVITS and GPT weights on Linux/CUDA.
5. Run local inference from trained weights on Linux or macOS.

## Non-Goals

- Do not keep beginner-oriented GUI flows.
- Do not support Windows scripts, PowerShell installers, or `.bat` launchers.
- Do not support `v1` in the CLI path.
- Do not implement `v3` or `v4` in the first CLI version, but keep the architecture extensible enough to add them later.
- Do not vendor or install `python-audio-separator`; the CLI only wraps an already installed `audio-separator` command.
- Do not retain UI i18n as a runtime dependency.
- Do not keep Korean or Cantonese text frontends in the supported training/inference path.

## Recommended Approach

Use a strangler migration:

1. Add a new CLI package and configuration layer.
2. Route the CLI through existing lower-level dataset, training, and inference primitives.
3. Extract logic away from `webui.py` where needed.
4. Verify the CLI training loop.
5. Remove GUI, Windows, broad i18n, and unused language/version branches after the CLI path is working.

This avoids breaking the current training stack before the replacement path is tested.

## Supported Platforms

- Training target: Linux with NVIDIA CUDA, primarily Fedora + RTX 4060 8GB.
- Inference target: Linux CUDA and macOS Apple Silicon CPU/MPS-compatible execution where supported by existing PyTorch paths.
- macOS training is not a primary target.

## Supported Model Versions

The first CLI version supports:

- `v2`
- `v2Pro`
- `v2ProPlus`

The default version is `v2ProPlus`.

`v2` remains as the stable fallback. `v1` is removed from the CLI path. `v3` and `v4` are not implemented in the first pass, but the version registry must be structured so they can be added later without rewriting command handlers.

## Supported Languages

Training and inference text frontends support only:

- `zh`
- `en`
- `ja`

Mixed-language handling can remain only when it segments among `zh`, `en`, and `ja`. Korean (`ko`) and Cantonese (`yue`) are removed from the supported CLI path.

## ASR Strategy

The CLI keeps only Faster-Whisper for ASR.

Supported ASR languages:

- `zh`
- `en`
- `ja`

The CLI can skip ASR when the user supplies a ready annotation list. FunASR, ModelScope ASR, and Cantonese ASR are removed from the supported CLI path.

## Audio Separation Strategy

The CLI provides an optional wrapper around `audio-separator` from `python-audio-separator`.

The repository does not install this dependency by default. `gsv separate` detects whether `audio-separator` exists in `PATH`. If it is missing, the command fails with a concise install/configuration hint.

The wrapper is intentionally thin:

- Input: one audio file or a directory.
- Output: a directory containing separated vocals.
- Core options: model, stem, output format, and output directory.
- Implementation: call the external command with `subprocess`, stream logs, and return non-zero on failure.

UVR5 is removed from the primary workflow after `gsv separate` is available.

## CLI Commands

Expose both command names:

- `gsv`
- `gpt-sovits`

Both names call the same command implementation.

### `gsv init`

Create a project directory and `gsv.yaml`.

Example:

```bash
gsv init myvoice --version v2ProPlus
```

Responsibilities:

- Create project folders.
- Write a minimal config file.
- Validate requested version and language defaults.
- Avoid downloading or training anything.

### `gsv separate`

Optionally call external `audio-separator`.

Example:

```bash
gsv separate input.wav --output-dir data/separated --stem vocals
```

Responsibilities:

- Detect `audio-separator`.
- Run separation for file or directory input.
- Put vocal outputs in a predictable directory.
- Never require this step for users who already have clean vocal audio.

### `gsv prep slice`

Slice long vocal audio into training clips.

Example:

```bash
gsv prep slice -c gsv.yaml --input data/vocals --output data/sliced
```

Responsibilities:

- Use the existing slicing implementation where possible.
- Provide non-GUI defaults.
- Produce a clip directory suitable for ASR or manual annotation.

### `gsv prep asr`

Run Faster-Whisper over sliced clips and produce an annotation list.

Example:

```bash
gsv prep asr -c gsv.yaml --input data/sliced --language zh
```

Responsibilities:

- Transcribe clips with Faster-Whisper.
- Emit `path|speaker|language|text`.
- Support `zh`, `en`, and `ja`.

### `gsv prep features`

Generate GPT-SoVITS intermediate training artifacts from an annotation list.

Example:

```bash
gsv prep features -c gsv.yaml
```

Responsibilities:

- Run text extraction.
- Run SSL/Hubert feature extraction.
- Run semantic extraction.
- Write artifacts under the configured experiment directory.

### `gsv prep all`

Run slice, optional ASR, and feature extraction in sequence.

Example:

```bash
gsv prep all -c gsv.yaml
```

Responsibilities:

- Run only configured steps.
- Skip ASR when an annotation list already exists and `--force-asr` is not set.
- Fail early with actionable path errors.

### `gsv train sovits`

Train the SoVITS stage.

Example:

```bash
gsv train sovits -c gsv.yaml --batch-size 2
```

Responsibilities:

- Generate the temporary SoVITS config from `gsv.yaml` and version registry values.
- Invoke the existing SoVITS training script.
- Save final weights to the version-specific weight directory.

### `gsv train gpt`

Train the GPT semantic stage.

Example:

```bash
gsv train gpt -c gsv.yaml --batch-size 1
```

Responsibilities:

- Generate the temporary GPT config from `gsv.yaml` and version registry values.
- Invoke the existing GPT training script.
- Save final weights to the version-specific weight directory.

### `gsv train all`

Run SoVITS training then GPT training.

Example:

```bash
gsv train all -c gsv.yaml
```

Responsibilities:

- Validate feature artifacts before training.
- Run both stages in deterministic order.
- Print final weight paths.

### `gsv infer`

Generate audio from trained weights and reference audio.

Example:

```bash
gsv infer -c gsv.yaml --text "你好，这是测试。" --ref ref.wav --out out.wav
```

Responsibilities:

- Load configured or explicit GPT and SoVITS weights.
- Support direct text, text file input, and output file path.
- Support `zh`, `en`, and `ja`.
- Avoid importing `inference_webui.py`.

## Configuration

`gsv.yaml` is the primary interface. CLI flags override config values for a single command.

Example:

```yaml
project:
  name: myvoice
  root: .

version: v2ProPlus
language: zh
speaker: myvoice

paths:
  raw_audio: data/raw
  separated_audio: data/separated
  sliced_audio: data/sliced
  annotation: data/train.list
  exp_root: logs
  pretrained_root: GPT_SoVITS/pretrained_models

asr:
  engine: faster-whisper
  model: large-v3
  precision: float16

separation:
  enabled: false
  command: audio-separator
  stem: vocals
  output_format: WAV

train:
  gpu: "0"
  sovits_batch_size: 2
  gpt_batch_size: 1
  sovits_epochs: 8
  gpt_epochs: 15
  save_every_epoch: 4
  save_latest: true
  save_every_weights: true
  grad_ckpt: true

infer:
  gpt_weight: ""
  sovits_weight: ""
  ref_audio: ""
  ref_text: ""
  ref_language: zh
  text_language: zh
```

## Version Registry

Introduce a single registry module that maps supported versions to:

- SoVITS config file.
- GPT config file.
- pretrained GPT path.
- pretrained SoVITS generator/discriminator paths.
- output weight directories.
- feature extraction behavior.
- inference model loading behavior.

Handlers must use the registry instead of open-coded version conditionals.

## File Organization

The implementation should create a small CLI package rather than continuing to grow top-level scripts.

Proposed structure:

```text
gsv_cli/
  __init__.py
  __main__.py
  app.py
  config.py
  versions.py
  paths.py
  separate.py
  prep.py
  train.py
  infer.py
  subprocesses.py
```

Existing lower-level GPT-SoVITS modules can remain in place during the first pass. GUI-specific modules should not be imported by the new CLI path.

## Deletion Plan After CLI Verification

After the CLI training loop is verified, remove or archive:

- `webui.py`
- `GPT_SoVITS/inference_webui.py`
- `GPT_SoVITS/inference_webui_fast.py`
- `GPT_SoVITS/inference_gui.py`
- `tools/uvr5/`
- `tools/subfix_webui.py`
- Windows launchers and installer scripts.
- Colab notebooks.
- non-English/Chinese/Japanese docs not needed for this fork.
- UI i18n locale files and runtime translation wrappers.
- Korean and Cantonese text frontends from the supported path.

Deletion should happen in separate commits after replacement commands pass verification.

## Git and Branch Strategy

The local repository should use:

- `upstream`: `https://github.com/RVC-Boss/GPT-SoVITS`
- `origin`: `https://github.com/Chesszyh/GPT-SoVITS`

Implementation should happen on:

```text
feature/cli-first-training-loop
```

Installation fixes made before this design should be committed separately from CLI work.

## Testing Strategy

The first implementation must include command-level tests for:

- Config loading and CLI override behavior.
- Version registry validation.
- Language validation for `zh/en/ja`.
- Rejection of unsupported versions and languages.
- `audio-separator` missing-command handling.
- Training command generation without launching long training jobs.
- Inference command argument validation.

Manual verification must include:

- `gsv init`.
- `gsv prep features` on a tiny fixture or dry-run path.
- `gsv train sovits --dry-run`.
- `gsv train gpt --dry-run`.
- `gsv infer --dry-run`.
- Real inference if local weights are available.

## Open Decisions Resolved

- CLI details are implementation-owned unless they affect architecture or destructive deletion.
- First CLI version supports the training loop, not every historical GUI tool.
- `audio-separator` is an optional external command wrapper.
- `gsv.yaml` is the primary configuration interface.
- `gsv` and `gpt-sovits` are both exposed.
