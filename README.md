# GPT-SoVITS CLI

This fork is being reduced to a CLI-first workflow for local TTS fine-tuning and inference.

## Supported Workflow

- Train on Linux with NVIDIA CUDA.
- Run inference on Linux or macOS.
- Use `gsv` or `gpt-sovits`; both commands are equivalent after editable install.
- Keep external vocal separation outside this repository. The CLI can call an existing `audio-separator` executable.

## Supported Versions

- `v2`
- `v2Pro`
- `v2ProPlus`

`v2ProPlus` is the default. `v1` is intentionally unsupported in this fork. The version registry is centralized in `gsv_cli/versions.py` so later `v3` or `v4` support can be added deliberately.

## Supported Languages

- `zh`
- `en`
- `ja`

The CLI config and language validation reject unsupported training and inference frontend languages.

## Install

Create and activate the environment:

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
```

Install for Fedora or another CUDA Linux host:

```bash
bash install.sh --device CU128 --source HF-Mirror
```

Install for macOS inference:

```bash
bash install.sh --device MPS --source HF-Mirror
```

If you install manually instead of running `install.sh`, run:

```bash
pip install -r requirements-extra.txt --no-deps
pip install -r requirements.txt
pip install -e .
```

## CLI Quick Start

Create a project config:

```bash
gsv init myvoice --version v2ProPlus
```

Optionally separate vocals with an already configured `audio-separator` command:

```bash
gsv separate input.wav --output-dir data/separated --stem vocals
```

Prepare dataset artifacts:

```bash
gsv prep slice -c myvoice/gsv.yaml --input data/separated --output data/sliced
gsv prep asr -c myvoice/gsv.yaml --input data/sliced --language zh
gsv prep features -c myvoice/gsv.yaml
```

Train on the CUDA host:

```bash
gsv train sovits -c myvoice/gsv.yaml --batch-size 2
gsv train gpt -c myvoice/gsv.yaml --batch-size 1
```

Run inference:

```bash
gsv infer \
  -c myvoice/gsv.yaml \
  --text "你好，这是测试。" \
  --ref ref.wav \
  --ref-text "参考文本。" \
  --out out.wav \
  --device cpu
```

Use dry-run mode before launching expensive steps:

```bash
gsv prep features -c myvoice/gsv.yaml --dry-run
gsv train sovits -c myvoice/gsv.yaml --dry-run
gsv train gpt -c myvoice/gsv.yaml --dry-run
gsv infer -c myvoice/gsv.yaml --text "hello" --ref ref.wav --ref-text "hello" --out out.wav --dry-run
```

## Config

`gsv init` writes `gsv.yaml`. The important defaults are:

```yaml
version: v2ProPlus
language: zh
paths:
  raw_audio: data/raw
  separated_audio: data/separated
  sliced_audio: data/sliced
  annotation: data/train.list
  exp_root: logs
asr:
  engine: faster-whisper
  model: large-v3
  precision: float16
train:
  gpu: "0"
  sovits_batch_size: 2
  gpt_batch_size: 1
```

For a Fedora RTX 4060 8G training host, keep small batches first:

```yaml
train:
  gpu: "0"
  sovits_batch_size: 2
  gpt_batch_size: 1
  grad_ckpt: true
```

For macOS inference, copy trained weights and set:

```yaml
infer:
  gpt_weight: GPT_weights_v2ProPlus/myvoice-e15.ckpt
  sovits_weight: SoVITS_weights_v2ProPlus/myvoice-e8.pth
  ref_audio: ref.wav
  ref_text: reference text
  ref_language: zh
  text_language: zh
```

Then run `gsv infer --device cpu` or test `--device mps` if your local PyTorch build supports it reliably.

## Development Checks

Run the local test suite:

```bash
python -m unittest discover -s tests -v
```

Check CLI entrypoints:

```bash
python -m gsv_cli --help
python -m gsv_cli init /tmp/gsv-smoke --version v2ProPlus
python -m gsv_cli train sovits -c /tmp/gsv-smoke/gsv.yaml --dry-run
python -m gsv_cli infer -c /tmp/gsv-smoke/gsv.yaml --text "hello" --ref ref.wav --ref-text "hello" --out /tmp/out.wav --dry-run
```
