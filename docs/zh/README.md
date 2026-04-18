# GPT-SoVITS CLI

这个 fork 只保留本地 CLI 训练和推理路径。

## 支持范围

- 训练：Linux + NVIDIA CUDA。
- 推理：Linux 或 macOS。
- 模型版本：`v2`、`v2Pro`、`v2ProPlus`，默认 `v2ProPlus`。
- 语言前端：`zh`、`en`、`ja`。
- 人声分离：仓库外部自行配置 `audio-separator`，CLI 只负责调用已有命令。

## 安装

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
bash install.sh --device CU128 --source HF-Mirror
```

macOS 推理环境可使用：

```bash
bash install.sh --device MPS --source HF-Mirror
```

## 训练闭环

```bash
gsv init myvoice --version v2ProPlus
gsv separate input.wav --output-dir data/separated --stem vocals
gsv prep slice -c myvoice/gsv.yaml --input data/separated --output data/sliced
gsv prep asr -c myvoice/gsv.yaml --input data/sliced --language zh
gsv prep features -c myvoice/gsv.yaml
gsv train sovits -c myvoice/gsv.yaml --batch-size 2
gsv train gpt -c myvoice/gsv.yaml --batch-size 1
```

## 推理

```bash
gsv infer \
  -c myvoice/gsv.yaml \
  --text "你好，这是测试。" \
  --ref ref.wav \
  --ref-text "参考文本。" \
  --out out.wav \
  --device cpu
```

先用 `--dry-run` 检查命令和配置，再启动真实训练或推理。
