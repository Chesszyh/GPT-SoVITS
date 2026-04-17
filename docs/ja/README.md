# GPT-SoVITS CLI

この fork はローカル CLI の学習と推論だけを対象にしています。

## サポート範囲

- 学習: Linux + NVIDIA CUDA。
- 推論: Linux または macOS。
- モデル版: `v2`、`v2Pro`、`v2ProPlus`。既定は `v2ProPlus`。
- 言語フロントエンド: `zh`、`en`、`ja`。
- ボーカル分離: 外部で設定済みの `audio-separator` コマンドを CLI から呼び出します。

## インストール

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
bash install.sh --device CU128 --source HF-Mirror
```

macOS 推論環境では次を使えます。

```bash
bash install.sh --device MPS --source HF-Mirror
```

## 学習ループ

```bash
gsv init myvoice --version v2ProPlus
gsv separate input.wav --output-dir data/separated --stem vocals
gsv prep slice -c myvoice/gsv.yaml --input data/separated --output data/sliced
gsv prep asr -c myvoice/gsv.yaml --input data/sliced --language ja
gsv prep features -c myvoice/gsv.yaml
gsv train sovits -c myvoice/gsv.yaml --batch-size 2
gsv train gpt -c myvoice/gsv.yaml --batch-size 1
```

## 推論

```bash
gsv infer \
  -c myvoice/gsv.yaml \
  --text "これはテストです。" \
  --ref ref.wav \
  --ref-text "参照テキスト。" \
  --out out.wav \
  --device cpu
```

実行前に `--dry-run` でコマンドと設定を確認してください。
