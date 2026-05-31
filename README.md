# super-resolution-gan

PyTorch で実装した、解像度非依存の超解像 GAN 研究用コードベースです。低解像度画像（LR）を条件として高解像度画像（HR）を生成する `ProgressiveSRGenerator` と、任意解像度の画像を評価できる `ResolutionAgnosticDiscriminator` を使って学習します。

## 主な機能

- HR 画像から LR/HR ペアを動的に生成するデータセット
- 1x1 から目標解像度まで段階的にアップサンプリングする生成器
- 条件付き・解像度非依存の識別器
- R3GAN 形式の敵対的損失、再構成損失、マルチスケール損失、LR 一貫性損失
- EMA、チェックポイント保存、検証サンプル出力、JSONL/TensorBoard ロギング
- `uv` による環境管理

## リポジトリ構成

```text
.
├── configs/              # 学習設定 YAML
├── datasets/             # LR/HR ペア生成、劣化処理、画像ピラミッド
├── docs/                 # 設計メモ
├── scripts/train.py      # 学習 CLI
├── src/
│   ├── losses/           # 各種損失関数
│   ├── models/           # 生成器・識別器
│   ├── training/         # Trainer、EMA、checkpoint、logging、validation
│   └── utils/            # 設定読み込み、乱数固定など
├── tests/                # pytest テスト
├── pyproject.toml        # 依存関係・pytest 設定
└── uv.lock               # lockfile
```

## 必要要件

- Python 3.14 以上
- [uv](https://docs.astral.sh/uv/) 0.5 以上を推奨
- NVIDIA GPU を使う場合は CUDA 13.0 系 PyTorch wheel を利用できる環境
  - このリポジトリの `pyproject.toml` は `torch` / `torchvision` を `pytorch-cu130` index から取得する設定です。
- CPU でもテストや小規模な動作確認は可能ですが、GAN の本格学習には GPU を推奨します。

## 環境構築

### 1. リポジトリを取得

```bash
git clone <repository-url>
cd super-resolution-gan
```

### 2. uv をインストール

`uv` が未インストールの場合は、以下のいずれかでインストールします。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

または、すでに Python / pip がある場合:

```bash
pip install uv
```

### 3. 依存関係を同期

```bash
uv sync
```

`uv sync` により `.venv/` が作成され、`pyproject.toml` と `uv.lock` に基づいて依存関係がインストールされます。

> CPU 版 PyTorch や別の CUDA バージョンを使いたい場合は、`pyproject.toml` の `[tool.uv.sources]` と `[[tool.uv.index]]` を環境に合わせて変更してから `uv sync` を実行してください。

### 4. インストール確認

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

GPU 環境では 2 行目が `True` になることを確認してください。

## データセットの準備

デフォルト設定では以下のディレクトリを参照します。

```text
data/
├── train/    # 学習用 HR 画像
└── val/      # 検証用 HR 画像（任意）
```

対応拡張子は `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp` です。`ImagePairDataset` は HR 画像を読み込み、設定に基づいて LR 画像を生成します。デフォルトでは 256x256 の HR crop から bicubic downsample により 64x64 相当の LR 入力を生成します。

例:

```bash
mkdir -p data/train data/val
cp /path/to/train/images/* data/train/
cp /path/to/val/images/* data/val/
```

## 設定ファイル

主な設定ファイルは以下です。

- `configs/default.yaml`: 共通のデフォルト設定
- `configs/sr_64_to_256.yaml`: 64→256 相当の超解像タスク設定

重要な項目:

```yaml
data:
  train_dir: data/train
  val_dir: data/val
  image_size_hr: [256, 256]
  image_size_lr: [64, 64]
  batch_size: 16
  num_workers: 8

degradation:
  downsample:
    method: bicubic
    scale: 4

project:
  output_dir: runs/sr_64_to_256

logging:
  tensorboard: true
  wandb:
    enabled: false
    project: super-resolution-gan

training:
  sample_every_kimg: 50
  sample_max_images: 4
  sample_dir: samples
```

メモリ不足が起きる場合は、まず `data.batch_size`、`data.num_workers`、`model.generator.base_channels`、`model.discriminator.base_channels` を小さくしてください。

## 実行手順

### 学習を開始

```bash
uv run python scripts/train.py --config configs/sr_64_to_256.yaml
```

実行すると、設定の `project.output_dir` 配下にログ、検証サンプル、チェックポイントが保存されます。

```text
runs/sr_64_to_256/
├── checkpoints/
│   ├── latest.pt
│   └── step_XXXXXXXX.pt
├── logs/
│   └── metrics.jsonl
├── samples/
│   └── step_XXXXXXXX_kimg_YYY.YYY.png
└── validation/
    └── step_XXXXXXXX.png
```

### チェックポイントから再開

```bash
uv run python scripts/train.py \
  --config configs/sr_64_to_256.yaml \
  --resume runs/sr_64_to_256/checkpoints/latest.pt
```

### 小規模な動作確認例

手元の画像や CPU で素早く確認したい場合は、一時的に設定ファイルをコピーして小さな値に変更します。

```bash
cp configs/sr_64_to_256.yaml configs/local_smoke.yaml
```

`configs/local_smoke.yaml` の例:

```yaml
data:
  batch_size: 1
  num_workers: 0

training:
  epochs: 1
  log_every: 1
  validate_every: 10
  save_every: 10
  sample_every_kimg: 1

model:
  generator:
    base_channels: 32
    max_channels: 128
  discriminator:
    base_channels: 32
    max_channels: 128
```

実行:

```bash
uv run python scripts/train.py --config configs/local_smoke.yaml
```

## テスト・品質チェック

### テスト実行

```bash
uv run pytest tests
```

### Ruff による lint

```bash
uv run ruff check .
```

### Ruff による format

```bash
uv run ruff format .
```

## 出力とログ

- スカラー値は `runs/.../logs/metrics.jsonl` に JSONL 形式で保存されます。
- `torch.utils.tensorboard` が利用可能な環境では TensorBoard ログも同じログディレクトリに保存されます。
- `logging.wandb.enabled: true` にすると、スカラーとサンプル画像を Weights & Biases にも送信します。`project`、`entity`、`name`、`mode`、`tags` などは `logging.wandb` 配下で指定できます。
- `training.sample_every_kimg` が 0 より大きい場合、学習中に指定 kimg（1000 images）ごとに LR upsample / SR / HR を並べたサンプル画像が `runs/.../samples/` に保存され、TensorBoard / W&B にも記録されます。
- 検証時には LR upsample / SR / HR を横に並べたサンプル画像が `runs/.../validation/` に保存されます。
- チェックポイントには generator、discriminator、EMA、optimizer、scheduler、config、乱数状態が含まれます。

TensorBoard で確認する場合:

```bash
uv add tensorboard
uv run tensorboard --logdir runs
```

## トラブルシューティング

### `uv sync` で PyTorch のインストールに失敗する

CUDA バージョンや OS に合う PyTorch wheel が必要です。CPU 版や別 CUDA 版を使う場合は、`pyproject.toml` の PyTorch index 設定を変更してください。

### `ValueError: no images found in 'data/train'` が出る

`data.train_dir` に画像ファイルが存在しません。`configs/*.yaml` の `data.train_dir` を確認し、対応拡張子の画像を配置してください。

### GPU メモリ不足になる

以下を順に小さくしてください。

1. `data.batch_size`
2. `model.generator.base_channels`
3. `model.discriminator.base_channels`
4. `data.image_size_hr`

### 検証が実行されない

`data.val_dir` が存在しない場合、検証 dataloader は作成されません。検証を使う場合は `data/val` を作成して画像を配置してください。

## 開発メモ

設計の詳細は `docs/design.md` を参照してください。
