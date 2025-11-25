# Build Large Language Model

書籍、LLM 自作入門をハンズオンで実装し、適宜コメント等で自分の学習をサポートすることが目的のレポジトリです。

## セットアップ

このプロジェクトは **uv** でパッケージ管理されています。

### 初回セットアップ

1. **依存関係のインストール**

   ```bash
   uv sync
   ```

2. **開発モードでプロジェクトをインストール**

   ```bash
   uv pip install -e .
   ```

   これにより、プロジェクト内のどのファイルからも `sys.path.append` を使わずに、以下のようにインポートできるようになります：

   ```python
   from attention.multi_head_attention import MultiHeadAttention
   from gpt_model.gpt_model import GPTModel
   from gpt_model.config import GPT_CONFIG_124M
   ```

## 実行方法

### モジュールとして実行

uv 環境で Python モジュールを実行する場合：

```bash
# Transformerブロックのテスト
uv run -m gpt_model.transformer

# GPTモデル全体のテスト
uv run -m gpt_model.gpt_model

# テキスト生成
uv run -m gpt_model.generate_text

# 事前学習ユーティリティ
uv run -m pre_training.util
```

### 他のファイルからインポート

開発インストール（`uv pip install -e .`）が完了していれば、プロジェクト内のどこからでも以下のようにインポートできます：

```python
from attention.multi_head_attention import MultiHeadAttention
from gpt_model.transformer import TransformerBlock
from gpt_model.config import GPT_CONFIG_124M
from pre_training.util import calc_loss_batch
```

**重要**: `sys.path.append(...)` は不要です。すべて絶対インポートで動作します。

## トラブルシューティング

### ImportError が発生する場合

開発インストールを再実行してください：

```bash
uv pip install -e .
```

### パッケージの再ビルドが必要な場合

```bash
uv pip uninstall build-large-language-model
uv pip install -e .
```

## 開発のヒント

- **新しいパッケージを追加した場合**: `pyproject.toml` の `[tool.setuptools] packages` リストに追加し、`uv pip install -e .` を再実行
- **相対インポートは使用しない**: 常にパッケージ名から始まる絶対インポートを使用（例: `from gpt_model.config import ...`）
- **モジュール実行**: スクリプトを直接実行せず、`uv run -m <package>.<module>` 形式で実行することを推奨
