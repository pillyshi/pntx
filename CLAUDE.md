# CLAUDE.md — pntx 開発指示

## プロジェクト概要

`pntx` は、ユーザが与える (positive, negative) テキストペアを学習素材として、

1. **生成**: 指定した側(正例 or 負例)のテキストを新たに生成する
2. **分類**: 任意のテキストを positive / negative に分類する

の2機能を提供する Python ライブラリ。

重要な前提:
- **正例/負例の意味論はユーザが定義する。** 感情ポジネガに限らず任意の対比軸(フォーマル/カジュアル、規約準拠/違反 など)を扱う。ライブラリはペアの意味を解釈せず、与えられたペアをそのまま few-shot 素材・スコアリング素材として使う。
- **llama.cpp インプロセス実行が主戦場。** LLM API(Anthropic 等)は副次的バックエンド。設計判断で迷ったら llama.cpp での性能・体験を優先する。
- 生成の用途は「データ拡張」と「成果物としての生成」の両方。前者は多様性、後者は品質を重視するが、メソッドは分けずパラメータで制御する。

## 公開 API(この形を維持すること)

```python
from pntx import PNTX

model = PNTX(backend=...)   # Backend インスタンス、または "llama" / "anthropic" 等の文字列

pairs = [
    ("この映画は最高だった", "この映画は退屈だった"),
    ("サポートが丁寧で助かった", "サポートの対応が雑だった"),
]
model.fit(pairs)

# 生成
texts: list[str] = model.generate(
    n=20,
    side="positive",        # "positive" | "negative"
    temperature=1.0,
    dedup=True,             # 生成物同士および seed ペアとの近似重複を除去
    verify=True,            # 自己分類で side に一致しないものを棄却
    min_confidence=0.8,     # verify 時の棄却閾値
)

# 分類
result = model.classify("店員さんの笑顔が素敵だった")
result.label        # "positive" | "negative"
result.confidence   # float (0.0–1.0)
result == "positive"  # True になるよう __eq__ を実装

results = model.classify_batch(texts)  # 必須。逐次呼び出しの単純ループにしないこと(後述)
```

## アーキテクチャ

### バックエンド抽象(`pntx/backends/`)

```python
class Backend(Protocol):
    def complete(self, prompt: str, *, temperature: float = ..., max_tokens: int = ..., stop: list[str] | None = ...) -> str: ...

class ScoringBackend(Backend, Protocol):
    def score_choices(self, prompt: str, choices: list[str]) -> list[float]:
        """prompt に続く各 choice の対数尤度を返す。分類の主経路。"""
```

- **LlamaCppBackend**(`llama-cpp-python` 使用): `ScoringBackend` を実装。`score_choices` は各 choice のトークン logprob 合計で実装する。共通 prefix の KV キャッシュ再利用を必ず行うこと(プロンプト設計側も、可変部分が末尾に来るよう共通 prefix を最大化する)。
- **AnthropicBackend**: まず `Backend` のみ実装(テキスト生成をパースして分類)。構造化出力や logprobs が使える場合の `ScoringBackend` 化は後回しでよい。
- 分類ロジックは二段構え:
  - バックエンドが `ScoringBackend` → few-shot プロンプト末尾でラベルトークン(例: `positive` / `negative`)の logprob を比較。confidence は softmax で算出。
  - そうでない → 生成テキストのパース。confidence はパース結果の確信度が取れなければ 1.0/0.5 等の規約値でよいが、docstring に明記する。

### fit と exemplar 選択(`pntx/selection.py`)

- `fit(pairs)` はペアの保持と前処理のみ。学習は行わない。
- ペア数がプロンプトに収まらない場合に備え、**プロンプトへ入れる代表ペアの選択戦略をプラガブルにする**:
  - `RandomSelector`(デフォルト・最初に実装)
  - `DiversitySelector`(多様性最大化。埋め込みが必要なら optional 依存)
  - `NearestSelector`(分類対象テキストに近いペアを動的選択。分類精度に効くので優先度高)
- Selector は `select(pairs, k, query: str | None) -> list[pair]` のインターフェースで統一。

### 生成ループ(`pntx/generate.py`)

- `verify=True` のとき内部で classify を呼び、`side` と不一致 or `min_confidence` 未満を棄却。棄却で n に満たない場合は追加生成でリトライ(上限回数を設け、満たせなければ警告付きで返す)。
- `dedup=True` のとき近似重複除去。初期実装は n-gram ベース(依存なし)でよい。埋め込みベースは optional。seed ペアとの重複除去も含む(データ拡張でほぼコピーを返さないため)。

### classify_batch

- LlamaCppBackend: 共通 prefix の KV キャッシュを温めてから各入力を評価する。
- AnthropicBackend: 並行リクエスト(`asyncio` + セマフォで同時数制限)。
- 逐次 for ループ実装は不可。バックエンドごとに最適化する。

## パッケージング

- 本体はゼロ依存(標準ライブラリのみ)を目指す。
- optional dependencies:
  - `pntx[llama]` → `llama-cpp-python`
  - `pntx[anthropic]` → `anthropic`
  - `pntx[embeddings]` → 埋め込みベースの dedup / DiversitySelector 用
- 未インストールのバックエンドを使おうとしたら、インストールコマンドを含む明確な ImportError を出す。
- `pyproject.toml`(hatchling or setuptools)、Python 3.10+。

## 実装順序

1. コア型(`ClassifyResult`、`Backend` / `ScoringBackend` Protocol)、`PNTX` 本体の骨格、`RandomSelector`
2. `LlamaCppBackend`: complete / score_choices / KV キャッシュ再利用
3. logprob 分類経路 + classify_batch(llama.cpp)
4. 生成ループ(verify / dedup / リトライ)
5. `AnthropicBackend` + パースベース分類 + 並行 batch
6. `NearestSelector` / `DiversitySelector`、埋め込み optional

## テスト

- バックエンドは `FakeBackend`(決め打ち応答を返す `ScoringBackend` 実装)でモックし、分類ロジック・生成ループ・selector をユニットテストする。実モデル・実 API を叩くテストは `tests/integration/` に分離し、デフォルトでは skip。
- 生成ループのテストは「verify で棄却→リトライ→上限到達で警告」の分岐を必ずカバーする。
- dedup は日本語・英語両方のケースを入れる(n-gram の粒度に注意。日本語は文字 n-gram を使う)。

## コーディング規約

- 型ヒント必須、`from __future__ import annotations` を使用。
- ドキュメントとコメントは英語、README は英語 + 日本語(README.ja.md)。
- ruff + mypy(strict)を CI に入れる。
- プロンプトテンプレートはコード内にハードコードせず `pntx/prompts.py` に集約し、ユーザが差し替え可能にする。

## やらないこと(スコープ外)

- 3クラス以上の分類(将来検討。ただし内部設計で ["positive", "negative"] をハードコードした定数散在にはしない)
- ファインチューニング・埋め込み分類器の学習
- CLI(ライブラリ API のみ。CLI は将来別途)

