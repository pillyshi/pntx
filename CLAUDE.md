# CLAUDE.md — pntx 開発指示

## プロジェクト概要

`pntx` は、ユーザが与える positive / negative の2つのテキストプールを学習素材として、

1. **`t2pn`**(text → positive/negative): 任意のテキストを positive / negative に分類する。**scikit-learn の Classifier** として実装する。
2. **`pn2t`**(positive/negative → text): 指定した側(正例)のテキストを新たに生成する。**imbalanced-learn 流の OverSampler** として実装する。目的の異なる2つの Sampler を持つ: `OverSampler`(分類器学習データ拡張用の hard positive 生成)と `SyntheticSampler`(データ公開用途の匿名化された代表的合成データ生成)。

の2コンポーネントを提供する Python ライブラリ。両者は同じ `pntx/backends/` を共有するが、公開 API 上は独立したクラスであり、両者を束ねるファサードクラスは持たない。

重要な前提:
- **正例/負例の意味論はユーザが定義する。** 感情ポジネガに限らず任意の対比軸(フォーマル/カジュアル、規約準拠/違反 など)を扱う。ライブラリはプールの意味を解釈せず、与えられたテキストをそのまま few-shot 素材・スコアリング素材として使う。
- **入力は scikit-learn / imbalanced-learn の `(X, y)` 規約に合わせる。** `X: list[str]`(生テキスト)、`y`(0/1 または `"positive"`/`"negative"` などの2値ラベル)。ペアリングを強制しない点は従来通り(`y` でグルーピングした結果、positive/negative それぞれの件数が揃う必要はない)。**ただし `t2pn.Classifier.fit` / `pn2t.OverSampler.fit_resample` はどちらも両クラスが最低1件ずつ存在することを要求する**(旧仕様にあった「片側のプールだけで fit」というスモークテスト用途は、sklearn/imblearn の分類データセット規約を採用したことに伴い廃止)。
- **llama.cpp インプロセス実行が主戦場。** 現在ビルトインで提供するバックエンドは `LlamaCppBackend` のみ(`AnthropicBackend` は一旦廃止 — 経緯は後述)。設計判断で迷ったら llama.cpp での性能・体験を優先する。`t2pn`・`pn2t` とも、LLM 呼び出しは共通の `Backend` 抽象を経由し、同じロード済みモデル(例: 同一の `LlamaCppBackend` インスタンス)を共有できるようにする。**それぞれが独自の LLM クライアントを持って別々にモデルをロードする実装は禁止**(ローカル推論のメモリ/VRAM を二重に食うため)。リモートAPIバックエンドを将来また追加する場合も、`Backend` プロトコルを実装するだけで足りる設計(`_backend_resolve.py` のレジストリに1行足すだけ)は維持すること。
- **`pn2t` には目的の異なる2つの Sampler がある。** `OverSampler`(分類器学習データ拡張、hard positive)と `SyntheticSampler`(匿名化された代表的合成データ、データ公開用途)。どちらも positive 側のみ生成、二値ラベル `{0, 1}` のみサポートという制約は共通。negative 側生成、3値以上への一般化は引き続きスコープ外(後述)。

## 公開 API(この形を維持すること)

```python
from pntx.t2pn import Classifier
from pntx.pn2t import OverSampler

# --- t2pn: 分類 (scikit-learn Classifier) ---
clf = Classifier(backend=...)   # Backend インスタンス、または "llama" 等のビルトインバックエンド名の文字列

X = ["この映画は最高だった", "サポートが丁寧で助かった", "この映画は退屈だった", "サポートの対応が雑だった"]
y = ["positive", "positive", "negative", "negative"]   # 0/1 でも可

clf.fit(X, y)                 # 学習ではなくプール保持+前処理(旧 PNTX.fit と同じ)
clf.predict(X)                # -> array-like of "positive"/"negative"(sklearn 標準の predict 契約)
clf.predict_proba(X)          # -> shape (n_samples, 2) の確率行列
clf.score(X, y)               # ClassifierMixin から無償で手に入る

# sklearn エコシステムにそのまま乗る
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X, y, cv=5)

# --- pn2t: 生成 (imbalanced-learn 流 OverSampler) ---
sampler = OverSampler(backend=..., n_synthesized=None)  # None は「バランスするまで」

X_aug, y_aug = sampler.fit_resample(X, y)   # 生成された positive テキストが末尾に追加される
sampler.generation_result_                  # boundary feature 分析 + 生成根拠(pydantic モデル)

# imbalanced-learn の Pipeline にもそのまま乗る(imbalanced-learn 自体は必須依存にしない。
# fit_resample を duck-typing で提供するだけで imblearn.pipeline.Pipeline は使える)

# --- pn2t: 匿名化された合成データ生成 (SyntheticSampler) ---
from pntx.pn2t import SyntheticSampler

sampler = SyntheticSampler(backend=..., n_synthesized=10)  # デフォルトなし、必須指定

X_syn, y_syn = sampler.fit_resample(X, y)     # negative は検証にのみ使い、プロンプトには含めない
sampler.generation_result_.synthetic_texts    # 生成テキスト + 監査用の generalized_from(何を一般化したか)
```

`ClassifyResult`(`.label`/`.confidence`/`__eq__`)による1件ずつの結果表現は廃止し、`predict`/`predict_proba` は sklearn 標準の配列ベース契約に統一する。

## アーキテクチャ

### バックエンド抽象(`pntx/backends/`)— 変更なし、`t2pn`/`pn2t` 共有

```python
class Backend(Protocol):
    def complete(self, prompt: str, *, temperature: float = ..., max_tokens: int = ..., stop: list[str] | None = ...) -> str: ...

class ScoringBackend(Backend, Protocol):
    def score_choices(self, prompt: str, choices: list[str]) -> list[float]:
        """prompt に続く各 choice の対数尤度を返す。分類の主経路。"""
```

- **LlamaCppBackend**(`llama-cpp-python` 使用): `ScoringBackend` を実装。`score_choices` は各 choice のトークン logprob 合計で実装する。共通 prefix の KV キャッシュ再利用を必ず行うこと。
- **AnthropicBackend は一旦廃止。** 以前は `Backend` のみ実装(テキスト生成をパースして分類/構造化出力)する副次的バックエンドとして存在したが、現時点ではビルトインのバックエンドは `LlamaCppBackend` のみ。復活させる場合も `Backend` プロトコルだけ実装すればよい設計(下記)は変えないこと。
- **構造化出力(pn2t が必要とする JSON スキーマ付き生成)は、対応バックエンドでは llama.cpp のグラマー制約デコーディングを使う。** `Backend` の必須メソッドは変えず、任意実装の `StructuredBackend`(`pntx/backends/base.py`、`complete_json(prompt, *, schema, temperature, max_tokens) -> str`)を追加。`LlamaCppBackend` はこれを実装し、`llama_cpp.LlamaGrammar.from_json_schema()` で pydantic の `model_json_schema()` から生成した GBNF grammar を `create_completion` に渡すことで、構文的に妥当なJSONを保証する(pydanticレベルの制約 — enum・数値レンジ等 — までは保証しないため、`model_validate_json()` によるバリデーションは引き続き必須)。`pntx/pn2t/_structured.py` の `complete_structured` は `isinstance(backend, StructuredBackend)` で分岐し、対応していれば `complete_json` を、対応していない(将来のリモートAPI系などの)バックエンドは従来通り「JSON出力を促すプロンプト → `complete()` → パース → 限られた回数までリトライ」にフォールバックする。この二段構えにより `Backend` 抽象そのものは変わらず、`t2pn` と `pn2t` は引き続き同じ Backend 実装・同じロード済みモデルを共有できる。

### `t2pn.Classifier`(`pntx/t2pn.py`)

- `sklearn.base.BaseEstimator` + `ClassifierMixin` を継承する。`X` は生テキストの list なので、数値配列を前提にした `check_X_y`/`check_array` を素通りさせるため estimator tags(`X_types: ["string"]` 相当、`no_validation` 系)を適切に設定すること。`sklearn.utils.estimator_checks.check_estimator` に literal に通す必要はないが、`Pipeline`/`cross_val_score` で壊れないことは確認する。
- `fit(X, y)` は `y` でグルーピングして positive/negative プールを作るだけで、学習は行わない(旧 `PNTX.fit` のロジックを流用)。ラベルは 0/1 でも `"positive"`/`"negative"` 文字列でも受け付ける。
- 分類ロジック自体(exemplar 選択 → プロンプト構築 → スコアリング or パース)は既存の `pntx/selection.py` / `pntx/prompts.py` / `core.py` の分類パスをそのまま移設する。二段構えの分岐(`ScoringBackend` なら logprob 比較、そうでなければ生成テキストのパース)も維持。
- `predict_batch` 相当は `predict`/`predict_proba` がバッチを受け取れることで代替する(旧 `classify_batch` の「逐次 for ループ禁止、バックエンドごとに最適化」という制約はそのまま `predict`/`predict_proba` の内部実装に引き継ぐ)。

### exemplar 選択(`pntx/selection.py`)— 変更なし、`t2pn`/`pn2t` 共有

- `RandomSelector` / `DiversitySelector` / `NearestSelector` は従来通り。`Selector.select(pool, k, query)` インターフェースも維持。
- `pn2t.OverSampler` の exemplar サンプリングは「件数 k」ではなく「トークン予算」ベース(下記)なので、`Selector` をそのまま使うのではなく、予算ベースのサンプリングヘルパーを別途 `pntx/selection.py` に追加する(`sample_method` として `RandomSelector` 等と同じ戦略名を共有できる設計が望ましい)。

### `pn2t.OverSampler`(`pntx/pn2t/`)— `semaxis.HardPositiveOverSampler` の完全移植 + Backend 統合

- `sklearn.base.BaseEstimator` を継承する(imbalanced-learn の `BaseOverSampler` は継承しない。`fit_resample` を duck-typing で提供するだけで `imblearn.pipeline.Pipeline` から利用可能なため、`imbalanced-learn` 自体は必須依存に加えない)。
- `fit_resample(X, y)`: 二値ラベルのみサポート(`{0, 1}`)。`y=1` を positive として扱い、生成されたテキストは `y=1` として末尾に追加される。**`pn2t` v1 は positive 側の生成のみ**(negative 側や3値以上への一般化は将来の拡張)。
- アルゴリズムは semaxis 実装をそのまま踏襲:
  1. positive/negative それぞれの exemplar をトークン予算内でサンプリング(`sample_method`: `random`/他、`context_limit` に基づく予算計算)。
  2. positive/negative の特徴・境界特徴(boundary features)を LLM に分析させ、"専門家なら positive と判定するが浅い分類器は negative と誤判定しうる" hard positive テキストを `batch_size` 件ずつバッチ生成。
  3. 完全一致ベースの dedup(`deduplicate=True` がデフォルト。元データ・既に採択した生成物との文字列一致のみを見る — 旧 `pntx/generate.py` にあった n-gram 近似重複除去とは別物で、v1 では使わない)。`SyntheticSampler` はこれとは別目的の漏洩検出レイヤーを追加で持つ(下記)。
  4. `target_count`(`n_synthesized` 明示指定 or `None` でクラスバランスまで自動計算)に達するまでバッチ生成を繰り返し、上限バッチ数に達したら警告付きで打ち切る。
- `generation_result_`(`positive_features`/`negative_features`/`boundary_features`/`hard_positives`、pydantic モデル)を fit 後に公開。`save(path)`/`load(path, backend=...)` で JSON へシリアライズ・復元できる。
- コンストラクタ引数(`batch_size`, `max_examples_per_class`, `deduplicate`, `context_limit`, `language`, `seed`, `sample_method`, `verbose`, `logger`)は semaxis 版を踏襲しつつ、`llm: BaseLLMClient | str` は `backend: Backend | str` に置き換える(pntx の `_resolve_backend` を再利用)。

### `pn2t.SyntheticSampler`(`pntx/pn2t/`)— 匿名化された代表的合成データ生成

- `OverSampler` とは独立したクラス(モード/パラメータではない)。目的関数が逆: `OverSampler` は境界を突く hard positive、`SyntheticSampler` は典型的・平均的な positive を、原文の具体的な情報(固有名詞・人名・日付・数値・場所など)を含まないよう生成する。ユースケースはプライバシー上公開できない元テキストプールの代わりに、分布を代表する合成データセットを公開すること。
- `resolve_backend`・`LLMEstimatorMixin`・`selection.sample_group`/`_SAMPLE_METHODS`・`pn2t._structured.complete_structured`(変更なしでそのまま再利用可能)など、`OverSampler` と同じ共有インフラの上に構築する。
- `fit_resample(X, y)` は `OverSampler` と同じ契約(二値ラベル、両クラス最低1件)を維持するが、**negative 側はラベル検証にのみ使い、生成プロンプトには含めない**(境界フレーミングを避けるため、かつ「positive に本質的 vs この1例に固有」の判断は複数の positive exemplar の共通性から行えるため)。
- `n_synthesized` に `OverSampler` のような「`None` でクラスバランスまで自動計算」という挙動はない(自然な目標がないため)。**デフォルトなしの必須パラメータ**にする。
- exemplar サンプリングは positive 側のみ(`OverSampler` の pos/neg 予算折半・バランス調整ロジックは不要)。token budget は `context_limit - overhead - max_tokens`(`OverSampler` と異なり `// 2` しない)。
- 匿名性はプロンプト指示だけでなく、`pntx.dedup.contains_verbatim_span(text, sources, min_len)` によるベストエフォートの漏洩検出でも担保する: 生成テキストが positive プールから `min_verbatim_span`(デフォルト20文字)以上の連続部分文字列をそのままコピーしていたら reject してリトライする。これは `OverSampler` の完全一致 dedup とも旧 n-gram 近似重複除去とも別物(近似重複検出ではなく漏洩検出が目的、パラフレーズされた漏洩までは検出できないヒューリスティック)。
- `generation_result_`(`style_features`/`content_features`/`synthetic_texts`、pydantic モデル)を fit 後に公開。各 `synthetic_texts[].generalized_from` は「何を一般化したかの種類」の監査ログであり、元の具体的内容そのものを含めないようプロンプトで明示的に禁止する(この監査フィールド自体が漏洩経路にならないようにするため)。`save`/`load` は `OverSampler` と同じ JSON ラウンドトリップパターン。

## パッケージング

- **コア依存として `scikit-learn` と `pydantic` を必須にする**(`t2pn.Classifier` の Estimator 契約、`pn2t.OverSampler` の構造化出力検証にそれぞれ必須のため)。「本体はゼロ依存」という従来方針は撤回し、ゼロ依存の対象はバックエンド実装(LLM SDK)と埋め込み系に限定する。
- optional dependencies:
  - `pntx[llama]` → `llama-cpp-python`
  - `pntx[embeddings]` → 埋め込みベースの dedup / DiversitySelector 用
  - (`pntx[anthropic]` → `anthropic` は `AnthropicBackend` の廃止に伴い削除。復活時は `_BACKEND_REGISTRY` にエントリを1行足すのと合わせて追加する)
- 未インストールのバックエンドを使おうとしたら、インストールコマンドを含む明確な ImportError を出す。
- `pyproject.toml`(`uv_build`)、Python 3.10+。

## 実装順序

1. 既存の `pntx/backends/`・`pntx/selection.py`・`pntx/prompts.py` はそのまま流用。`pntx/core.py`(`PNTX` ファサード)と `pntx/generate.py`(旧 verify/dedup 生成ループ)は削除。
2. `pntx/t2pn.py`: 既存の分類ロジック(`core.py` の `classify`/`classify_batch` 相当)を sklearn `BaseEstimator`/`ClassifierMixin` に載せ替え。`fit(X, y)` のラベルグルーピング、text-input 用の estimator tags 設定。
3. `pntx/pn2t/`: `Backend.complete()` 上の構造化出力ヘルパー(JSON プロンプト + pydantic 検証 + リトライ)。boundary feature プロンプト(`pntx/pn2t/prompts.py`)。予算ベース exemplar サンプリング。`HardPositiveOverSampler` のロジック本体の移植(`fit_resample`/`save`/`load`)。
4. `benchmarks/t2pn/run.py` を新 `t2pn.Classifier` に合わせて更新。
5. `AnthropicBackend` 経由での `t2pn`/`pn2t` 動作確認(構造化出力ヘルパーがバックエンド非依存であることの検証)。
6. `NearestSelector`/`DiversitySelector` の `pn2t` 側サンプリングへの統合、埋め込み optional。

## テスト

- バックエンドは `FakeBackend`(決め打ち応答を返す `ScoringBackend` 実装)でモックする。実モデル・実 API を叩くテストは `tests/integration/` に分離し、デフォルトでは skip。
- `t2pn.Classifier`: `fit`/`predict`/`predict_proba` のユニットテストに加えて、`sklearn.pipeline.Pipeline`・`cross_val_score` に組み込んで壊れないことを確認するテストを持つ。
- `pn2t.OverSampler`: 「boundary feature 分析 → hard positive 生成 → dedup で棄却 → リトライ → 上限到達で警告」の分岐を必ずカバー。`n_synthesized=None` のクラスバランス自動計算、`save`/`load` の往復も対象。
- `pn2t.SyntheticSampler`: `OverSampler` と同様の分岐に加え、negative 側がプロンプトに含まれないことの直接検証、`contains_verbatim_span` による漏洩 dedup(reject → リトライ、`min_verbatim_span` 可変、`deduplicate=False` で無効化されること)を必ずカバー。
- dedup(完全一致・`contains_verbatim_span`)は日本語・英語両方のケースを入れる。
- 旧仕様にあった「片側のプールだけで fit → generate(verify=False)」のスモークテストは廃止(前提の通り、両クラス1件以上が必須になったため)。

## コーディング規約

- 型ヒント必須、`from __future__ import annotations` を使用。
- ドキュメントとコメントは英語、README は英語 + 日本語(README.ja.md)。
- ruff + mypy(strict)を CI に入れる。
- プロンプトテンプレートはコード内にハードコードせず、`t2pn`/`pn2t` それぞれの `prompts.py` に集約し、ユーザが差し替え可能にする。

## やらないこと(スコープ外)

- 3クラス以上の分類(将来検討。ただし内部設計で `["positive", "negative"]` をハードコードした定数散在にはしない)
- `pn2t` の negative 側生成(hard negative 相当)は引き続きスコープ外
- 「成果物としての生成」(品質重視・任意サイド生成)は `pn2t.SyntheticSampler` として実装済み。ただし意味的なパラフレーズ漏洩の自動検出(embedding ベースの類似度検証等)は引き続きスコープ外 — `contains_verbatim_span` による verbatim コピー検出のみのベストエフォート
- ファインチューニング・埋め込み分類器の学習
- CLI(ライブラリ API のみ。CLI は将来別途)
