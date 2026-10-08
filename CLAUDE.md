# CLAUDE.md — pntx 開発指示

## プロジェクト概要

`pntx` は、ユーザが与える positive / negative の2つのテキストプールを学習素材として、

1. **`t2pn`**(text → positive/negative): 任意のテキストを positive / negative に分類する。目的の異なる複数の **scikit-learn Classifier**(`BaseEstimator` + `ClassifierMixin`、共通の `(X, y)` 契約)を持つ: LLM プロンプティングでその場で分類する `LLMPromptingClassifier`(学習なし、旧 `Classifier`)と、事前学習済みエンコーダを実際に fine-tuning する `FineTuningClassifier`(既定は multilingual BERT だが、`transformers` の `AutoModelForSequenceClassification` 経由なので特定のアーキテクチャに縛られない)。
2. **`pn2t`**(positive/negative → text): 指定した側(正例)のテキストを新たに生成する。**imbalanced-learn 流の over-sampler ファミリー**として実装する(アルゴリズムごとに1クラス、`<Name>OverSampler` 命名、共通基底 `BaseLLMOverSampler` — 後述)。現在は目的の異なる3クラスを持つ: `HardPositiveOverSampler`(分類器学習データ拡張用の hard positive 生成)、`CounterfactualOverSampler`(既存の負例を最小編集して正例にする反実仮想データ拡張)、`TypicalPositiveOverSampler`(データ公開用途の、具体情報を一般化した典型的な正例の生成)。

の2コンポーネントを提供する Python ライブラリ。両者は同じ `pntx/backends/` を共有するが、公開 API 上は独立したクラスであり、両者を束ねるファサードクラスは持たない。

重要な前提:
- **正例/負例の意味論はユーザが定義する。** 感情ポジネガに限らず任意の対比軸(フォーマル/カジュアル、規約準拠/違反 など)を扱う。ライブラリはプールの意味を解釈せず、与えられたテキストをそのまま few-shot 素材・スコアリング素材として使う。
- **入力は scikit-learn / imbalanced-learn の `(X, y)` 規約に合わせる。** `X: list[str]`(生テキスト)、`y`(2値ラベル)。`y` のどちらの値が "positive" かは `pntx._labels.resolve_binary_labels` で解決する(0/1 や -1/1 のような数値ペアは大きい方が自動的に positive、`"positive"`/`"negative"` 文字列はそのまま、それ以外の任意の2値(例: `"spam"`/`"ham"`)は `pos_label` 引数で明示)— 詳細は後述の「ラベル解決」節。ペアリングを強制しない点は従来通り(`y` でグルーピングした結果、positive/negative それぞれの件数が揃う必要はない)。**ただし `t2pn` の各 Classifier(`LLMPromptingClassifier.fit`/`FineTuningClassifier.fit`)、`pn2t` の各 over-sampler(`HardPositiveOverSampler.fit_resample`/`TypicalPositiveOverSampler.fit_resample`)はどちらも両クラスが最低1件ずつ存在することを要求する**(旧仕様にあった「片側のプールだけで fit」というスモークテスト用途は、sklearn/imblearn の分類データセット規約を採用したことに伴い廃止)。
- **llama.cpp インプロセス実行が LLM 系コンポーネントの主戦場。** 現在ビルトインで提供する LLM バックエンドは `LlamaCppBackend` のみ(`AnthropicBackend` は一旦廃止 — 経緯は後述)。設計判断で迷ったら llama.cpp での性能・体験を優先する。`t2pn.LLMPromptingClassifier`・`pn2t` とも、LLM 呼び出しは共通の `Backend` 抽象を経由し、同じロード済みモデル(例: 同一の `LlamaCppBackend` インスタンス)を共有できるようにする。**それぞれが独自の LLM クライアントを持って別々にモデルをロードする実装は禁止**(ローカル推論のメモリ/VRAM を二重に食うため)。リモートAPIバックエンドを将来また追加する場合も、`Backend` プロトコルを実装するだけで足りる設計(`_backend_resolve.py` のレジストリに1行足すだけ)は維持すること。**`t2pn.FineTuningClassifier` はこの `Backend` 抽象の対象外。** LLM 補完/スコアリングを一切使わず、`transformers`/`torch` で事前学習済みエンコーダを直接 fine-tuning する別経路であり、`LlamaCppBackend` などとモデルロードを共有する必要も想定もない。
- **`pn2t` には目的の異なる over-sampler がある。** `HardPositiveOverSampler`(分類器学習データ拡張、hard positive)と `TypicalPositiveOverSampler`(具体情報を一般化した典型的な正例、データ公開用途。差分プライバシーではないので「匿名化」とは呼ばない)。どれも positive 側のみ生成、二値ラベルのみサポート(エンコーディングは `resolve_binary_labels` 経由で `t2pn` と共通)という制約は共通。negative 側生成、3値以上への一般化は引き続きスコープ外(後述)。

## 公開 API(この形を維持すること)

```python
from pntx.t2pn import LLMPromptingClassifier, FineTuningClassifier
from pntx.pn2t import HardPositiveOverSampler

# --- t2pn: 分類 (scikit-learn Classifier) ---
clf = LLMPromptingClassifier(backend=...)   # Backend インスタンス、または "llama" 等のビルトインバックエンド名の文字列

X = ["この映画は最高だった", "サポートが丁寧で助かった", "この映画は退屈だった", "サポートの対応が雑だった"]
y = ["positive", "positive", "negative", "negative"]   # 0/1 でも可

clf.fit(X, y)                 # 学習ではなくプール保持+前処理(旧 PNTX.fit と同じ)
clf.predict(X)                # -> array-like of "positive"/"negative"(sklearn 標準の predict 契約)
clf.predict_proba(X)          # -> shape (n_samples, 2) の確率行列
clf.score(X, y)               # ClassifierMixin から無償で手に入る

clf.save(path)                 # backend は含めず positive/negative プール+設定だけを JSON へ
loaded = LLMPromptingClassifier.load(path, backend=...)   # backend は読み込み時に別途注入

# sklearn エコシステムにそのまま乗る
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X, y, cv=5)

# --- t2pn: 事前学習済みエンコーダの fine-tuning による分類 (同じ Classifier 契約に乗る別実装) ---
ft_clf = FineTuningClassifier(
    model_name="bert-base-multilingual-cased",  # 既定は multilingual BERT
    class_weight="balanced",  # 既定は None; sklearn 慣習で不均衡プールに対応
)

ft_clf.fit(X, y)               # 実際に事前学習済みエンコーダを fine-tuning する(学習が発生する)
ft_clf.predict(X)
ft_clf.predict_proba(X)
cross_val_score(ft_clf, X, y, cv=5)   # 同じ Estimator 契約なので LLMPromptingClassifier と差し替え可能

ft_clf.save(path)               # fine-tuned な重みごとディレクトリへ永続化(backend という概念がそもそもない)
loaded_ft = FineTuningClassifier.load(path)

# --- pn2t: 生成 (imbalanced-learn 流 over-sampler) ---
sampler = HardPositiveOverSampler(
    backend=...,
    sampling_strategy="auto",  # imblearn と同じ意味論(既定 "auto" はバランスするまで)
    random_state=0,
)

X_aug, y_aug = sampler.fit_resample(X, y)   # 生成された positive テキストが末尾に追加される
sampler.generation_result_                  # boundary feature 分析 + 生成根拠(pydantic モデル)

# imbalanced-learn の Pipeline にもそのまま乗る(imbalanced-learn 自体は必須依存にしない。
# fit_resample を duck-typing で提供するだけで imblearn.pipeline.Pipeline は使える)

# --- pn2t: 具体情報を一般化した典型的な正例の生成 (TypicalPositiveOverSampler) ---
from pntx.pn2t import TypicalPositiveOverSampler

sampler = TypicalPositiveOverSampler(backend=..., sampling_strategy={1: n_pos + 10})  # デフォルトなし、必須指定

X_syn, y_syn = sampler.fit_resample(X, y)     # negative は検証にのみ使い、プロンプトには含めない
sampler.generation_result_.synthetic_texts    # 生成テキスト + 監査用の generalized_from(何を一般化したか)

# --- pn2t: 負例の最小編集による反実仮想データ拡張 (CounterfactualOverSampler) ---
from pntx.pn2t import CounterfactualOverSampler

sampler = CounterfactualOverSampler(
    backend=...,
    verify=LLMPromptingClassifier(backend=...),  # 既定 "self"。"none" / 分類器(既定 verify_cv=5 でクロスフィッティング)も可
)
X_cf, y_cf = sampler.fit_resample(X, y)       # 各正例は既存の負例(pivot)の最小編集。pivot は X にあるのでペアになる
sampler.generation_result_.edits              # source_index/source_text → text、changed_spans、edit_ratio
sampler.generation_result_.rejected           # 棄却された候補と理由
```

`ClassifyResult`(`.label`/`.confidence`/`__eq__`)による1件ずつの結果表現は廃止し、`predict`/`predict_proba` は sklearn 標準の配列ベース契約に統一する。

0.16.0 で `OverSampler` → `HardPositiveOverSampler`、`SyntheticSampler` → `TypicalPositiveOverSampler`、`n_synthesized`/`seed` → `sampling_strategy`/`random_state` に改名した。旧名は `DeprecationWarning` 付きで 0.17.x まで残し、**0.18.0 で削除する**(クラス名は `pntx/pn2t/__init__.py` の PEP 562 `__getattr__`、パラメータは sklearn 慣習の `"deprecated"` センチネル既定値で実装)。

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

### `LLMEstimatorMixin`(`pntx/_sklearn.py`)— 変更なし、`t2pn.LLMPromptingClassifier`/`pn2t` 共有

- 既存の共有ミックスイン。`backend` を保持する Estimator の `sklearn.base.clone()` 互換性(`__sklearn_clone__` で `get_params()` から再構築し、`Backend` の非 deep-copy 可能な内部状態 — 例: `llama_cpp.Llama` の ctypes ポインタ — を deep copy しようとして壊れるのを防ぐ)専用であり、**`save`/`load` とは無関係**。
- `save`/`load` はこのミックスインでは提供しない。`pn2t` の over-sampler は共通基底 `BaseLLMOverSampler` が `save(path)`/`load(path, backend=...)` を一度だけ実装している(`backend` を除いた fitted state を JSON にシリアライズし、`load` 時に `backend` を再注入する)。`t2pn.LLMPromptingClassifier` の `save`/`load` も同じパターンを個別に実装している(**`backend` を素朴に pickle/JSON化しない**のが要点: `LlamaCppBackend` のようなロード済みモデルを抱えるオブジェクトを毎回シリアライズするのは重すぎるし、`llama_cpp.Llama` 内部状態はそもそも安全に pickle できる保証がない)。
- `t2pn.FineTuningClassifier` の `save`/`load` はこれらのどれとも別物: `backend` という概念がなく、代わりに学習済み重み自体を永続化する必要があるため独自実装になる(下記)。

### ラベル解決(`pntx/_labels.py`)— `t2pn`/`pn2t` 全クラス共有

`resolve_binary_labels(y, *, pos_label=None) -> (negative_label, positive_label)` を提供する共有ヘルパー。`t2pn.LLMPromptingClassifier.fit`/`t2pn.FineTuningClassifier.fit`/`pn2t` の各 over-sampler の `fit_resample` の全クラスがこれを通して `y` の2値のどちらが "positive" かを解決する(データセットのラベルエンコーディングを `t2pn`/`pn2t` 間で揃え直す必要がないようにするため)。解決規則:

1. `pos_label` が明示されていればそれが最優先(`y` に含まれる2値のどちらかである必要がある)。
2. 両方が数値(`int`/`float`/`bool`)なら大きい方が positive(`{0, 1}` → `1`、`{-1, 1}` → `1`)。
3. `"positive"`/`"negative"`(`pntx.types.POSITIVE`/`NEGATIVE`)ちょうどそのペアなら、そのまま使う。
4. それ以外(任意の非数値ペア、例: `{"spam", "ham"}`)は一意に決まらないため `pos_label` 必須 — 省略時は `ValueError`。

`y` に含まれる distinct な値が2つでない場合も `ValueError`(この検証を兼ねるため、`t2pn`/`pn2t` の各 `fit`/`fit_resample` は「両クラス最低1件」チェックを別途書かない — `resolve_binary_labels` が通れば自動的に満たされている)。`t2pn` 側の `classes_` は `[negative_label, positive_label]`(この順)で構築し、`predict_proba` の列順もこれに従う。`pn2t` 側は生成テキストを `positive_label` の値でラベル付けして末尾に追加する(`1` 決め打ちではない)。

### `t2pn` Classifier ファミリー(`pntx/t2pn/`)

`t2pn.py` 単一ファイルではなく `pntx/t2pn/` パッケージとし(`pn2t` と同型の構成)、分類アプローチの異なる複数の Classifier を持つ。全て `sklearn.base.BaseEstimator` + `ClassifierMixin` を継承し、`X: list[str]` / 二値ラベル `y` という共通の `(X, y)` 契約に従う。`X` は生テキストの list なので、数値配列を前提にした `check_X_y`/`check_array` を素通りさせるため estimator tags(`X_types: ["string"]` 相当、`no_validation` 系)を各クラスで適切に設定すること。`sklearn.utils.estimator_checks.check_estimator` に literal に通す必要はないが、`Pipeline`/`cross_val_score` で壊れないことは確認する。

#### `t2pn.LLMPromptingClassifier`(`pntx/t2pn/prompting.py`)

- `fit(X, y)` は `y` でグルーピングして positive/negative プールを作るだけで、学習は行わない(旧 `PNTX.fit` のロジックを流用)。`y` のラベルエンコーディングは `resolve_binary_labels`(前述)経由で解決する(0/1、-1/1、`"positive"`/`"negative"` は自動、それ以外は `pos_label` 明示)。
- 分類ロジック自体(exemplar 選択 → プロンプト構築 → スコアリング or パース)は既存の `pntx/selection.py` / `pntx/prompts.py` / `core.py` の分類パスをそのまま移設する。二段構えの分岐(`ScoringBackend` なら logprob 比較、そうでなければ生成テキストのパース)も維持。
- `predict_batch` 相当は `predict`/`predict_proba` がバッチを受け取れることで代替する(旧 `classify_batch` の「逐次 for ループ禁止、バックエンドごとに最適化」という制約はそのまま `predict`/`predict_proba` の内部実装に引き継ぐ)。
- `save`/`load` を追加する(`pn2t` の over-sampler と同じパターン)。`backend` を除いた fitted state(`classes_`/`positive_`/`negative_`/`exemplar_positive_`/`exemplar_negative_`/`exemplar_prefix_`/`calibration_weights_`)だけを JSON 化し、`load(path, backend=...)` で `backend` を再注入する。

#### `t2pn.FineTuningClassifier`(`pntx/t2pn/finetuning.py`)

- `LLMPromptingClassifier` とは分類アプローチが根本的に異なる: `Backend`/LLM 補完を一切使わず、`transformers` の `AutoModelForSequenceClassification`/`AutoTokenizer` で事前学習済みエンコーダに分類ヘッドを乗せて実際に fine-tuning する。プロンプト・exemplar 選択は不要。
- `AutoModelForSequenceClassification` はアーキテクチャ非依存(`model_name` を差し替えるだけで BERT/RoBERTa/DeBERTa/多言語モデルなど任意の HF hub チェックポイントに切り替えられる)なので、クラス名は特定の BERT アーキテクチャに縛られない `FineTuningClassifier` とする。ただし `model_name` の既定値は multilingual BERT(`bert-base-multilingual-cased`)にする(`TypicalPositiveOverSampler.sampling_strategy` と違い、自然なデフォルトが存在するため必須パラメータにはしない)。
- `fit(X, y)` は本当に学習を行う(この点は `LLMPromptingClassifier`/`pn2t` の over-sampler の「fit はプール保持のみで学習しない」という前提から明示的に外れる、`t2pn` 内で唯一の例外)。トークナイズ → 分類ヘッド(必要なら全体)の fine-tuning を `epochs`/`learning_rate`/`batch_size` 等のハイパーパラメータに従って行い、結果のモデル状態を `model_`/`tokenizer_` などの fitted attributes に保持する。
- `predict`/`predict_proba` はバッチ forward pass + softmax で実装する(`LLMPromptingClassifier` 同様、逐次 for ループは避ける)。
- `class_weight`(`None`(既定)/`"balanced"`/`{class_label: weight}` dict、sklearn 慣習)でクラス不均衡に対応する。`LLMPromptingClassifier` はプロンプトごとに大きい側のプールを小さい側に合わせてトリムすることでバランスを取るが、`FineTuningClassifier` は fit 済みプールをそのまま学習に使うため、不均衡なままだと損失が多数派クラスに引っ張られる — その対策は `class_weight` の責務であり、`pn2t` の over-sampler のようなサンプリング側のバランス調整は行わない。`"balanced"` は `n_samples / (2 * class_count)`(`sklearn.utils.class_weight.compute_class_weight("balanced", ...)` と同じ式)。`transformers` の `AutoModelForSequenceClassification` の `labels=` 経由の内蔵 loss はクラス重みを受け付けないため、`model(**encoded).logits` から `torch.nn.CrossEntropyLoss(weight=...)` で自前に loss を計算する。
- 学習が発生するため `LLMPromptingClassifier` と異なりデータ量・計算コストに敏感(GPU 推奨)。どの程度のデータ量から実用的かはベンチマークで検証する。
- optional dependency `pntx[finetuning]`(`transformers`, `torch`)未インストール時は明確な ImportError。
- `save`/`load` で fine-tuned な重みごと永続化できるようにする(`pn2t` の over-sampler の JSON ラウンドトリップとは異なり、モデル重みを含むためディレクトリ or アーカイブ形式になる想定)。

### exemplar 選択(`pntx/selection.py`)— 変更なし、`t2pn`/`pn2t` 共有

- `RandomSelector` / `DiversitySelector` / `NearestSelector` は従来通り。`Selector.select(pool, k, query)` インターフェースも維持。
- `pn2t` の over-sampler の exemplar サンプリングは「件数 k」ではなく「トークン予算」ベース(下記)なので、`Selector` をそのまま使うのではなく、予算ベースのサンプリングヘルパーを別途 `pntx/selection.py` に追加する(`sample_method` として `RandomSelector` 等と同じ戦略名を共有できる設計が望ましい)。

### `pn2t` over-sampler ファミリーの構成(`pntx/pn2t/`)

imbalanced-learn の `over_sampling` モジュールと同じ整理にする:

- **アルゴリズムごとに1クラス、名前は `<Name>OverSampler`。** 汎用名 `OverSampler` を特定アルゴリズムに使わない(imblearn に `OverSampler` という具象クラスがないのと同じ)。同一アルゴリズムの小さな変種はパラメータ(imblearn の `BorderlineSMOTE(kind=...)` 相当)、目的や生成メカニズムが違えば別クラス。今後追加する over-sampler(例: `research/ideas/counterfactual-edit-sampler.md` の `CounterfactualOverSampler`)もこの規則に従う。
- **共通基底 `BaseLLMOverSampler`(`pntx/pn2t/_base.py`、imblearn の `BaseOverSampler` 相当)。** `sampling_strategy`/`random_state` の解決、ラベル解決、バッチ生成ループ(リトライ上限・警告)、完全一致 dedup、exemplar サンプリング、`save`/`load` を一度だけ実装する。サブクラスはプロンプト構築・結果スキーマ・追加の受け入れフィルタなどのフックだけを実装する。`__init__` は sklearn の `get_params` のため各サブクラスで全パラメータを明示する。基底クラスはフックが不安定なため現時点では公開 API にしない(`pntx.pn2t.__all__` に含めない)。
- **共通パラメータ名は imblearn に揃える:** `sampling_strategy`(imblearn と同じ意味論を positive 側生成に限定したもの — `"auto"` 等の文字列、`(0, 1]` の float、`{label: リサンプリング後の総数}` の dict、それを返す callable。negative 側の生成を要求する指定は `ValueError`、文字列指定で負例が対象になる場合は0件生成)と `random_state`(`int`/`numpy.random.RandomState`/`None`)。解決ロジックは `pntx.pn2t._base.resolve_n_to_generate`(imbalanced-learn は import しない)。
- `sklearn.base.BaseEstimator` を継承する(imbalanced-learn の `BaseOverSampler` は継承しない。`fit_resample` を duck-typing で提供するだけで `imblearn.pipeline.Pipeline` から利用可能なため、`imbalanced-learn` 自体は必須依存に加えない)。

### `pn2t.HardPositiveOverSampler`(`pntx/pn2t/_hard_positive.py`)— `semaxis.HardPositiveOverSampler` の完全移植 + Backend 統合

- `fit_resample(X, y)`: 二値ラベルのみサポート。どちらが positive かは `resolve_binary_labels`(前述)で解決し、生成されたテキストはその positive ラベルの値で末尾に追加される(`1` 決め打ちではない)。**`pn2t` v1 は positive 側の生成のみ**(negative 側や3値以上への一般化は将来の拡張)。
- アルゴリズムは semaxis 実装をそのまま踏襲:
  1. positive/negative それぞれの exemplar をトークン予算内でサンプリング(`sample_method`: `random`/他、`context_limit` に基づく予算計算)。
  2. positive/negative の特徴・境界特徴(boundary features)を LLM に分析させ、"専門家なら positive と判定するが浅い分類器は negative と誤判定しうる" hard positive テキストを `batch_size` 件ずつバッチ生成。
  3. 完全一致ベースの dedup(`deduplicate=True` がデフォルト。元データ・既に採択した生成物との文字列一致のみを見る — 旧 `pntx/generate.py` にあった n-gram 近似重複除去とは別物で、v1 では使わない)。`TypicalPositiveOverSampler` はこれとは別目的の漏洩検出レイヤーを追加で持つ(下記)。
  4. `sampling_strategy`(既定 `"auto"` でクラスバランスまで)から求めた生成件数に達するまでバッチ生成を繰り返し、上限バッチ数に達したら警告付きで打ち切る。
- `generation_result_`(`positive_features`/`negative_features`/`boundary_features`/`hard_positives`、pydantic モデル)を fit 後に公開。`save(path)`/`load(path, backend=...)` で JSON へシリアライズ・復元できる(`backend` は除外し、`load` 時に再注入 — 実装は `BaseLLMOverSampler`)。
- コンストラクタ引数(`batch_size`, `max_examples_per_class`, `deduplicate`, `context_limit`, `language`, `sample_method`, `verbose`, `logger`)は semaxis 版を踏襲しつつ、`llm: BaseLLMClient | str` は `backend: Backend | str` に置き換える(pntx の `_resolve_backend` を再利用)。生成件数・乱数シード(旧 `n_synthesized`/`seed`)は imblearn 流の `sampling_strategy`/`random_state` に置き換えた(旧名は 0.18.0 まで非推奨エイリアス)。加えて `pos_label`(前述の `resolve_binary_labels` に渡す)を追加。

### `pn2t.CounterfactualOverSampler`(`pntx/pn2t/_counterfactual.py`)— 負例の最小編集による反実仮想データ拡張

設計の根拠と未決事項は `research/ideas/counterfactual-edit-sampler.md`(Kaushik et al. 2020 / Gardner et al. 2020 のノート参照)。

- 既存の**負例を pivot として**、正例ラベルが当てはまるようにする最小編集を LLM に作らせる(Kaushik et al. の3条件: ラベルが反転する・一貫性を保つ・不要な変更をしない)。pivot は元々 `X` にあるので、編集後の正例を末尾に足すだけで「元の例と反実仮想」のペアになる。**生成するのは正例のみ**で、正例 → 負例の編集はスコープ外(negative 側生成と同じ扱い)。
- 正例はプロンプトに**参照例としてだけ**入れる(正例の意味はユーザ定義なので、何に向けて編集するかを LLM に示す必要がある)。正例は pivot にしない。pivot は負例から `sample_method` で予算内に非復元抽出し、全負例を試し終えてから再利用する。
- 候補ごとのチェック順(基底クラスのパイプライン): ① 常に適用 — 形式(pivot にない `->`/`→` を含む = `changed_spans` の書式をテキストに混ぜた。qwen2.5-7B で実際に起きた)、no-op、最小性(`pntx.dedup.edit_ratio` が `max_edit_ratio` 以下、暫定既定 0.5)、`verify="self"` の自己判定 ② `deduplicate=True` のとき完全一致 dedup ③ 分類器による検証(バッチ単位)。棄却理由は `generation_result_.rejected` に残す。
- `verify` は `"none"`/`"self"`/`predict` を持つ分類器で、**既定は `"self"`**(パイロットベンチマーク `benchmarks/pn2t/counterfactual_pilot.py` で決定。結果はアイデアファイル参照: `"self"` は追加コストなしで `"none"` より精度が高く、難しい正しい編集を失わない。分類器は精度最高だが簡単な編集ばかり残す)。分類器の場合は `verify_cv`(既定 5)で `StratifiedGroupKFold`(同一テキストは同じ fold)によるクロスフィッティング: 各編集は自分の pivot を学習していない `clone(verify)` で判定し、clone は必要な fold の分だけ遅延 fit、渡されたインスタンス自体は fit しない。`verify_cv="prefit"` は渡された分類器をそのまま使う(`CalibratedClassifierCV(cv="prefit")` と同じ流儀)。
- LLM の出力スキーマ(`CounterfactualBatch`、`pivot_id` はバッチ内番号)と保存する結果(`CounterfactualGenerationResult`、`source_index` は `X` の添字)は別モデル。そのため基底クラスは結果型・バッチ型・アイテム型の3つの型パラメータを持つ。
- 0.16.0 の非推奨パラメータ(`n_synthesized`/`seed`)は受け付けない(新クラスなので)。

### `pn2t.TypicalPositiveOverSampler`(`pntx/pn2t/_typical_positive.py`)— 具体情報を一般化した典型的な正例の生成

- `HardPositiveOverSampler` とは独立したクラス(モード/パラメータではない)。目的関数が逆: `HardPositiveOverSampler` は境界を突く hard positive、`TypicalPositiveOverSampler` は典型的・平均的な positive を、原文の具体的な情報(固有名詞・人名・日付・数値・場所など)を含まないよう生成する。ユースケースはプライバシー上公開できない元テキストプールの代わりに、分布を代表する合成データセットを公開すること。ただし正例を生成プロンプトにそのまま入れるので差分プライバシーではなく、クラス名・ドキュメントで「匿名化」を約束しない(形式的保証が必要なら Aug-PE や DP fine-tuning を案内する)。
- `resolve_backend`・`LLMEstimatorMixin`(`pntx/_sklearn.py`、`clone()` 互換性用途で `t2pn.LLMPromptingClassifier` とも共有 — `save`/`load` とは無関係、上記参照)・`selection.sample_group`/`_SAMPLE_METHODS`・`pn2t._structured.complete_structured` など、`HardPositiveOverSampler` と同じ共有インフラ(`BaseLLMOverSampler`)の上に構築する。
- `fit_resample(X, y)` は `HardPositiveOverSampler` と同じ契約(二値ラベル、両クラス最低1件)を維持するが、**negative 側はラベル検証にのみ使い、生成プロンプトには含めない**(境界フレーミングを避けるため、かつ「positive に本質的 vs この1例に固有」の判断は複数の positive exemplar の共通性から行えるため)。
- `sampling_strategy` に `HardPositiveOverSampler` のような既定値(`"auto"`)はない(自然な目標がないため)。**デフォルトなしの必須パラメータ**にする(シグネチャ上は非推奨の `n_synthesized` を受け付けるため `None` 既定で、どちらも無ければ `fit_resample` で `ValueError`)。
- exemplar サンプリングは positive 側のみ(`HardPositiveOverSampler` の pos/neg 予算折半・バランス調整ロジックは不要)。token budget は `context_limit - overhead - max_tokens`(`HardPositiveOverSampler` と異なり `// 2` しない)。
- 具体情報の除去はプロンプト指示だけでなく、`pntx.dedup.contains_verbatim_span(text, sources, min_len)` によるベストエフォートの漏洩検出でも担保する: 生成テキストが positive プールから `min_verbatim_span`(デフォルト20文字)以上の連続部分文字列をそのままコピーしていたら reject してリトライする。これは `HardPositiveOverSampler` の完全一致 dedup とも旧 n-gram 近似重複除去とも別物(近似重複検出ではなく漏洩検出が目的、パラフレーズされた漏洩までは検出できないヒューリスティック)。
- `generation_result_`(`style_features`/`content_features`/`synthetic_texts`、pydantic モデル)を fit 後に公開。各 `synthetic_texts[].generalized_from` は「何を一般化したかの種類」の監査ログであり、元の具体的内容そのものを含めないようプロンプトで明示的に禁止する(この監査フィールド自体が漏洩経路にならないようにするため)。`save`/`load` は `BaseLLMOverSampler` の共通実装(`backend` を除外、`load(path, backend=...)` で再注入)。

## パッケージング

- **コア依存として `scikit-learn` と `pydantic` を必須にする**(`t2pn` の各 Classifier の Estimator 契約、`pn2t` の over-sampler の構造化出力検証にそれぞれ必須のため)。「本体はゼロ依存」という従来方針は撤回し、ゼロ依存の対象はバックエンド実装(LLM SDK)・埋め込み系・`FineTuningClassifier` 用の学習ライブラリに限定する。
- optional dependencies:
  - `pntx[llama]` → `llama-cpp-python`
  - `pntx[finetuning]` → `transformers`, `torch`(`t2pn.FineTuningClassifier` の fine-tuning 用)
  - `pntx[embeddings]` → 埋め込みベースの dedup / DiversitySelector 用
  - (`pntx[anthropic]` → `anthropic` は `AnthropicBackend` の廃止に伴い削除。復活時は `_BACKEND_REGISTRY` にエントリを1行足すのと合わせて追加する)
- 未インストールのバックエンドを使おうとしたら、インストールコマンドを含む明確な ImportError を出す。
- `pyproject.toml`(`uv_build`)、Python 3.10+。

## 実装順序

1. 既存の `pntx/backends/`・`pntx/selection.py`・`pntx/prompts.py`・`pntx/_sklearn.py`(`LLMEstimatorMixin`)はそのまま流用。`pntx/core.py`(`PNTX` ファサード)と `pntx/generate.py`(旧 verify/dedup 生成ループ)は削除。
2. `pntx/t2pn/prompting.py`: 既存の分類ロジック(`core.py` の `classify`/`classify_batch` 相当)を `LLMPromptingClassifier` として sklearn `BaseEstimator`/`ClassifierMixin` + `LLMEstimatorMixin` に載せ替え。`fit(X, y)` のラベルグルーピング、text-input 用の estimator tags 設定、`save`/`load` を追加。
3. `pntx/pn2t/`: `Backend.complete()` 上の構造化出力ヘルパー(JSON プロンプト + pydantic 検証 + リトライ)。boundary feature プロンプト(`pntx/pn2t/prompts.py`)。予算ベース exemplar サンプリング。`HardPositiveOverSampler` のロジック本体の移植(`fit_resample`、`LLMEstimatorMixin` 経由の `save`/`load`)。
4. `benchmarks/t2pn/run.py` を新 `t2pn.LLMPromptingClassifier` に合わせて更新。
5. `AnthropicBackend` 経由での `t2pn`/`pn2t` 動作確認(構造化出力ヘルパーがバックエンド非依存であることの検証)。
6. `NearestSelector`/`DiversitySelector` の `pn2t` 側サンプリングへの統合、埋め込み optional。
7. `pntx/t2pn/finetuning.py`: `FineTuningClassifier` を追加(`pntx[finetuning]` optional dependency、`Backend` 抽象とは独立)。`LLMPromptingClassifier` と同じ `(X, y)` 契約・`predict`/`predict_proba` 出力形状を維持しつつ、実際の fine-tuning ループを実装。

## テスト

- バックエンドは `FakeBackend`(決め打ち応答を返す `ScoringBackend` 実装)でモックする。実モデル・実 API を叩くテストは `tests/integration/` に分離し、デフォルトでは skip。
- `t2pn.LLMPromptingClassifier`: `fit`/`predict`/`predict_proba` のユニットテストに加えて、`sklearn.pipeline.Pipeline`・`cross_val_score` に組み込んで壊れないことを確認するテストを持つ。`save`/`load` の往復(`backend` を含めずシリアライズされること、`load` 時に別の `FakeBackend` を注入して復元できること)も対象。
- `t2pn.FineTuningClassifier`: 事前学習済みチェックポイントのダウンロードなしでユニットテストを完結させるため、`transformers` の `AutoConfig`(小さい `hidden_size`/`num_hidden_layers` 等)からランダム初期化した極小モデルで `fit`/`predict`/`predict_proba`・`Pipeline`/`cross_val_score` 互換・`save`/`load` を検証する(開発環境によっては huggingface.co 等の外部ホストに到達できない場合があるため、実在の事前学習済みチェックポイントのダウンロードを伴う確認は `tests/integration/` 側に分離する)。`class_weight`(`None`/`"balanced"`/dict)の重み計算が正しいこと、不正な値で `ValueError` になることも対象。
- `pn2t.HardPositiveOverSampler`: 「boundary feature 分析 → hard positive 生成 → dedup で棄却 → リトライ → 上限到達で警告」の分岐を必ずカバー。`sampling_strategy="auto"` のクラスバランス自動計算、`save`/`load` の往復も対象。
- `pn2t.TypicalPositiveOverSampler`: `HardPositiveOverSampler` と同様の分岐に加え、negative 側がプロンプトに含まれないことの直接検証、`contains_verbatim_span` による漏洩 dedup(reject → リトライ、`min_verbatim_span` 可変、`deduplicate=False` で無効化されること)を必ずカバー。
- `pn2t.CounterfactualOverSampler`: 最小編集の受理(日本語・英語)、`max_edit_ratio` 超過・no-op・形式不正・完全一致重複・不正な `pivot_id` の棄却とリトライ、正例が pivot にならないこと、`verify` の3方式、クロスフィッティング(各編集を判定する clone の学習データに pivot も生成物も含まれない、渡したインスタンスは fit されない、`"prefit"` は clone/fit しない、K が小さい方のクラス件数を超えると `ValueError`)。LLM 出力は pivot に依存するので、テストはプロンプトから pivot を読み取って編集を返す fake backend を使う。
- `pn2t` 共通(`BaseLLMOverSampler`): `sampling_strategy` の各形式(文字列・float・dict・callable)が imblearn の over-sampling と同じ件数を返すこと、不正値・negative 側の生成要求で `ValueError` になること、`random_state` の int/`RandomState`、旧クラス名・旧パラメータの `DeprecationWarning` と互換動作(0.18.0 の削除時にこれらのテストも消す)。
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
- 「成果物としての生成」(品質重視・任意サイド生成)は `pn2t.TypicalPositiveOverSampler` として実装済み。ただし意味的なパラフレーズ漏洩の自動検出(embedding ベースの類似度検証等)は引き続きスコープ外 — `contains_verbatim_span` による verbatim コピー検出のみのベストエフォート
- 埋め込みモデル自体の学習(`DiversitySelector`/`NearestSelector`/dedup で使う埋め込みは既存の学習済みモデルを利用するのみ)。`t2pn.FineTuningClassifier` による分類器の fine-tuning はスコープ内(上記アーキテクチャ参照)
- CLI(ライブラリ API のみ。CLI は将来別途)
