# pntx

[English README](https://github.com/pillyshi/pntx/blob/main/README.md)

`pntx` は、ユーザが用意した `positive`/`negative` の2つのテキストプールから、独立した2つのコンポーネントを提供する Python ライブラリです。

1. **`pntx.t2pn`**(テキスト → positive/negative)— 任意のテキストを `positive` か `negative` に分類する [scikit-learn](https://scikit-learn.org/) の `Classifier` 群です。
   - **`LLMPromptingClassifier`**: LLM の few-shot プロンプティング/スコアリングで分類します(学習なし)。
   - **`FineTuningClassifier`**: 事前学習済みの `transformers` エンコーダを実際に fine-tuning します(既定は multilingual BERT)。
2. **`pntx.pn2t`**(positive/negative → テキスト)— 目的の異なる [imbalanced-learn](https://imbalanced-learn.org/) 流の over-sampler 群です。
   - **`HardPositiveOverSampler`**: 分類器の学習データを補うため、判別の難しい正例(hard positive)を生成します。
   - **`CounterfactualOverSampler`**: 既存の負例を最小限に編集して正例にします(反実仮想データ拡張)。
   - **`TypicalPositiveOverSampler`**: 典型的な正例を生成します(hard positive の対極)。これも分類器の学習データを補うためのものです。

「positive」と「negative」の意味はユーザが決めます。感情のポジネガに限らず、フォーマル/カジュアル、規約準拠/違反など、例で定義できる任意の対比を扱えます。`pntx` はプールの意味を解釈せず、few-shot やスコアリングの素材として使うだけです。

```python
from pntx.t2pn import LLMPromptingClassifier
from pntx.pn2t import HardPositiveOverSampler

# --- t2pn: LLM の few-shot プロンプティングによる分類(学習なし) ---
clf = LLMPromptingClassifier(backend="llama", backend_kwargs={"model_path": "model.gguf"})

X = ["この映画は最高だった", "サポートが丁寧で助かった",
     "この映画は退屈だった", "サポートの対応が雑だった"]
y = ["positive", "positive", "negative", "negative"]  # 0/1 でも可

clf.fit(X, y)
clf.predict(["店員さんがとても親切だった"])        # array(['positive'], dtype='<U8')
clf.predict_proba(["店員さんがとても親切だった"])  # shape (1, 2)、列の順は clf.classes_ に従う

# scikit-learn のエコシステムにそのまま乗る
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X, y, cv=5)

# fit 済みのプールを保存(backend は含めない。読み込み時に渡し直す)
clf.save("classifier.json")
loaded = LLMPromptingClassifier.load("classifier.json", backend="llama",
                                     backend_kwargs={"model_path": "model.gguf"})

# --- t2pn: 事前学習済みエンコーダの fine-tuning による分類(pntx[finetuning]) ---
from pntx.t2pn import FineTuningClassifier

ft_clf = FineTuningClassifier(class_weight="balanced")  # 既定の model_name は multilingual BERT
ft_clf.fit(X, y)                 # こちらは実際に学習する
ft_clf.predict_proba(["店員さんがとても親切だった"])
ft_clf.save("finetuned/")        # テキストだけでなく学習済みの重みも保存する
loaded_ft = FineTuningClassifier.load("finetuned/")

# --- pn2t: hard positive の生成(imbalanced-learn 流の over-sampler) ---
sampler = HardPositiveOverSampler(
    backend="llama",
    backend_kwargs={"model_path": "model.gguf"},
    sampling_strategy="auto",  # 既定: クラスがバランスするまで正例を生成する
    random_state=0,
)

X_aug, y_aug = sampler.fit_resample(X, [1, 1, 0, 0])  # 二値ラベルのみ。positive クラスは 1
sampler.generation_result_.hard_positives  # 生成したテキストと、それぞれの LLM による根拠

# --- pn2t: (hard ではなく)典型的な正例の生成 ---
from pntx.pn2t import TypicalPositiveOverSampler

typical = TypicalPositiveOverSampler(
    backend="llama",
    backend_kwargs={"model_path": "model.gguf"},
    sampling_strategy={1: 2 + 10},  # 既存の正例 2件 + 新規 10件。既定は "auto"
)
X_typ, y_typ = typical.fit_resample(X, [1, 1, 0, 0])
typical.generation_result_.typical_positives  # 生成したテキスト

# --- pn2t: 負例の最小編集による反実仮想データ拡張 ---
from pntx.pn2t import CounterfactualOverSampler

cf = CounterfactualOverSampler(
    backend="llama",
    backend_kwargs={"model_path": "model.gguf"},
    verify="self",  # 既定。"none" や分類器も指定できる(後述)
)
X_cf, y_cf = cf.fit_resample(X, [1, 1, 0, 0])
cf.generation_result_.edits     # (source_index, source_text) -> text と、変更箇所
cf.generation_result_.rejected  # 棄却された候補とその理由
```

`HardPositiveOverSampler.fit_resample` は「hard positive」を生成します。専門家なら positive と判定するものの、浅い分類器や訓練を受けていない人なら negative と誤判定しかねないテキストです。生成の前に、2つのクラスを分ける特徴を backend に分析させます。

- [`semaxis`](https://github.com/pillyshi/semaxis) の同名クラスを完全に移植したもので、`pntx` 独自の `Backend` 抽象を経由します。そのため、自前でモデルを読み込まずに `LLMPromptingClassifier` とロード済みのモデルを共有できます。
- v1 が生成するのは positive 側だけで、二値ラベルのみに対応します(`0`/`1`、`-1`/`1`、`"positive"`/`"negative"`、それ以外のペアは `pos_label` で指定)。
- `imbalanced-learn` 自体は依存に含めていません。`fit_resample` を duck-typing で提供しているので、別途インストールすれば `imblearn.pipeline.Pipeline` から使えます。
- hard positive は contrast set や反実仮想編集(既存の例を最小限に編集してラベルを反転させたもの)ではありません。それぞれがラベル positive の新しい例であり、「難しさ」は LLM の判断で、分類器で検証したものではありません。

`sampling_strategy` は imbalanced-learn と同じ意味論を、positive クラスの生成に限定したものです。

- `"auto"`(`HardPositiveOverSampler` の既定): 正例の数が負例と並ぶまで生成します。
- `(0, 1]` の float: リサンプリング後の正例/負例の比率です。
- dict `{pos_label: n}`: リサンプリング後の正例の*総数*です。
- そのような dict を返す callable `f(y)` も使えます。

負例の生成が必要になる指定は `ValueError` になります。`random_state`(int か `numpy.random.RandomState`)は exemplar のサンプリングのシードです。

`TypicalPositiveOverSampler.fit_resample` は hard positive の対極にあたる、positive クラスの*典型的な*テキストを生成します。正例プールの文体や内容を再現しつつ、exemplar の言い換えではなく、それぞれ新しい状況を描いたテキストにします。

- `sampling_strategy` の既定は `HardPositiveOverSampler` と同じく `"auto"` です。
- 負例のプールも必要ですが(同じ二値ラベルの検証のため)、backend には一切見せません。生成には正例の exemplar だけを使います。負例と対比させると、典型例ではなくクラス境界を軸にした生成になってしまうためです。
- 完全一致の重複除去に加えて、正例の exemplar から長い部分文字列をそのままコピーした生成文を棄却して再生成するチェック(`min_verbatim_span`、既定 20文字)があります。
- **匿名化はしません。** exemplar はそのままプロンプトに入り、生成文にその具体的な情報が残ることがあります。非公開データの代わりに公開するデータを作る用途には向きません。
- 典型的な正例が分類器の役に立つかは、まだベンチマークしていません(下記参照)。

`CounterfactualOverSampler.fit_resample` は*ペア*を作ります。生成する正例はそれぞれ、既存の負例(「pivot」)を、positive のラベルが当てはまるように最小限だけ編集したものです。

- 編集の指示は [Kaushik et al. (ICLR 2020)](https://arxiv.org/abs/1909.12434) の反実仮想の書き換え指示に従います。ラベルが反転すること、文章として一貫していること、それ以外は何も変えないこと、の3条件です。
- pivot は元から `X` にあるので、結果で学習すると、最小限の変更の両側を見ることになります。同論文が示したとおり、この仕組みによって分類器が見せかけの特徴に頼りにくくなります。
- 候補はすべて、形式チェック、無変更(no-op)チェック、最小性チェック(`max_edit_ratio`: pivot のうち変更された部分の文字単位の割合、既定 `0.5`)を通過する必要があります。その後、完全一致の重複除去を経て、ラベルを検証します。

- `verify="none"` は編集をそのまま信用します。`verify="self"` は、LLM 自身が同じ応答の中で positive ではないと判定した編集を捨てます(追加の呼び出しは不要ですが、編集した本人が自分の出力を採点することになり、小さなモデルはこれが苦手です)。
- `verify=<分類器>`(例: 同じロード済みモデルを使う `LLMPromptingClassifier(backend=backend)` や `FineTuningClassifier`)は、分類器が positive と予測しなかった編集を捨てます。既定ではクロスフィッティングします(`verify_cv=5`)。各編集は、その pivot で学習していない clone で判定されます。判定対象の編集元の負例そのもので学習した検証器は、正しい反転を棄却しがちだからです。`verify_cv="prefit"` を指定すると、fit 済みの分類器をそのまま使います。

`verify` の既定値 `"self"` は、Kaushik et al. の IMDb データでのパイロットベンチマーク(`benchmarks/pn2t/counterfactual_pilot.py`)で決めました。

- `"self"` は追加コストなしで `"none"` より精度が高くなりました。
- 分類器による検証が最も精度が高かったものの、残った編集の多くは浅い分類器にとってもともと簡単なものでした。
- 検証方法よりも、編集に使うモデルのほうが重要です。長いレビューを完全に反転できたのは 7B モデルで約半分でした(人手の編集は同じ LLM 判定で約 0.8〜0.9)。3B モデルはほとんど全面的に書き換えてしまいました。
- 日本語では、より強い編集モデルを使ってください。後述のベンチマークでは、qwen2.5-7B では効果がなく、Gemma 4 12B では効果がありました。

棄却された候補は理由(`"edit_too_large"`、`"self_check_failed"`、`"verifier_rejected"` など)とともに `generation_result_.rejected` に残るので、各フィルタが何を捕まえたかを確認できます。

### データ拡張は分類器の役に立つか？

`benchmarks/pn2t/downstream.py` で、Kaushik et al. の反実仮想拡張済み IMDb データを使って測定しました。

**設定.** 少量で不均衡な学習データ(正例 25件 / 負例 125件のレビュー)に、各条件で正例を 100件追加します。拡張した条件はどれも、既存の正例を複製して同じ件数にした**比較対象(duplicate)**と比べます。この比較対象が重要です。エポック数が固定なら、例が増えるだけで学習量が増え、クラスのバランスも良くなるからです。

評価には2つのテストセットを使います。元のテストレビューと、その*反実仮想の書き換え版*(同じレビューを人手で最小限に編集してラベルを反転させたもの)です。結果は 3 seed で、生成モデルは llama.cpp 上の qwen2.5-7B(Q4_K_M)です。

**結果.** 書き換え版テストでの、比較対象に対する AUC の改善幅(括弧内は比較対象を上回った seed の数):

| 条件 | TF-IDF + ロジスティック回帰 | BERT(`FineTuningClassifier`) |
|---|---|---|
| EDA(Wei & Zou 2019) | −0.020 (1/3) | −0.165 (0/3) |
| `HardPositiveOverSampler` | −0.050 (0/3) | +0.021 (2/3) |
| **`CounterfactualOverSampler`** | **+0.171 (3/3)** | **+0.196 (3/3)** |
| 人手の反実仮想編集(参考) | +0.281 (3/3) | +0.208 (3/3) |

**わかったこと.**

- **`CounterfactualOverSampler` は、どちらの分類器でもすべての seed で効果がありました。** BERT では、書き換え版テストで人手の反実仮想編集による改善の約 94% を再現しました(AUC 0.966 対 0.978、正解率 0.889 対 0.907。比較対象は 0.770 / 0.610)。元のテストでも人手と同等でした(AUC 0.921 対 0.919)。
- **EDA も `HardPositiveOverSampler` も、既存の正例を複製するだけの場合を安定して上回ることはありませんでした。**
- **効果は別のレビュードメインにも及びます。** IMDb で学習した同じモデルを、正例と負例を半々にしたドメイン外のサンプル(Yelp 1,000件、Amazon 1,000件)でも評価しました。書き換え版テストは人手の参考データと同じ作り方をしている、という懸念に答えるためです。
  - BERT では、`CounterfactualOverSampler` は今回も 3/3 の seed で比較対象を上回りました。AUC の改善は Yelp で +0.070、Amazon で +0.060 です。人手の反実仮想編集は +0.074 と +0.068 で、AUC はどちらも 0.946(人手は 0.950 と 0.953)に達しました。
  - TF-IDF では +0.125 と +0.082(3/3)で、人手の反実仮想編集は +0.168 と +0.132 でした。
- **日本語でも効果がありますが、編集モデルが十分に強い場合に限られます。** 同じ設計を、[WRIME](https://github.com/ids-cv/wrime) の日本語 SNS 投稿で、3つの編集モデルを使って実施しました。
  - Gemma 4 12B(QAT、Q4)では、`CounterfactualOverSampler` は 3/3 の seed で比較対象を上回りました。AUC の改善は TF-IDF で +0.040、日本語 BERT で +0.038(0.915 対 0.877)です。
  - qwen2.5-7B では効果がありませんでした。日本語の編集の約半分が棄却され、残りの多くも正例とは言いにくいものでした。
  - Gemma 4 E4B はその中間でした。
  - `HardPositiveOverSampler` はどの編集モデルでも効果がありませんでした。
  - WRIME には人手の反実仮想編集がないので、ここには参考となる上限がありません。

**注意点.**

- 3 seed で、学習データの件数は1通りだけです。英語の結果は、学習ドメインが1つ(映画レビュー)、生成モデルも1つです。
- WRIME では、正解率の改善が AUC の改善よりずっと大きくなっています。これは主に分類器が正例をより多く予測するようになったためで、順位付けが良くなったためではないので、AUC で比べてください。
- BERT の設定は、チューニングしていないライブラリの既定値(3エポック、`max_length=128`)です。
- 書き換え版テストは人手の参考データと同じ作り方をしています。上のドメイン外の結果にはこの偏りはありません。
- `TypicalPositiveOverSampler` はベンチマークの条件に加えましたが、まだ実行していません。

全体の表、seed ごとの範囲、注意点は `research/ideas/downstream-augmentation-benchmark.md` に、ベンチマークの再実行方法は `benchmarks/README.md` にあります。

> **0.19.0 より前からのアップグレード.** `TypicalPositiveOverSampler` は、プライバシー目的ではなく分類器のデータ拡張用の sampler という位置づけになりました。プロンプトで具体的な情報の一般化を指示しなくなり、`sampling_strategy` の既定が `"auto"` になり、結果は `generation_result_.typical_positives`(各要素は `text` のみ)になりました。`synthetic_texts` と `generalized_from` は廃止です。以前のバージョンで保存した結果は読み込めません。

> **0.16.0 より前からのアップグレード.** `OverSampler` は `HardPositiveOverSampler` に、`SyntheticSampler` は `TypicalPositiveOverSampler` に、それらの `n_synthesized`/`seed` パラメータは `sampling_strategy`/`random_state`(imbalanced-learn の名前)になりました。旧名は 0.16.0 で非推奨となり、**0.18.0 で削除済み**なので、現在は `ImportError`/`TypeError` になります。`n_synthesized=n` は `sampling_strategy={pos_label: 既存の正例数 + n}` に、`n_synthesized=None` は `"auto"` に置き換えてください。int の `random_state` は、同じ int の `seed` と同じ結果になります。

## インストール

`pntx` はパッケージ管理に [uv](https://docs.astral.sh/uv/) を使います。

```bash
uv add pntx                # コア(scikit-learn + pydantic)
uv add "pntx[llama]"       # + llama.cpp のインプロセス backend
uv add "pntx[finetuning]"  # + FineTuningClassifier(transformers + torch)
uv add "pntx[embeddings]"  # + selector 用の意味的類似度
```

`scikit-learn` と `pydantic` はコア依存です(それぞれ、`t2pn` の各分類器の scikit-learn 契約と、`pn2t` の over-sampler の構造化 LLM 出力に必要です)。それ以外の backend や機能はそれぞれ extra に分かれていて、インストールせずに使おうとすると、実行すべきインストールコマンドを含む明確な `ImportError` が出ます。

## Backend

`pntx` は `Backend` プロトコルを通してモデルを動かします。これは `LLMPromptingClassifier` と `pn2t` の over-sampler で共有されます。`FineTuningClassifier` はこの抽象を**一切使いません**。LLM の呼び出しがなく、fine-tuning した `transformers` エンコーダだけを使うので、他と共有するロード済みモデルがありません。

- **`LlamaCppBackend`**(`pntx[llama]`)— `llama-cpp-python` で GGUF モデルをインプロセスで動かします。最も重点的に調整している主要な backend です。分類ではトークンの対数確率を直接使い(`score_choices`)、バッチ分類では共通の few-shot prefix の KV キャッシュを全件で使い回し、1件ごとに再評価しません。

```python
clf = LLMPromptingClassifier(backend="llama", backend_kwargs={"model_path": "model.gguf"})

# backend のインスタンスを直接渡すこともできる(テストでの依存性注入など)
from pntx.backends.llama import LlamaCppBackend
clf = LLMPromptingClassifier(backend=LlamaCppBackend(model_path="model.gguf"))
```

リモート API の backend は、`Backend` プロトコル(`pntx.backends.base.Backend`)を実装してインスタンスを直接渡せば、後から追加できます。現在、ビルトインのものは同梱していません。

`backend_kwargs` は `backend` を文字列で指定したときだけ使われます。`**kwargs` ではなく1つの dict にしているのは、`LLMPromptingClassifier` と `pn2t` の over-sampler を scikit-learn の `get_params()`/`clone()` と互換に保つためです。

`LlamaCppBackend` には、ローカルの `model_path` か、Hugging Face Hub から `Llama.from_pretrained` で GGUF モデルを取得するための `repo_id`(`filename` で1ファイルに絞り込み可)を指定します。それ以外のキーワード(`n_ctx`、`n_gpu_layers`、`flash_attn`、`verbose` など)はそのまま `llama_cpp.Llama` に渡されます。

```python
clf = LLMPromptingClassifier(
    backend="llama",
    backend_kwargs={
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "*q4_k_m.gguf",
        "n_ctx": 4096,
        "n_gpu_layers": -1,  # 全レイヤーを GPU にオフロード
        "flash_attn": True,
    },
)
```

1つのロード済みモデルを `LLMPromptingClassifier` と `pn2t` の over-sampler で共有するには(ローカル推論では推奨。同じ GGUF を二重に読み込まずに済みます)、backend を一度だけ作って、そのインスタンスをそれぞれに渡します(`CounterfactualOverSampler` の検証器にも渡せます)。

```python
from pntx.backends.llama import LlamaCppBackend

backend = LlamaCppBackend(model_path="model.gguf")
clf = LLMPromptingClassifier(backend=backend)
sampler = HardPositiveOverSampler(backend=backend)
typical = TypicalPositiveOverSampler(backend=backend)
cf = CounterfactualOverSampler(backend=backend, verify=LLMPromptingClassifier(backend=backend))
```

## Exemplar の選択

fit したテキストが(どちらの側でも)プロンプトに無理なく収まる数より多いときは、`Selector` がどれを使うかを決めます。`LLMPromptingClassifier` は positive と negative のプールそれぞれに対して独立に呼び出します。

- **`RandomSelector`**(既定)— 一様ランダムな部分集合。
- **`NearestSelector`** — 分類対象のテキストに最も似たテキストを選びます。テキストごとに動的に選択します。
- **`DiversitySelector`** — 最大限多様な部分集合を貪欲に選びます。
- **`BudgetSelector`** — トークン予算に収まるだけ選びます(`pn2t` の over-sampler が exemplar のサンプリングに内部で使います)。

`NearestSelector` と `DiversitySelector` は `similarity_fn` を取ります。既定は依存なしの文字 n-gram 類似度(`pntx.dedup.similarity`)です。意味的な類似度を使うには `pntx.embeddings.cosine_similarity_fn()`(`pntx[embeddings]` が必要)を渡します。

```python
from pntx.t2pn import LLMPromptingClassifier
from pntx.selection import NearestSelector

clf = LLMPromptingClassifier(
    backend="llama", backend_kwargs={"model_path": "model.gguf"}, selector=NearestSelector()
)
```

`LLMPromptingClassifier` は selector の `query_aware` 属性によって、**静的**か**動的**かを判断します(`RandomSelector`/`DiversitySelector`/`BudgetSelector` は静的で、ビルトインで動的なのは `NearestSelector` だけです)。

- **静的**(既定): exemplar の選択・並び順・キャリブレーション(後述)はすべて `fit()` で一度だけ決まり、以降の `predict`/`predict_proba` の呼び出しで使い回されます。これにより、`BatchScoringBackend`(例: `LlamaCppBackend`)は共通の few-shot prefix を1回の呼び出しにつき一度だけ評価し、その KV キャッシュをバッチ全体で使い回せます。
- **動的**(例: `NearestSelector`): 各テキストに本当に関連する exemplar は事前にはわからないので、`predict`/`predict_proba` の中でテキストごとに選択をやり直します。`BatchScoringBackend` では上記の共通 prefix の KV キャッシュの使い回しができなくなり(テキストごとに prefix と backend の呼び出しが別になる)、`LLMPromptingClassifier` はこのレイテンシのトレードオフを知らせるため、呼び出しごとに一度 `UserWarning` を出します。

`LLMPromptingClassifier` は `ScoringBackend` の経路で、既定で **content-free calibration**(Zhao et al. 2021, "Calibrate Before Use")も行います。空のプレースホルダのクエリを同じ few-shot prefix でスコアリングして、prefix 自体のラベルの偏り(どの exemplar がどの順で入ったかによる産物で、few-shot プロンプトはこれに敏感なことが知られています)を推定し、実際の予測をその基準値で割って正規化し直します。無効にして生の(キャリブレーションしていない)softmax を得るには、`LLMPromptingClassifier(..., calibrate=False)` を渡します。

`pn2t` の over-sampler は `Selector` を取りません。代わりに、コンストラクタ引数 `sample_method` で*予算ベース*のサンプリング戦略を選びます(`HardPositiveOverSampler` では semaxis 自身の `sample_method`/`embedding_model` の完全な移植で、`TypicalPositiveOverSampler` も正例だけの exemplar サンプリングに同じ仕組みを使います)。

- **`"random"`**(既定)— 一様ランダムな部分集合を、トークン予算が尽きるまで詰めます(内部では `BudgetSelector`)。
- **`"kmeans"`** — `embedding_model`(`pntx[embeddings]` が必要)でプールを埋め込み、K-Means のクラスタごとに代表テキストを1つ選びます。
- **`"votek"`** — プールを埋め込み、代表性と多様性のバランスをとる Vote-K アルゴリズム(Su et al. 2022)を実行します。

```python
sampler = HardPositiveOverSampler(
    backend="llama",
    backend_kwargs={"model_path": "model.gguf"},
    sample_method="votek",
    embedding_model="paraphrase-albert-small-v2",  # sentence-transformers のモデル名
)
```

## 開発

```bash
uv sync                          # 開発用の依存をインストール
uv run pytest                    # ユニットテスト(統合テストは既定で skip)
uv run ruff check .
uv run mypy src tests
```

実モデルや API を叩く統合テストはオプトインです。

```bash
PNTX_LLAMA_MODEL_PATH=/path/to/model.gguf uv run pytest tests/integration
```
