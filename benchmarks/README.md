# DA4 adapter conversion benchmarks (OMMX v3)

固定seedからOMMX v3 Instanceを直接生成し、`inspect_ommx_v2`と共通の問題を測定します。
`update_ommx_v3`のコミット`510b7d0`を取り込み、OMMX 3.0.0b5で検証します。

## 測定結果

- [2026年9月8日: OMMX 2.6.1 / 3.0.0b5、全294測定](benchmark-results-20260908-beta5.md)
- [2026年8月25日: OMMX v2/v3比較（Preparation workload整合前）](benchmark-results-20260825.md)

上記は次数検証の追加前に取得した結果です。`510b7d0`取り込み後の再測定は未実施です。

## Instance

| Instance | 目的関数・制約 | Formulation | 推奨サイズ |
| --- | --- | --- | --- |
| `knapsack` | Binary線形目的関数と不等式 | `regular` | 100 / 400 / 900 |
| `assignment` | Binary線形目的関数と重複OneHot | `regular` / `one-hot` | 10 / 20 / 30 |
| `tsp` | Binary 2次目的関数と重複OneHot | `regular` / `one-hot` | 10 / 20 / 30 |
| `one-hot-preparation` | Indicator/SOS1のPreparation | `one-hot` | 10 / 20 / 30 |

既存の`clique`は、サイズ10 / 20 / 30・seed 0で2次等式の2乗後に4次項が残り、
DA4の次数上限を超えるため、全件測定の対象から除外しています。
生成関数は残していますが、Adapterへの変換時に`OMMXDA4AdapterError`になります。

v3では通常制約とfirst-classな`OneHotConstraint`を別々に生成します。
`assignment`と`tsp`では行方向と列方向のOneHotが変数を共有します。変数数が同じ場合は
先に生成したグループが`_one_hot_dict`を通じてDA4 native groupへ入り、重複する残りは
`_penalty_one_hot_dict`を通じて通常等式相当のペナルティへ変換されます。

## Preparation

`one-hot-preparation`では`--special-constraints`で比較するlowering式を、
`--preparation`でdirect／preparedを選択します。`indicator-sos1`は前半の
グループをIndicator、後半をSOS1とし、特殊制約の合計を常に`size`個にします。

| Preparation | Adapterへ渡すactive constraints | 用途 |
| --- | --- | --- |
| `none` | OneHot `size`個 + lowering後と同じ通常制約 `size`個 | v3 direct。`prepare()`を呼ばない |
| `recommended` | OneHot `size`個 + lowering済み通常制約 `size`個 | SourceのIndicator/SOS1合計`size`個を事前にlowering |

`--preparation none`ではactiveなIndicator/SOS1を生成せず、そのlowering結果を
通常制約として直接生成します。`recommended`のSourceではfirst-classなIndicator/SOS1を
生成し、Adapterの推奨Policyで通常制約へloweringします。したがってdirectとpreparedは
active制約数・数式・変数・目的関数・DA4 Requestが同一で、Preparation履歴だけが異なります。

Indicator/SOS1相当制約はOneHotから導かれる冗長制約なので、すべてのケースで
実行可能領域と目的関数も同一です。v2では同じlowering結果を通常制約として直接生成します。

## 測定対象

- `prepare`: コピー・Policy生成を除いた`Instance.prepare()`のみ
- `instance-to-request`: direct、または測定外でPreparationした後の`OMMXDA4Adapter(instance).sampler_input`
- `response-to-solution`: 合成済みの実行可能`QuboResponse`に対する`adapter.decode()`

DA4 API、ネットワーク通信、solver時間は測定に含めません。時間測定では初回実行と、
3回のウォームアップ後20回の中央値を記録します。メモリ測定では初回と2回目の
ピークメモリを記録します。

## 再現可能な全件測定

Python 3.12.10、seed 0、合成レスポンス1種類・frequency 16を共通条件とします。
`sample_count`はfrequencyであり、異なる解の数ではありません。CSVには
`unique_solutions`も記録します（decodeは1、その他の処理は0）。
合成状態はInstanceの変数メタデータから構成し、実行可能性とDA4の変数ID対応を
計測開始前に検証します。レスポンス生成・実行可能性検証はdecode時間に含みません。

DA4のトークンは不要です。DA4Client・ネットワーク・求解を呼ぶ経路、
`sample()` / `solve()`やE2Eの測定はありません。公開APIは既存テストで回帰確認します。
ベンチマークテストではDA4Clientの生成があれば失敗させます。

```console
uv sync --frozen --all-extras --group benchmark --python 3.12.10
uv run --frozen --group benchmark python -m benchmarks.run \
  --output benchmark_results/20260908-beta5 \
  --seed 0 --sample-count 16 --warmup 3 --repeat 20
```

`benchmarks.run`は全ケースを順番に実行し、測定ごとに別プロセスを起動します。
他のベンチマークを並列に実行しないでください。`--metric timing`または`memory`で
片方だけ実行できます。出力先は新しいディレクトリを指定します。

- `timing.csv`: 初回時間、warmup後の中央値、seed・warmup・repeat・依存バージョン。
- `memory.csv`: 初回と2回目のmemrayピーク、seed・memray・依存バージョン。
- `metadata.json`: Git SHA、作業ツリー状態、lockfileのSHA256、実行環境、全依存、
  測定ケース、完了件数、成否。失敗時には部分結果とエラーを残します。

Instance生成、コピー、Policy生成はすべて測定外です。`prepare`を除きPreparationも
測定外です。時間測定中はGCを停止します。memray 1.20.0は測定専用の
`benchmark` dependency groupに固定し、Adapterの配布依存には追加しません。
memrayのピークは追跡対象アロケーションの指標で、プロセス全体のRSSではありません。

DA4のv2はOneHot hintsをnative groupの選択に使うので、v2/v3ともに
`assignment` / `tsp`の`regular`と`one-hot`を測定します。
v3ではnative groupの開始位置を保証するゼロ係数項がRequestへ追加されます。
v2/v3の同値性確認では変数IDの対応を戻し、ゼロ係数項を除いて数式を比較します。

| Operation | 通常問題 | Preparation比較用 | 合計 |
| --- | ---: | ---: | ---: |
| `instance-to-request` | 15 | 21 | 36 |
| `response-to-solution` | 15 | 21 | 36 |
| `prepare` | 0 | 9 | 9 |
| 合計 | 30 | 51 | 81 |

時間81条件＋メモリ81条件＝162測定です。比較用21条件はOneHotのみの
baseline 3サイズ＋特殊制約3種 × direct/prepared × 3サイズです。
`prepare`はlowering対象がある3種 × 3サイズのみを測定します。
v2/v3の比較には、両ブランチに共通するClique以外の条件を使用します。

## 時間

```console
mkdir -p benchmark_results

for size in 100 400 900; do
  uv run --frozen python -m benchmarks.timing instance-to-request \
    --instance knapsack --formulation regular --size "$size" \
    | tee "benchmark_results/v3-knapsack-instance-to-request-timing-${size}.csv"
done

for formulation in regular one-hot; do
  for size in 10 20 30; do
    uv run --frozen python -m benchmarks.timing instance-to-request \
      --instance assignment --formulation "$formulation" --size "$size" \
      | tee "benchmark_results/v3-assignment-${formulation}-request-timing-${size}.csv"

    uv run --frozen python -m benchmarks.timing response-to-solution \
      --instance tsp --formulation "$formulation" --size "$size" \
      | tee "benchmark_results/v3-tsp-${formulation}-decode-timing-${size}.csv"
  done
done

for special_constraints in indicator sos1 indicator-sos1; do
  for size in 10 20 30; do
    uv run --frozen python -m benchmarks.timing prepare \
      --instance one-hot-preparation --formulation one-hot \
      --special-constraints "$special_constraints" \
      --preparation recommended --size "$size" \
      | tee "benchmark_results/v3-${special_constraints}-prepare-timing-${size}.csv"

    for preparation in none recommended; do
      uv run --frozen python -m benchmarks.timing instance-to-request \
        --instance one-hot-preparation --formulation one-hot \
        --special-constraints "$special_constraints" \
        --preparation "$preparation" --size "$size" \
        | tee "benchmark_results/v3-${special_constraints}-${preparation}-request-timing-${size}.csv"

      uv run --frozen python -m benchmarks.timing response-to-solution \
        --instance one-hot-preparation --formulation one-hot \
        --special-constraints "$special_constraints" \
        --preparation "$preparation" --size "$size" \
        | tee "benchmark_results/v3-${special_constraints}-${preparation}-decode-timing-${size}.csv"
    done
  done
done
```

## ピークメモリ

サイズごとに別プロセスで実行します。

```console
uv run --frozen --group benchmark python -m benchmarks.memory instance-to-request \
  --instance tsp --formulation one-hot --size 20

uv run --frozen --group benchmark python -m benchmarks.memory prepare \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 --preparation recommended --size 20

uv run --frozen --group benchmark python -m benchmarks.memory instance-to-request \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 --preparation none --size 20

uv run --frozen --group benchmark python -m benchmarks.memory instance-to-request \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 --preparation recommended --size 20
```
