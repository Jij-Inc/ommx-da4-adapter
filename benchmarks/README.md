# DA4 adapter conversion benchmarks (OMMX v2)

固定seedからOMMX v2 Instanceを直接生成し、OMMX v3版と同じ問題を測定します。
比較基準のOMMXは2.6.1へ固定します。

## Instance

| Instance | 目的関数・制約 | Formulation | 推奨サイズ |
| --- | --- | --- | --- |
| `knapsack` | Binary線形目的関数と不等式 | `regular` | 100 / 400 / 900 |
| `assignment` | Binary線形目的関数と重複OneHot | `regular` / `one-hot` | 10 / 20 / 30 |
| `tsp` | Binary 2次目的関数と重複OneHot | `regular` / `one-hot` | 10 / 20 / 30 |
| `clique` | 1次・2次等式のペナルティ化 | `regular` | 10 / 20 / 30 |
| `one-hot-preparation` | v3 Preparation専用Instanceの比較基準 | `one-hot` | 10 / 20 / 30 |

v2のOneHotは通常の等式制約と`ConstraintHints.OneHot`の組で表現します。
`assignment`と`tsp`では行方向と列方向のOneHotが変数を共有します。DA4 native
groupに採用されなかったOneHotは、通常の等式制約としてペナルティへ変換されます。

`one-hot-preparation`はv3版と同じ変数・目的関数・OneHotグループを持つbaselineです。
v2にはfirst-classなIndicator/SOS1と`Instance.prepare()`がないため、特殊制約を
指定した場合はv3のlowering後と同じ不等式を通常制約として直接生成します。

| `--special-constraints` | 通常制約として追加するlowering結果 |
| --- | --- |
| `none` | なし |
| `indicator` | Indicator相当を`size`個 |
| `sos1` | SOS1相当を`size`個 |
| `indicator-sos1` | 合計`size`個（前半Indicator相当、後半SOS1相当） |

このため、v2の直接変換とv3の直接変換／`recommended` Preparation後では、
変数・目的関数・OneHot・通常制約の数と内容を揃えて比較できます。

## 測定対象

- `instance-to-request`: `OMMXDA4Adapter(instance).sampler_input`
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
| `instance-to-request` | 18 | 12 | 30 |
| `response-to-solution` | 18 | 12 | 30 |
| 合計 | 36 | 24 | 60 |

時間60条件＋メモリ60条件＝120測定です。Preparation比較用の12条件は
`none` / `indicator` / `sos1` / `indicator-sos1` × 3サイズです。

## 時間

```console
mkdir -p benchmark_results

for size in 100 400 900; do
  uv run --frozen python -m benchmarks.timing instance-to-request \
    --instance knapsack --formulation regular --size "$size" \
    | tee "benchmark_results/v2-knapsack-instance-to-request-timing-${size}.csv"
done

for formulation in regular one-hot; do
  for size in 10 20 30; do
    uv run --frozen python -m benchmarks.timing instance-to-request \
      --instance assignment --formulation "$formulation" --size "$size" \
      | tee "benchmark_results/v2-assignment-${formulation}-request-timing-${size}.csv"

    uv run --frozen python -m benchmarks.timing response-to-solution \
      --instance tsp --formulation "$formulation" --size "$size" \
      | tee "benchmark_results/v2-tsp-${formulation}-decode-timing-${size}.csv"
  done
done

for special_constraints in none indicator sos1 indicator-sos1; do
  for size in 10 20 30; do
    uv run --frozen python -m benchmarks.timing instance-to-request \
      --instance one-hot-preparation --formulation one-hot \
      --special-constraints "$special_constraints" --preparation none \
      --size "$size" \
      | tee "benchmark_results/v2-one-hot-preparation-${special_constraints}-request-timing-${size}.csv"

    uv run --frozen python -m benchmarks.timing response-to-solution \
      --instance one-hot-preparation --formulation one-hot \
      --special-constraints "$special_constraints" --preparation none \
      --size "$size" \
      | tee "benchmark_results/v2-one-hot-preparation-${special_constraints}-decode-timing-${size}.csv"
  done
done
```

## ピークメモリ

サイズごとに別プロセスで実行します。

```console
uv run --frozen --group benchmark python -m benchmarks.memory instance-to-request \
  --instance tsp --formulation one-hot --size 20

uv run --frozen --group benchmark python -m benchmarks.memory response-to-solution \
  --instance tsp --formulation one-hot --size 20

uv run --frozen --group benchmark python -m benchmarks.memory instance-to-request \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 --preparation none --size 20
```
