# DA4 adapter conversion benchmarks (OMMX v3)

固定seedからOMMX v3 Instanceを直接生成し、`inspect_ommx_v2`と同じ問題を測定します。

## Instance

| Instance | 目的関数・制約 | Formulation | 推奨サイズ |
| --- | --- | --- | --- |
| `knapsack` | Binary線形目的関数と不等式 | `regular` | 100 / 400 / 900 |
| `assignment` | Binary線形目的関数と重複OneHot | `regular` / `one-hot` | 10 / 20 / 30 |
| `tsp` | Binary 2次目的関数と重複OneHot | `regular` / `one-hot` | 10 / 20 / 30 |
| `clique` | 1次・2次等式のペナルティ化 | `regular` | 10 / 20 / 30 |
| `one-hot-preparation` | Indicator/SOS1のPreparation | `one-hot` | 10 / 20 / 30 |

v3では通常制約とfirst-classな`OneHotConstraint`を別々に生成します。
`assignment`と`tsp`では行方向と列方向のOneHotが変数を共有します。変数数が同じ場合は
先に生成したグループが`_one_hot_dict`を通じてDA4 native groupへ入り、重複する残りは
`_penalty_one_hot_dict`を通じて通常等式相当のペナルティへ変換されます。

## Preparation

`one-hot-preparation`では`--special-constraints`から次を選択します。

| Case | Source | Preparation後 |
| --- | --- | --- |
| `none` | OneHot | Preparationなし |
| `indicator` | OneHot + Indicator | OneHotを保持し、Indicatorを通常制約へlower |
| `sos1` | OneHot + SOS1 | OneHotを保持し、SOS1を通常制約へlower |
| `indicator-sos1` | OneHot + Indicator + SOS1 | OneHotを保持し、両方をlower |

特殊制約はOneHotから導かれる冗長制約なので、すべてのケースで実行可能領域は同一です。

## 測定対象

- `prepare`: コピー・Policy生成を除いた`Instance.prepare()`のみ
- `instance-to-request`: 必要なPreparation後の`OMMXDA4Adapter(instance).sampler_input`
- `response-to-solution`: 合成済みの実行可能`QuboResponse`に対する`adapter.decode()`

DA4 API、ネットワーク通信、solver時間は測定に含めません。時間測定では初回実行と、
3回のウォームアップ後20回の中央値を記録します。メモリ測定では初回と2回目の
ピークメモリを記録します。

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
      --special-constraints "$special_constraints" --size "$size" \
      | tee "benchmark_results/v3-${special_constraints}-prepare-timing-${size}.csv"

    uv run --frozen python -m benchmarks.timing instance-to-request \
      --instance one-hot-preparation --formulation one-hot \
      --special-constraints "$special_constraints" --size "$size" \
      | tee "benchmark_results/v3-${special_constraints}-request-timing-${size}.csv"
  done
done
```

## ピークメモリ

サイズごとに別プロセスで実行します。

```console
uv run --frozen --with memray python -m benchmarks.memory instance-to-request \
  --instance tsp --formulation one-hot --size 20

uv run --frozen --with memray python -m benchmarks.memory prepare \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 --size 20
```
