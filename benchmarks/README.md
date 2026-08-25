# DA4 adapter conversion benchmarks (OMMX v2)

固定seedからOMMX v2 Instanceを直接生成し、OMMX v3版と同じ問題を測定します。
OMMXはv2系最新の2.6.1へ固定します。

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
uv run --frozen --with memray python -m benchmarks.memory instance-to-request \
  --instance tsp --formulation one-hot --size 20

uv run --frozen --with memray python -m benchmarks.memory response-to-solution \
  --instance tsp --formulation one-hot --size 20

uv run --frozen --with memray python -m benchmarks.memory instance-to-request \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 --preparation none --size 20
```
