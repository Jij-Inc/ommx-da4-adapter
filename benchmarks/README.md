# DA4 adapter conversion benchmarks (OMMX v3)

固定seedからOMMX v3 Instanceを直接生成し、`inspect_ommx_v2`と同じ問題を測定します。

## 測定結果

- [2026年8月25日: OMMX v2/v3比較（Preparation workload整合前）](benchmark-results-20260825.md)

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
uv run --frozen --with memray python -m benchmarks.memory instance-to-request \
  --instance tsp --formulation one-hot --size 20

uv run --frozen --with memray python -m benchmarks.memory prepare \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 --preparation recommended --size 20

uv run --frozen --with memray python -m benchmarks.memory instance-to-request \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 --preparation none --size 20

uv run --frozen --with memray python -m benchmarks.memory instance-to-request \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 --preparation recommended --size 20
```
