#!/usr/bin/env bash
# run_bench.sh  ——  仅用 make 做 16×5 次跑分
set -euo pipefail

tiles=(32 64 128 256)
modes=(0 1 2 3 4)
csv=results.csv

echo "tile_m,tile_n,copy_mode,avg_time_ms" > "$csv"

for tm in "${tiles[@]}"; do
  for tn in "${tiles[@]}"; do
    echo -e "\n=== build TILE_M=$tm TILE_N=$tn ==="
    # 强制重新编译；若 Makefile 支持 incremental，可去掉 clean
    make clean >/dev/null
    make TILE_M=$tm TILE_N=$tn -j$(nproc)

    for mode in "${modes[@]}"; do
      echo "   >> run mode=$mode"
      out=$(./main --copy_mode="$mode" --iterations=200)

      # 抓 “Average time: XXX ms”——倒数第二个字段就是时间
      t=$(echo "$out" | awk '/Average time:/ {print $(NF-1)}')

      echo "      ${t} ms"
      echo "$tm,$tn,$mode,$t" >> "$csv"
    done
  done
done

echo -e "\n全部完成！结果保存在 $csv"
