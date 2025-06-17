#!/bin/bash

# --- 配置区 ---

# 1. 定义要修改的源文件
SOURCE_FILE="main.cu"

# 2. 定义存放所有结果的文件夹名
RESULTS_DIR="benchmark_results"

# 3. 定义每次运行的迭代次数
ITERATIONS=200

# 4. 定义要测试的Tile尺寸数组
TILE_SIZES=(32 64 128 256)

# 5. 定义共享内存上限（单位：KB）
SMEM_LIMIT_KB=227

# --- 脚本主逻辑 ---

# 检查源文件是否存在
if [ ! -f "$SOURCE_FILE" ]; then
    echo "错误: 源文件 '$SOURCE_FILE' 未找到。"
    exit 1
fi

# 在开始前，先备份原始文件，以便结束后恢复
cp "$SOURCE_FILE" "$SOURCE_FILE.bak"
echo "原始文件已备份为 $SOURCE_FILE.bak"

# 创建结果文件夹，如果不存在的话
mkdir -p "$RESULTS_DIR"
echo "基准测试结果将保存在 '$RESULTS_DIR/' 文件夹中。"
echo "============================================================"

# 开始双重循环，遍历所有Tile尺寸组合
for tile_m in "${TILE_SIZES[@]}"; do
  for tile_n in "${TILE_SIZES[@]}"; do
    
    # --- 为每个组合创建独立的CSV文件 ---
    CSV_FILE="${RESULTS_DIR}/results_${tile_m}x${tile_n}.csv"
    echo "concurrent_copies,average_time_ms" > "$CSV_FILE"
    echo "=> 开始测试组合: TILE_M = $tile_m, TILE_N = $tile_n. 结果将写入 $CSV_FILE"

    # --- 步骤 1: 计算最大并发拷贝数 ---
    # sizeof(half_t) = 2 bytes
    smem_per_copy_kb=$(( (tile_m * tile_n * 2) / 1024 ))
    
    if [ $smem_per_copy_kb -eq 0 ]; then
        echo "   - 警告: 单个Tile尺寸过小，跳过此组合。"
        continue
    fi

    max_copies=$(( SMEM_LIMIT_KB / (smem_per_copy_kb + 1) ))

    if [ $max_copies -lt 1 ]; then
      echo "   - 警告: Tile尺寸 ($tile_m x $tile_n) 过大。跳过此组合。"
      echo ""
      continue
    fi
    echo "   - 最大并发拷贝数: $max_copies. 将从 $max_copies 迭代到 1."

    # --- 新增的内层循环：从最大并发数迭代到1 ---
    for (( copy_iter=$max_copies; copy_iter>=1; copy_iter-- )); do
      echo "    -> 正在测试: concurrent_copies = $copy_iter"

      # --- 步骤 2: 动态生成要替换的目标行 ---
      target_line_pattern="copy_host_tma_load_and_store_kernel_multicast<.*>(M, N, iterations);"
      replacement_line="copy_host_tma_load_and_store_kernel_multicast<$copy_iter, false, 2, $tile_m, $tile_n>(M, N, iterations);"

      sed -i -E "s/$target_line_pattern/$replacement_line/" "$SOURCE_FILE"
      echo "       - 已修改 '$SOURCE_FILE'。"

      # --- 步骤 3: 编译和运行 ---
      echo "       - 正在编译 (make)..."
      make > errors.txt 2>&1
    
      if [ $? -ne 0 ]; then
        echo "       - 错误: 编译失败！(concurrent_copies = $copy_iter)。请检查 errors.txt。"
        continue
      fi

      echo "       - 正在运行 ./main --iterations=$ITERATIONS..."
      output=$(./main --iterations=$ITERATIONS)

      # --- 步骤 4: 解析结果并记录 ---
      avg_time=$(echo "$output" | grep "Average time:" | awk '{print $3}')

      if [ -z "$avg_time" ]; then
        echo "       - 警告: 未能提取到平均时间。可能是运行时错误。"
        echo "$output"
        echo "$copy_iter,ERROR" >> "$CSV_FILE"
      else
        echo "       - 提取到平均时间: $avg_time ms"
        echo "$copy_iter,$avg_time" >> "$CSV_FILE"
      fi
    done # 内层循环结束

    echo "------------------------------------------------------------"

  done # tile_n 循环结束
done # tile_m 循环结束

# --- 清理工作 ---
# 循环结束后，从备份中恢复原始文件
mv "$SOURCE_FILE.bak" "$SOURCE_FILE"
echo "============================================================"
echo "测试完成！原始文件 $SOURCE_FILE 已恢复。"
echo ""
echo "所有结果已保存在 '$RESULTS_DIR' 文件夹中。"
