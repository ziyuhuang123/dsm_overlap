#!/bin/bash

# --- 配置区 ---

# 1. 定义要修改的源文件
SOURCE_FILE="main.cu"
# 2. 定义输出结果的CSV文件名
CSV_FILE="benchmark_results.csv"
# 3. 定义每次运行的迭代次数
ITERATIONS=200
# 4. 定义copy_iter的循环范围
START_ITER=1
END_ITER=6

# --- 脚本主逻辑 ---

# 检查源文件是否存在
if [ ! -f "$SOURCE_FILE" ]; then
    echo "错误: 源文件 '$SOURCE_FILE' 未找到。"
    exit 1
fi

# 在开始前，先备份原始文件，以便结束后恢复
cp "$SOURCE_FILE" "$SOURCE_FILE.bak"
echo "原始文件已备份为 $SOURCE_FILE.bak"

# 创建CSV文件并写入表头
echo "copy_iter,average_time_ms" > "$CSV_FILE"
echo "基准测试结果将保存在 '$CSV_FILE' 文件中。"

echo "--------------------------------------"
echo "--- 开始执行基准测试循环 ---"
echo "--------------------------------------"

# 循环遍历您指定的copy_iter值
for copy_iter in $(seq $START_ITER $END_ITER); do
    echo "=> 正在测试: copy_iter = $copy_iter"

    # 步骤 1: 修改C++源文件
    # 使用sed命令查找目标行，并替换第一个模板参数为当前的copy_iter值
    # -i 表示直接在原文件上修改
    # 正则表达式 s/(...<)[0-9]+,/\1$copy_iter,/ 匹配并替换
    echo "   - 正在修改源文件..."
    sed -i -E "s/(copy_host_tma_load_and_store_kernel_multicast<)[0-9]+,/\1$copy_iter,/" "$SOURCE_FILE"

    # 步骤 2: 编译代码
    # 将编译输出和错误都重定向到errors.txt
    echo "   - 正在编译 (make)..."
    make > errors.txt 2>&1
    
    # 检查make命令是否成功执行
    if [ $? -ne 0 ]; then
        echo "   - 错误: 编译失败 (copy_iter = $copy_iter)。请检查 errors.txt 文件获取详细信息。"
        # 编译失败时，恢复原始文件并退出
        mv "$SOURCE_FILE.bak" "$SOURCE_FILE"
        exit 1
    fi

    # 步骤 3: 执行程序并捕获输出
    echo "   - 正在运行 ./main --iterations=$ITERATIONS..."
    output=$(./main --iterations=$ITERATIONS)

    # 步骤 4: 从输出中解析出平均时间
    # 使用grep找到包含"Average time:"的行，然后用awk提取第3个字段（即时间数值）
    avg_time=$(echo "$output" | grep "Average time:" | awk '{print $3}')

    # 步骤 5: 检查是否成功提取到时间
    if [ -z "$avg_time" ]; then
        echo "   - 警告: 未能为 copy_iter = $copy_iter 提取到平均时间。程序输出如下："
        echo "$output"
        # 记录一个空值以保持CSV格式一致
        echo "$copy_iter," >> "$CSV_FILE"
    else
        echo "   - 提取到平均时间: $avg_time ms"
        # 步骤 6: 将结果追加到CSV文件
        echo "$copy_iter,$avg_time" >> "$CSV_FILE"
    fi
    echo "" # 输出一个空行，让日志更清晰
done

echo "--------------------------------------"
echo "--- 基准测试循环结束 ---"
echo "--------------------------------------"

# 循环结束后，从备份中恢复原始文件
mv "$SOURCE_FILE.bak" "$SOURCE_FILE"
echo "原始文件 $SOURCE_FILE 已恢复。"
echo ""

# 打印最终的CSV结果
echo "最终测试结果如下:"
cat "$CSV_FILE"
