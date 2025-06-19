import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import sys

# ==============================================================================
# (0) 环境设置：中文字体
# ==============================================================================
# 此设置会尝试从一个预设的列表中寻找可用的中文字体
preferred_fonts = ["SimHei", "Microsoft YaHei", "SimSun", "Noto Sans CJK SC", "Source Han Sans CN"]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for font in preferred_fonts:
    if font in available_fonts:
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False

# ==============================================================================
# (1) 数据读取与处理
# ==============================================================================
csv_filename = 'results_new.csv'
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"错误：找不到文件 '{csv_filename}'。")
    print("请确保您的数据文件和这个Python脚本在同一个文件夹中。")
    sys.exit()

# 验证CSV文件是否包含必要的列
required_columns = {'tile_m', 'tile_n', 'copy_mode', 'avg_time_ms'}
if not required_columns.issubset(df.columns):
    print(f"错误：CSV文件 '{csv_filename}' 缺少必要的列。")
    print(f"需要包含的列: {required_columns}")
    sys.exit()

# 创建 'block_size' 列
df['block_size'] = df['tile_m'].astype(str) + 'x' + df['tile_n'].astype(str)

# 按照 tile_m 的值进行分组
grouped_by_tile_m = df.groupby('tile_m')

# ==============================================================================
# (2) 为每个 tile_m 分组绘制并保存图表
# ==============================================================================
print("开始生成分组图表...")

# 为每个分组循环，生成一张图表
for tile_m_value, group_df in grouped_by_tile_m:

    # 准备当前图表的数据
    # 使用 pivot_table 来重塑数据，方便绘图
    pivot_group = group_df.pivot_table(index='block_size', columns='copy_mode', values='avg_time_ms')

    # 获取X轴的标签和位置
    x_labels = pivot_group.index
    x_positions = np.arange(len(x_labels))

    copy_mode_labels = ["global only", "global + DSM(TMA)", "global + global", "global + DSM(reg)", "DSM(TMA) only"]
    bar_width = 0.15

    # 设置画布
    fig, ax = plt.subplots(figsize=(12, 7))

    # 循环绘制5种copy_mode的柱子
    for i, mode in enumerate(range(5)):
        # 计算每个系列柱子的中心位置
        offset = (i - 2) * bar_width  # 偏移量，-2, -1, 0, 1, 2
        bar_positions = x_positions + offset

        # 获取当前copy_mode的数据
        if mode in pivot_group.columns:
            bar_values = pivot_group[mode]
            rects = ax.bar(bar_positions, bar_values, bar_width, label=copy_mode_labels[i])
            ax.bar_label(rects, fmt="%.3f", padding=3, fontsize=8, rotation=90)

    # --- 图表美化 ---
    ax.set_ylabel("时间 (ms)")
    ax.set_xlabel("Block Size")
    ax.set_title(f"不同复制策略执行时间 (tile_m = {tile_m_value})")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.legend(ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15))

    ax.margins(y=0.2)  # 增加Y轴的上边距，给数值标签留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # 调整布局，防止标题和图例重叠

    # --- 保存图表 ---
    output_filename = f"block_size_comparison_tile_m_{tile_m_value}.png"
    plt.savefig(output_filename)
    print(f"图表已成功保存为: '{output_filename}'")
    plt.close(fig)  # 关闭当前图表，防止在内存中累积

print("\n所有分组图表已全部生成完毕。")