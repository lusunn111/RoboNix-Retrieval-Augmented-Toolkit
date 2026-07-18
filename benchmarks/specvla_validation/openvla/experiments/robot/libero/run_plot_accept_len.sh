#!/bin/bash
# 绘制accept_length散点图
#
# 使用方法: bash run_plot_accept_len.sh

set -e

SPECVLA_ROOT="/path/to/SpecVLA"
cd $SPECVLA_ROOT

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate specvla

# 设置环境变量
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO

echo "=========================================="
echo "绘制 Accept Length 散点图"
echo "=========================================="

python openvla/experiments/robot/libero/plot_accept_len.py

echo ""
echo "=========================================="
echo "完成！图片保存在:"
echo "  openvla/experiments/robot/libero/figs/"
echo "=========================================="
