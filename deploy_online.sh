#!/bin/bash
# 线上部署脚本：拉取最新代码并应用 RunningHub 工作流配置
# 用法：在插件目录执行 ./deploy_online.sh

set -e
echo "=== 拉取最新代码 ==="
git pull origin main

echo ""
echo "=== 检查 config.toml ==="
if [ ! -f config.toml ]; then
    echo "config.toml 不存在，请先创建或从 config.toml.example 复制"
    exit 1
fi

echo ""
echo "=== 配置说明 ==="
echo "参考 config.runninghub.example.toml 将 RunningHub 工作流配置合并到 config.toml"
echo "关键配置："
echo "  - generation.default_model = runninghub_xiaolv"
echo "  - selfie.random_style = true （自拍随机选 standard/mirror/photo/cosplay）"
echo "  - models.runninghub_xiaolv 需填写真实 api_key"
echo ""
echo "=== 部署完成 ==="
