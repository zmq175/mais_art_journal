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
echo "=== 请确保 config.toml 包含以下配置 ==="
echo "将以下内容合并到 config.toml 中（若已存在可跳过）："
echo ""
cat << 'CONFIG_EOF'

# ----- 以下为 RunningHub 工作流 + 自拍随机风格配置 -----

[generation]
default_model = "runninghub_xiaolv"

[selfie]
random_style = true
default_style = "standard"

[models.runninghub_xiaolv]
name = "RunningHub-小绿LoRA"
format = "runninghub-workflow"
api_key = "你的RunningHub_API_KEY"
model = "2002109398342352897"
fixed_size_enabled = true
default_size = "1024x1520"
node_info_list = [
  { nodeId = "48", fieldName = "编辑文本", fieldValue = "${prompt}" },
  { nodeId = "13", fieldName = "width", fieldValue = "${width}" },
  { nodeId = "13", fieldName = "height", fieldValue = "${height}" }
]

CONFIG_EOF
echo ""
echo "请将 api_key 替换为你的真实 RunningHub API Key"
echo ""
echo "=== 部署完成 ==="
