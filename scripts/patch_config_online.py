#!/usr/bin/env python3
"""在远程服务器上运行，将 config.toml 更新为 RunningHub 工作流 + 随机 cosplay 配置"""
import re

CONFIG_PATH = "/root/maibot/MaiBot-latest/plugins/1021143806_mais_art_journal/config.toml"

RUNNINGHUB_BLOCK = '''
# RunningHub 小绿 LoRA 工作流
name = "RunningHub-小绿LoRA"
format = "runninghub-workflow"
api_key = "d75ec2e065604c85b286e1b57c63e00a"
model = "2002109398342352897"
fixed_size_enabled = true
default_size = "1024x1520"
node_info_list = [
  { nodeId = "48", fieldName = "编辑文本", fieldValue = "${prompt}" },
  { nodeId = "13", fieldName = "width", fieldValue = "${width}" },
  { nodeId = "13", fieldName = "height", fieldValue = "${height}" }
]
'''

def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. generation.default_model
    content = re.sub(
        r'(default_model\s*=\s*)"[^"]*"',
        r'\1"runninghub_xiaolv"',
        content,
        count=1
    )

    # 2. selfie: add random_style, keep default_style
    if "random_style" not in content:
        content = content.replace(
            "# 自拍默认风格",
            "# 是否随机选择自拍风格（standard/mirror/photo/cosplay）\nrandom_style = true\n\n# 自拍默认风格"
        )

    # 3. pic_command_model
    content = re.sub(
        r'(pic_command_model\s*=\s*)"[^"]*"',
        r'\1"runninghub_xiaolv"',
        content,
        count=1
    )

    # 4. auto_selfie.selfie_model
    content = re.sub(
        r'(selfie_model\s*=\s*)"[^"]*"',
        r'\1"runninghub_xiaolv"',
        content,
        count=1
    )

    # 5. Add [models.runninghub_xiaolv] if not exists
    if "[models.runninghub_xiaolv]" not in content:
        # 在 [models.model1] 之前插入
        content = content.replace(
            "[models.model1]",
            "[models.runninghub_xiaolv]" + RUNNINGHUB_BLOCK + "\n[models.model1]"
        )

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print("config.toml 已更新")

if __name__ == "__main__":
    main()
