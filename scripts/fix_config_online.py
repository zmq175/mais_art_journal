#!/usr/bin/env python3
"""修复 config.toml：恢复备份并正确插入 runninghub_xiaolv"""
import re

CONFIG_PATH = "/root/maibot/MaiBot-latest/plugins/1021143806_mais_art_journal/config.toml"
BACKUP_PATH = CONFIG_PATH + ".bak"

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
    with open(BACKUP_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. generation.default_model
    content = re.sub(
        r'(default_model\s*=\s*)"[^"]*"',
        r'\1"runninghub_xiaolv"',
        content,
        count=1
    )

    # 2. selfie: add random_style
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

    # 5. 只替换独立的 [models.model1] 行（section 表头），不替换注释中的
    content = re.sub(
        r'\n(\[models\.model1\])\n',
        '\n[models.runninghub_xiaolv]' + RUNNINGHUB_BLOCK + r'\n\1\n',
        content,
        count=1
    )

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print("config.toml 已修复")

if __name__ == "__main__":
    main()
