#!/usr/bin/env python3
"""RunningHub 工作流简单测试 - 运行后删除，勿提交"""
import base64
import json
import sys
import time
import urllib.request

API_KEY = "YOUR_API_KEY"  # 测试时填入，勿提交
WORKFLOW_ID = "2002109398342352897"
BASE = "https://www.runninghub.cn"

# 人像提示词（可增删修改）
PROMPTS = [
    "一位年轻的东亚女生 cos 日系经典 JK 制服造型，留着柔顺乌黑的长直发，有轻薄的空气齐刘海，鬓角有细碎的发丝修饰脸型，妆容是清透的日系裸妆，浅棕色卧蚕，淡粉色唇釉，表情温柔恬静。她穿着藏青色领的白色水手服上衣，搭配浅灰色百褶格裙，黑色中筒棉袜，棕色小皮鞋，领口系着红色领结，手里抱着一本日系漫画书，侧身站在日式学校的木质走廊里，午后的阳光从窗户斜洒进来，地面有光影斑驳，背景是模糊的教室门窗和绿植，画面是高清真人写真质感，皮肤细腻通透，光影柔和自然，整体清新治愈。",
    "一位二十岁左右的亚洲女孩，黑色长发微卷披肩，淡妆，穿着米色针织开衫和白色连衣裙，坐在咖啡馆靠窗的位置，手里捧着拿铁，窗外是模糊的街景，自然光从侧面打来，柔和的逆光勾勒发丝，画面是电影感写真风格，氛围安静治愈。",
    "一位气质清新的东亚少女，扎着低马尾，穿着浅蓝色衬衫和白色百褶裙，站在图书馆的书架前，手里拿着一本书，暖黄色灯光从头顶洒下，背景是虚化的书籍，表情专注恬静，画面是日系胶片质感，肤色通透，光影自然。",
]


def post(path, body):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def run_one(prompt: str, out_name: str) -> bool:
    """跑单次生成，返回是否成功"""
    body = {
        "apiKey": API_KEY,
        "workflowId": WORKFLOW_ID,
        "nodeInfoList": [
            {"nodeId": "48", "fieldName": "编辑文本", "fieldValue": prompt},
            {"nodeId": "13", "fieldName": "width", "fieldValue": "1024"},
            {"nodeId": "13", "fieldName": "height", "fieldValue": "1520"},
        ],
        "addMetadata": True,
    }
    print(f"提交任务: {prompt[:40]}...")
    try:
        resp = post("/task/openapi/create", body)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        print(f"提交失败 HTTP {e.code}: {err[:500]}")
        return False
    except Exception as e:
        print(f"提交失败: {e}")
        return False

    if resp.get("code") != 0:
        print(f"API 错误: {resp.get('msg', resp)}")
        return False

    data = resp.get("data") or resp
    task_id = data.get("taskId") or data.get("task_id")
    if not task_id:
        print(f"未返回 taskId: {resp}")
        return False
    print(f"任务已提交, taskId={task_id}")

    print("轮询结果...")
    for i in range(150):
        time.sleep(2)
        try:
            q = post("/task/openapi/outputs", {"apiKey": API_KEY, "taskId": str(task_id)})
        except Exception as e:
            print(f"查询异常: {e}")
            continue
        code = q.get("code", -1)
        if code == 804 or code == 813:
            print(f"  [{i+1}] 排队/运行中...")
            continue
        if code == 805:
            print(f"任务失败: {q.get('msg')} {q.get('data', {})}")
            return False
        if code == 0:
            results = q.get("data") or []
            if results:
                url = results[0].get("fileUrl") or results[0].get("url")
                if url:
                    print("下载图片...")
                    with urllib.request.urlopen(url, timeout=60) as r:
                        img = r.read()
                    with open(out_name, "wb") as f:
                        f.write(img)
                    print(f"成功！图片已保存: {out_name}")
                    return True
            print("任务完成但无图片")
            return False
        print(f"  [{i+1}] 未知状态 code={code}")

    print("轮询超时")
    return False


def main():
    # 用法: python test_runninghub.py [0|1|2]  默认 0
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    idx = max(0, min(idx, len(PROMPTS) - 1))
    prompt = PROMPTS[idx]
    out = f"test_output_{idx + 1}.png"
    ok = run_one(prompt, out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
