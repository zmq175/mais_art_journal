"""RunningHub API 基础模块

共用逻辑：HTTP 请求、任务轮询、结果下载
API 文档：https://www.runninghub.cn/runninghub-api-doc-cn/
"""

import base64
import json
import random
import time
import urllib.request
from typing import Dict, Any, Tuple, Optional, List

from .base_client import BaseApiClient, logger


RUNNINGHUB_BASE = "https://www.runninghub.cn"


class BaseRunningHubClient(BaseApiClient):
    """RunningHub REST API 基础客户端"""

    def _build_opener(self):
        """构建 opener（支持代理）"""
        proxy_config = self._get_proxy_config()
        if proxy_config:
            proxy_handler = urllib.request.ProxyHandler({
                'http': proxy_config['http'],
                'https': proxy_config['https']
            })
            return urllib.request.build_opener(proxy_handler)
        return urllib.request.build_opener()

    def _http_post(
        self,
        path: str,
        body: dict,
        api_key: str,
        opener=None
    ) -> Tuple[Optional[dict], Optional[str]]:
        """发送 POST 请求，返回 (解析后的 data, 错误信息)"""
        if opener is None:
            opener = self._build_opener()
        url = f"{RUNNINGHUB_BASE}{path}"
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Host", "www.runninghub.cn")
        req.add_header("Authorization", f"Bearer {api_key}")

        try:
            with opener.open(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                code = data.get("code", -1)
                if code != 0:
                    msg = data.get("msg", str(data))
                    return None, f"RunningHub API 错误 (code={code}): {msg}"
                return data.get("data"), None
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
                err_data = json.loads(err_body)
                msg = err_data.get("msg", err_body[:200])
            except Exception:
                msg = str(e)
            return None, f"HTTP {e.code}: {msg}"
        except Exception as e:
            return None, str(e)

    def _query_task(self, task_id: str, api_key: str, opener=None) -> Tuple[Optional[dict], Optional[str]]:
        """查询任务状态与结果，POST /openapi/v2/query"""
        body = {"taskId": str(task_id)}
        if opener is None:
            opener = self._build_opener()
        url = f"{RUNNINGHUB_BASE}/openapi/v2/query"
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")

        try:
            with opener.open(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                # V2 直接返回任务信息，无 code 包裹
                if "status" in data:
                    return data, None
                if "code" in data and data.get("code") == 0 and "data" in data:
                    return data["data"], None
                return None, data.get("msg", str(data))
        except Exception as e:
            return None, str(e)

    def _poll_until_done(
        self,
        task_id: str,
        api_key: str,
        timeout: int = 300,
        interval: float = 2.0,
        opener=None
    ) -> Tuple[Optional[List[dict]], Optional[str]]:
        """轮询任务直到完成（用于 quick/ai-app，使用 /openapi/v2/query）"""
        start = time.time()
        while time.time() - start < timeout:
            info, err = self._query_task(task_id, api_key, opener)
            if err:
                return None, err
            status = info.get("status", "").upper()
            if status == "SUCCESS":
                results = info.get("results") or []
                return results, None
            if status == "FAILED":
                em = info.get("errorMessage") or info.get("failedReason", "任务失败")
                return None, em
            time.sleep(interval)
        return None, f"任务轮询超时 ({timeout}s)"

    def _poll_workflow_outputs(
        self,
        task_id: str,
        api_key: str,
        timeout: int = 300,
        interval: float = 2.0,
        opener=None
    ) -> Tuple[Optional[List[dict]], Optional[str]]:
        """轮询 ComfyUI 工作流结果，使用 /task/openapi/outputs"""
        if opener is None:
            opener = self._build_opener()
        url = f"{RUNNINGHUB_BASE}/task/openapi/outputs"
        body = {"apiKey": api_key, "taskId": str(task_id)}
        start = time.time()

        while time.time() - start < timeout:
            try:
                payload = json.dumps(body).encode("utf-8")
                req = urllib.request.Request(url, data=payload, method="POST")
                req.add_header("Content-Type", "application/json")
                req.add_header("Authorization", f"Bearer {api_key}")
                with opener.open(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                return None, str(e)

            code = data.get("code", -1)
            if code == 0:
                out = data.get("data")
                if isinstance(out, list) and len(out) > 0:
                    return out, None
                return None, "任务完成但无输出"
            if code == 804:  # APIKEY_TASK_IS_RUNNING
                time.sleep(interval)
                continue
            if code == 813:  # APIKEY_TASK_IS_QUEUED
                time.sleep(interval)
                continue
            msg = data.get("msg", str(code))
            fail = data.get("data") or {}
            if isinstance(fail, dict) and fail.get("failedReason"):
                msg = str(fail["failedReason"])[:300]
            return None, f"任务失败 ({msg})"

        return None, f"任务轮询超时 ({timeout}s)"

    def _download_image_b64(self, url: str, opener=None) -> Optional[str]:
        """从 URL 下载图片并返回 base64"""
        if opener is None:
            opener = self._build_opener()
        try:
            req = urllib.request.Request(url, method="GET")
            with opener.open(req, timeout=60) as resp:
                return base64.b64encode(resp.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"{self.log_prefix} (RunningHub) 下载图片失败: {e}")
            return None

    def _resolve_node_info_list(
        self,
        template: List[dict],
        prompt: str,
        model_config: Dict[str, Any],
        size: str,
        strength: float = None,
        input_image_base64: str = None,
        uploaded_image_path: Optional[str] = None
    ) -> List[dict]:
        """将 nodeInfoList 模板中的占位符替换为实际值"""
        try:
            w, h = size.lower().split("x")
            width, height = int(w), int(h)
        except Exception:
            width, height = 1024, 1024

        seed = model_config.get("seed", -1)
        if seed == -1:
            seed = random.randint(1, 10_000_000_000)

        full_prompt = prompt + model_config.get("custom_prompt_add", "")
        negative = model_config.get("negative_prompt_add", "")
        steps = int(model_config.get("num_inference_steps", 20))
        cfg = float(model_config.get("guidance_scale", 7))
        denoise = float(strength) if strength is not None else 0.7

        replacements = {
            "${prompt}": full_prompt,
            "${negative_prompt}": negative,
            "${seed}": str(seed),
            "${width}": str(width),
            "${height}": str(height),
            "${steps}": str(steps),
            "${cfg}": str(cfg),
            "${denoise}": str(denoise),
        }
        if uploaded_image_path:
            replacements["${image}"] = uploaded_image_path

        out = []
        for item in template:
            row = dict(item)
            val = row.get("fieldValue") or row.get("fieldData") or ""
            if isinstance(val, str):
                for k, v in replacements.items():
                    val = val.replace(k, str(v))
                row["fieldValue"] = val
            out.append(row)
        return out
