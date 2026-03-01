"""RunningHub 快捷创作 API 客户端

通过 POST /task/openapi/quick-ai-app/run 调用标准模型
（如 Flux 文生图、Nano-Banana、Qwen 文生图、参考生图等）。

配置示例：
  format = "runninghub-quick"
  api_key = "your-32-char-api-key"
  model = "webapp_id"   # 可选，也可用 webapp_id
  webapp_id = "1957344152277151746"
  quick_create_code = "001"
  node_info_list = [
    {"nodeId": "45", "fieldName": "text", "fieldValue": "${prompt}"},
    {"nodeId": "101", "fieldName": "select", "fieldValue": "2", "description": "设置比例"}
  ]
  支持占位符: ${prompt}, ${negative_prompt}, ${seed}, ${width}, ${height}, ${steps}, ${cfg}
"""

from typing import Dict, Any, Tuple

from .runninghub_base import BaseRunningHubClient, logger


class RunningHubQuickClient(BaseRunningHubClient):
    """RunningHub 快捷创作 API 客户端"""

    format_name = "runninghub-quick"

    def _make_request(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        size: str,
        strength: float = None,
        input_image_base64: str = None
    ) -> Tuple[bool, str]:
        """发起快捷创作任务并等待结果"""
        api_key = (model_config.get("api_key") or "").strip()
        if not api_key or api_key in ("YOUR_API_KEY_HERE", "xxxxxxxxxxxxxx"):
            return False, "RunningHub 快捷创作需要配置有效的 api_key"

        webapp_id = model_config.get("webapp_id") or model_config.get("model", "")
        if not webapp_id:
            return False, "RunningHub 快捷创作需要配置 webapp_id 或 model"

        quick_create_code = model_config.get("quick_create_code", "001")
        node_info_list = model_config.get("node_info_list", [])
        if not node_info_list:
            return False, "RunningHub 快捷创作需要配置 node_info_list（从快捷创作页面 API 示例获取）"

        resolved_list = self._resolve_node_info_list(
            node_info_list, prompt, model_config, size, strength,
            input_image_base64, uploaded_image_path=None
        )

        body = {
            "apiKey": api_key,
            "webappId": str(webapp_id),
            "quickCreateCode": str(quick_create_code),
            "nodeInfoList": resolved_list,
        }

        opener = self._build_opener()
        data, err = self._http_post(
            "/task/openapi/quick-ai-app/run", body, api_key, opener
        )
        if err:
            logger.error(f"{self.log_prefix} (RunningHub-Quick) 提交失败: {err}")
            return False, err

        task_id = None
        if isinstance(data, dict):
            task_id = data.get("taskId") or data.get("task_id")
        if not task_id:
            return False, "提交成功但未返回 taskId"

        logger.info(f"{self.log_prefix} (RunningHub-Quick) 任务已提交, taskId={task_id}")

        results, poll_err = self._poll_until_done(str(task_id), api_key, timeout=300, opener=opener)
        if poll_err:
            return False, poll_err

        if not results:
            return False, "任务完成但无输出结果"

        img_url = results[0].get("url") if isinstance(results[0], dict) else None
        if not img_url:
            return False, "结果中无图片 URL"

        b64 = self._download_image_b64(img_url, opener)
        if not b64:
            return False, "下载图片失败"
        return True, b64
