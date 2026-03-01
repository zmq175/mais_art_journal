"""RunningHub AI 应用 API 客户端

通过 POST /task/openapi/ai-app/run 发起 AI 应用任务。
在 AI 应用详情页可查看 nodeInfoList 示例。

配置示例：
  format = "runninghub-ai-app"
  api_key = "your-32-char-api-key"
  model = "1877265245566922800"   # 可选，也可用 webapp_id
  webapp_id = "1877265245566922800"
  node_info_list = [
    {"nodeId": "122", "fieldName": "prompt", "fieldValue": "${prompt}"}
  ]
  支持占位符: ${prompt}, ${negative_prompt}, ${seed}, ${width}, ${height}, ${steps}, ${cfg}
"""

from typing import Dict, Any, Tuple

from .runninghub_base import BaseRunningHubClient, logger


class RunningHubAiAppClient(BaseRunningHubClient):
    """RunningHub AI 应用 API 客户端"""

    format_name = "runninghub-ai-app"

    def _make_request(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        size: str,
        strength: float = None,
        input_image_base64: str = None
    ) -> Tuple[bool, str]:
        """发起 AI 应用任务并等待结果"""
        api_key = (model_config.get("api_key") or "").strip()
        if not api_key or api_key in ("YOUR_API_KEY_HERE", "xxxxxxxxxxxxxx"):
            return False, "RunningHub AI 应用需要配置有效的 api_key"

        webapp_id = model_config.get("webapp_id") or model_config.get("model", "")
        if not webapp_id:
            return False, "RunningHub AI 应用需要配置 webapp_id 或 model"

        node_info_list = model_config.get("node_info_list", [])
        if not node_info_list:
            return False, "RunningHub AI 应用需要配置 node_info_list（从应用详情页 API 示例获取）"

        resolved_list = self._resolve_node_info_list(
            node_info_list, prompt, model_config, size, strength,
            input_image_base64, uploaded_image_path=None
        )

        body = {
            "apiKey": api_key,
            "webappId": str(webapp_id),
            "nodeInfoList": resolved_list,
        }
        instance_type = model_config.get("instance_type")
        if instance_type:
            body["instanceType"] = instance_type
        webhook = model_config.get("webhook_url")
        if webhook:
            body["webhookUrl"] = webhook

        opener = self._build_opener()
        data, err = self._http_post("/task/openapi/ai-app/run", body, api_key, opener)
        if err:
            logger.error(f"{self.log_prefix} (RunningHub-AiApp) 提交失败: {err}")
            return False, err

        task_id = None
        if isinstance(data, dict):
            task_id = data.get("taskId") or data.get("task_id")
        if not task_id:
            return False, "提交成功但未返回 taskId"

        logger.info(f"{self.log_prefix} (RunningHub-AiApp) 任务已提交, taskId={task_id}")

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
