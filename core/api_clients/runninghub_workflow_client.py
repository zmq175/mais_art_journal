"""RunningHub 工作流 API 客户端

通过 POST /task/openapi/create 发起 ComfyUI 工作流任务。
需要 workflow_id 和 node_info_list 配置。

配置示例：
  format = "runninghub-workflow"
  api_key = "your-32-char-api-key"
  model = "1904136902449209346"   # 即 workflow_id
  node_info_list = [
    {"nodeId": "6", "fieldName": "text", "fieldValue": "${prompt}"},
    {"nodeId": "3", "fieldName": "seed", "fieldValue": "${seed}"},
    {"nodeId": "5", "fieldName": "width", "fieldValue": "${width}"},
    {"nodeId": "5", "fieldName": "height", "fieldValue": "${height}"}
  ]
  支持占位符: ${prompt}, ${negative_prompt}, ${seed}, ${width}, ${height}, ${steps}, ${cfg}
"""

from typing import Dict, Any, Tuple, Optional

from .runninghub_base import BaseRunningHubClient, logger


class RunningHubWorkflowClient(BaseRunningHubClient):
    """RunningHub 工作流 REST API 客户端"""

    format_name = "runninghub-workflow"

    def _make_request(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        size: str,
        strength: float = None,
        input_image_base64: str = None
    ) -> Tuple[bool, str]:
        """发起工作流任务并等待结果"""
        api_key = (model_config.get("api_key") or "").strip()
        if not api_key or api_key in ("YOUR_API_KEY_HERE", "xxxxxxxxxxxxxx"):
            return False, "RunningHub 工作流需要配置有效的 api_key"

        workflow_id = model_config.get("model") or model_config.get("workflow_id", "")
        if not workflow_id:
            return False, "RunningHub 工作流需要配置 model 或 workflow_id"

        node_info_list = model_config.get("node_info_list", [])
        if not node_info_list:
            return False, "RunningHub 工作流需要配置 node_info_list（从导出工作流 API 获取）"

        # 替换占位符（图生图需用户预先上传图片并填入 node_info_list）
        resolved_list = self._resolve_node_info_list(
            node_info_list, prompt, model_config, size, strength,
            input_image_base64, uploaded_image_path=None
        )

        body = {
            "apiKey": api_key,
            "workflowId": str(workflow_id),
            "nodeInfoList": resolved_list,
            "addMetadata": model_config.get("add_metadata", True),
        }
        webhook = model_config.get("webhook_url")
        if webhook:
            body["webhookUrl"] = webhook

        opener = self._build_opener()
        data, err = self._http_post("/task/openapi/create", body, api_key, opener)
        if err:
            logger.error(f"{self.log_prefix} (RunningHub-Workflow) 提交失败: {err}")
            return False, err

        task_id = None
        if isinstance(data, dict):
            task_id = data.get("taskId") or data.get("task_id")
        if not task_id:
            return False, "提交成功但未返回 taskId"

        logger.info(f"{self.log_prefix} (RunningHub-Workflow) 任务已提交, taskId={task_id}")

        results, poll_err = self._poll_workflow_outputs(str(task_id), api_key, timeout=300, opener=opener)
        if poll_err:
            return False, poll_err

        if not results:
            return False, "任务完成但无输出结果"

        first = results[0] if isinstance(results[0], dict) else {}
        img_url = first.get("url") or first.get("fileUrl")
        if not img_url:
            return False, "结果中无图片 URL"

        b64 = self._download_image_b64(img_url, opener)
        if not b64:
            return False, "下载图片失败"
        return True, b64
