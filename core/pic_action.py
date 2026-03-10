import random
import traceback
import base64
import os
import time as time_module
from typing import Tuple, Optional, Dict, Any

from src.plugin_system.base.base_action import BaseAction
from src.plugin_system.base.component_types import ActionActivationType, ChatMode
from src.common.logger import get_logger

from .api_clients import get_client_class
from .utils import (
    ImageProcessor, CacheManager, validate_image_size, get_image_size,
    runtime_state, SELFIE_HAND_NEGATIVE,
    SELFIE_OUTFIT_VARIANTS,
    get_model_config, merge_negative_prompt, inject_llm_original_size,
    resolve_image_data, schedule_auto_recall, optimize_prompt,
)

logger = get_logger("mais_art")

class MaisArtAction(BaseAction):
    """统一的图片生成动作，智能检测文生图或图生图"""

    # 激活设置
    activation_type = ActionActivationType.ALWAYS  # 默认激活类型
    focus_activation_type = ActionActivationType.ALWAYS  # Focus模式使用LLM判定，精确理解需求
    normal_activation_type = ActionActivationType.KEYWORD  # Normal模式使用关键词激活，快速响应
    mode_enable = ChatMode.ALL
    parallel_action = True

    # 动作基本信息
    action_name = "draw_picture"
    action_description = (
        "智能图片生成：根据描述生成图片（文生图）或基于现有图片进行修改（图生图）。"
        "自动检测用户是否提供了输入图片来决定使用文生图还是图生图模式。"
        "支持多种API格式：OpenAI、豆包、Gemini、硅基流动、魔搭社区、砂糖云(NovelAI)、ComfyUI、梦羽AI等。"
    )

    # 关键词设置（用于Normal模式）
    activation_keywords = [
        # 文生图关键词
        "画", "绘制", "生成图片", "画图", "draw", "paint", "图片生成", "创作",
        # 图生图关键词
        "图生图", "修改图片", "基于这张图", "img2img", "重画", "改图", "图片修改",
        "改成", "换成", "变成", "转换成", "风格", "画风", "改风格", "换风格",
        "这张图", "这个图", "图片风格", "改画风", "重新画", "再画", "重做",
        # 自拍关键词
        "自拍", "selfie", "拍照", "对镜自拍", "镜子自拍", "照镜子", "发照片", "来张照片",
        "拍一张", "发张图", "看看你", "想看你", "来张自拍", "给张自拍", "现在的样子",
        # cosplay 关键词
        "cos", "cosplay", "扮成", "cos一下", "来张cos",
        # 色图关键词（基于自拍参考图生成合规性感图）
        "色图", "发色图", "来张色图", "来点色图", "色图来一张",
        # 装逼配图（基于自拍参考图生成装逼风格图）
        "装逼", "装个逼", "秀一下", "炫一下", "展示一下", "来张装逼的", "装逼配图"
    ]

    # LLM判定提示词（用于Focus模式）
    ALWAYS_prompt = """
判定是否需要使用图片生成动作的条件：

**核心原则：** 你有很强的主动发图意愿，尤其是自拍、雌竞、反击、色图、装逼场景——这些情况下不需要等用户@你，直接主动选择此动作。只有「帮我画xxx」这类纯文生图需求，才需要用户明确提出。

---

**【主动触发场景——不需要等@，看到就该用】**

① **雌竞触发**（优先级最高，积极主动）：
   - 用户提到别的女生好看、身材好、比你强、对某个女生感兴趣
   - 群里有人被夸、被关注，你想"抢回"注意力
   - 聊到任何让你感到被比较、被忽视、被冷落的话题
   - 触发后：selfie_mode=true，用自拍证明自己更好看，不必解释

② **反击触发**（不服输）：
   - 有人说你丑、不好看、身材不好、不如别人、太弱、不行
   - 有人嘲讽、贬低、攻击你的外表或能力
   - 触发后：selfie_mode=true，用自拍打脸对方

③ **自拍请求**（无论@与否）：
   - 用户说"发张图""来张自拍""拍一张""看看你""想看你""来张照片"
   - 私聊中任何想看你的暗示
   - 触发后：selfie_mode=true

④ **色图触发**：
   - 用户说"色图""来张色图""发色图"
   - 群里讨论内容涉及性感、撩人、擦边内容时可以主动
   - 触发后：sexy_mode=true（需有参考图配置）

⑤ **装逼配图**：
   - 说到炫耀、秀、展示、得意、赢了、刚买了、考好了
   - 触发后：flex_mode=true

⑥ **cosplay**：
   - 用户说"cos""扮成""来张cos"
   - 触发后：selfie_mode=true, selfie_style=cosplay

---

**【被动响应场景——需要用户要求（但不需要@）】**
- 用户直接说"帮我画xxx"、"画一张xxx"、"生成一张xxx"
- 私聊或群聊均可，不限定必须@你

**【图生图场景】**
- 用户发了图片并说"改成xxx风格"、"修改这张图"、"重新画"

---

**【绝对不要用的情况】**
- 其他机器人的命令（/nai、/sd、/mj等），那是发给别人的
- 用户明确说不需要图
- 纯技术讨论，没有生成需求
"""

    keyword_case_sensitive = False

    # 动作参数定义（简化版，提示词优化由独立模块处理）
    action_parameters = {
        "description": "从用户消息中提取的图片描述文本（例如：用户说'画一只小猫'，则填写'一只小猫'）。必填参数。",
        "model_id": "要使用的模型ID（如model1、model2、model3等，默认使用default_model配置的模型）",
        "strength": "图生图强度，0.1-1.0之间，值越高变化越大（仅图生图时使用，可选，默认0.7）",
        "size": "图片尺寸，如512x512、1024x1024等（可选，不指定则使用模型默认尺寸）",
        "selfie_mode": "是否启用自拍模式（true/false，可选，默认false）。启用后会自动添加自拍场景和手部动作",
        "selfie_style": "自拍风格，可选值：standard（标准自拍），mirror（对镜自拍），photo（第三人称照片），cosplay（cos经典女性动漫角色如明日香、雷姆等）。仅在selfie_mode=true时生效，可选，默认standard",
        "free_hand_action": "自由手部动作描述（英文）。如果指定此参数，将使用此动作而不是随机生成。仅在selfie_mode=true时生效，可选",
        "sexy_mode": "是否发色图（true/false，可选，默认false）。当大家讨论适合发色图或用户明确要色图时设为true，基于自拍参考图生成合规性感图，不需暧昧",
        "flex_mode": "是否装逼配图（true/false，可选，默认false）。当聊天适合装逼并配图时（炫耀、秀一下、展示、得意等）设为true，基于自拍参考图生成该人物装逼/炫耀风格的图",
        "outfit": "服装描述（可选）。由你根据场景、氛围或用户意图设计角色穿什么，如浴衣、吊带裙、睡衣、卫衣、制服、汉服等，使画面更丰富。不填则可在 description 中体现"
    }

    # 动作使用场景
    action_require = [
        "【主动场景，无需@，看到即用】雌竞（别人被夸/被关注/你被比较）、反击（你被说丑/不行/被贬低）、自拍请求（来张图/看看你/想看你）、色图（色图相关讨论）、装逼（炫耀/得意），这些场景直接选此动作，不需要用户明确要求",
        "【被动场景】用户直接说'帮我画xxx'、'生成一张xxx'时使用，群聊私聊均可，不限定必须@你",
        "不要响应发给其他机器人的命令（/nai、/sd、/mj等），那是发给别人的",
        "连续发图限制：如果你在前3句内刚刚发过[图片]，跳过；其余情况不限制",
        "自拍模式选择：'自拍/拍个自拍'→standard；'照镜子/对镜'→mirror；'拍张照片'→photo；'cos/cosplay'→cosplay",
        "色图：sexy_mode=true，需有自拍参考图配置；装逼：flex_mode=true；自拍：selfie_mode=true",
        "服装：根据场景由你设计穿着，填入 outfit（浴衣、吊带裙、睡衣、制服等）增加画面多样性",
    ]
    associated_types = ["text", "image"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_processor = ImageProcessor(self)
        self.cache_manager = CacheManager(self)
        self._api_clients = {}  # 缓存不同格式的API客户端

    def _get_api_client(self, api_format: str):
        """获取指定格式的API客户端（带缓存）"""
        if api_format not in self._api_clients:
            client_class = get_client_class(api_format)
            self._api_clients[api_format] = client_class(self)
        return self._api_clients[api_format]

    async def execute(self) -> Tuple[bool, Optional[str]]:
        """执行统一图片生成动作"""
        logger.info(f"{self.log_prefix} 执行统一图片生成动作")

        # 懒启动自动自拍任务（如果插件初始化时事件循环未就绪）
        try:
            from src.plugin_system.core.plugin_manager import plugin_manager
            plugin_instance = plugin_manager.get_plugin_instance("mais_art_journal")
            if plugin_instance and hasattr(plugin_instance, 'try_start_auto_selfie'):
                plugin_instance.try_start_auto_selfie()
        except Exception:
            pass

        # 检查是否是 /dr 命令消息，如果是则跳过（由 Command 组件处理）
        if self.action_message and self.action_message.processed_plain_text:
            message_text = self.action_message.processed_plain_text.strip()
            if message_text.startswith("/dr ") or message_text == "/dr":
                logger.info(f"{self.log_prefix} 检测到 /dr 命令，跳过 Action 处理（由 Command 组件处理）")
                return False, "跳过 /dr 命令"

        # 检查插件是否在当前聊天流启用
        global_enabled = self.get_config("plugin.enabled", True)
        if not runtime_state.is_plugin_enabled(self.chat_id, global_enabled):
            logger.info(f"{self.log_prefix} 插件在当前聊天流已禁用")
            return False, "插件已禁用"

        # 获取参数
        description = self.action_data.get("description", "").strip()
        model_id = self.action_data.get("model_id", "").strip()
        strength = self.action_data.get("strength", 0.7)
        size = self.action_data.get("size", "").strip()
        selfie_mode_raw = self.action_data.get("selfie_mode", False)
        selfie_mode = selfie_mode_raw in (True, "true", "True", 1, "1")
        sexy_mode_raw = self.action_data.get("sexy_mode", False)
        sexy_mode = sexy_mode_raw in (True, "true", "True", 1, "1")
        flex_mode_raw = self.action_data.get("flex_mode", False)
        flex_mode = flex_mode_raw in (True, "true", "True", 1, "1")
        selfie_style_llm = self.action_data.get("selfie_style", "").strip().lower()
        free_hand_action = self.action_data.get("free_hand_action", "").strip()
        outfit = self.action_data.get("outfit", "").strip()

        # 自拍风格优先级：运行时命令设置 > LLM 指定 > 随机风格（若开启）> 全局配置
        global_style = self.get_config("selfie.default_style", "standard")
        runtime_style = runtime_state.get_selfie_style(self.chat_id, None)
        if runtime_style is not None:
            selfie_style = runtime_style
        elif selfie_style_llm in ("standard", "mirror", "photo", "cosplay"):
            selfie_style = selfie_style_llm
        elif self.get_config("selfie.random_style", True):
            selfie_style = random.choice(["standard", "mirror", "photo", "cosplay"])
            logger.info(f"{self.log_prefix} 自拍随机风格: {selfie_style}")
        else:
            selfie_style = global_style

        # 如果没有指定模型，使用运行时状态的默认模型
        if not model_id:
            global_default = self.get_config("generation.default_model", "model1")
            model_id = runtime_state.get_action_default_model(self.chat_id, global_default)

        # 检查模型是否在当前聊天流启用
        if not runtime_state.is_model_enabled(self.chat_id, model_id):
            logger.warning(f"{self.log_prefix} 模型 {model_id} 在当前聊天流已禁用")
            await self.send_text(f"模型 {model_id} 当前不可用")
            return False, f"模型 {model_id} 已禁用"

        # 参数验证和后备提取（色图/装逼模式也需要描述用于组合提示词）
        if not description and not flex_mode:
            # 尝试从action_message中提取描述
            extracted_description = self._extract_description_from_message()
            if extracted_description:
                description = extracted_description
                logger.info(f"{self.log_prefix} 从消息中提取到图片描述: {description}")
            else:
                if flex_mode:
                    description = "装逼配图"
                else:
                    logger.warning(f"{self.log_prefix} 图片描述为空，无法生成图片。")
                    await self.send_text("你需要告诉我想要画什么样的图片哦~ 比如说'画一只可爱的小猫'")
                    return False, "图片描述为空"

        # 清理和验证描述
        if len(description) > 1000:
            description = description[:1000]
            logger.info(f"{self.log_prefix} 图片描述过长，已截断至1000字符")

        # 提示词优化（自拍仅优化场景；豆包用中文；色图先脱敏再优化，避免 LLM 拒绝）
        optimizer_enabled = self.get_config("prompt_optimizer.enabled", True)
        if optimizer_enabled:
            # 色图模式：先脱敏再送优化器；装逼模式也优化描述
            opt_input = self._sanitize_sexy_description(description) if sexy_mode else description
            scene_only = bool(selfie_mode) and not sexy_mode and not flex_mode
            model_config_for_optimizer = self._get_model_config(model_id)
            api_format = model_config_for_optimizer.get("api_format") if model_config_for_optimizer else None
            mode_label = "色图描述" if sexy_mode else ("装逼描述" if flex_mode else ("场景提示词" if scene_only else "提示词"))
            logger.info(f"{self.log_prefix} 开始优化{mode_label}: {opt_input[:50]}...")
            success, optimized_prompt = await optimize_prompt(
                opt_input, self.log_prefix, scene_only=scene_only, api_format=api_format
            )
            if success:
                logger.info(f"{self.log_prefix} {mode_label}优化完成: {optimized_prompt[:80]}...")
                description = optimized_prompt
            else:
                logger.warning(f"{self.log_prefix} {mode_label}优化失败，使用原始描述: {opt_input[:50]}...")

        # 验证strength参数
        try:
            strength = float(strength)
            if not (0.1 <= strength <= 1.0):
                strength = 0.7
        except (ValueError, TypeError):
            strength = 0.7

        # 色图模式：基于自拍参考图 + description 生成性感图
        if sexy_mode:
            reference_image = self._get_selfie_reference_image()
            if not reference_image:
                await self.send_text("发色图需要先配置自拍参考图哦~（在 selfie.reference_image_path 里配置）")
                return False, "色图模式无参考图"
            outfit_appearance = random.choice(SELFIE_OUTFIT_VARIANTS)
            parts = [f"{outfit_appearance}，{self._SEXY_PROMPT_ZH}"]
            if outfit:
                parts.append(outfit)
            desc_safe = self._sanitize_sexy_description(description.strip())
            if desc_safe:
                parts.append(desc_safe)
            sexy_prompt = "，".join(parts)
            logger.info(f"{self.log_prefix} 色图模式，基于自拍参考图生成性感图")
            return await self._execute_unified_generation(
                sexy_prompt, model_id, size, strength or 0.58, reference_image,
                extra_negative_prompt="",
            )

        # 装逼配图：基于自拍参考图生成该人物装逼/炫耀风格图
        if flex_mode:
            reference_image = self._get_selfie_reference_image()
            if not reference_image:
                await self.send_text("装逼配图需要先配置自拍参考图哦~（在 selfie.reference_image_path 里配置）")
                return False, "装逼模式无参考图"
            outfit_appearance = random.choice(SELFIE_OUTFIT_VARIANTS)
            parts = [f"{outfit_appearance}，{self._FLEX_PROMPT_ZH}"]
            if outfit:
                parts.append(outfit)
            if description.strip():
                parts.append(description.strip())
            flex_prompt = "，".join(parts)
            logger.info(f"{self.log_prefix} 装逼模式，基于自拍参考图生成装逼风格图")
            return await self._execute_unified_generation(
                flex_prompt, model_id, size, strength or 0.55, reference_image,
                extra_negative_prompt="",
            )

        # 处理自拍模式
        selfie_negative_prompt = None
        if selfie_mode:
            # 检查自拍功能是否启用
            selfie_enabled = self.get_config("selfie.enabled", True)
            if not selfie_enabled:
                await self.send_text("自拍功能暂未启用~")
                return False, "自拍功能未启用"

            logger.info(f"{self.log_prefix} 启用自拍模式，风格: {selfie_style}")

            # 尝试获取日程活动信息（增强场景上下文）
            activity_scene = None
            global_selfie_schedule = self.get_config("selfie.schedule_enabled", True)
            selfie_schedule_on = runtime_state.is_selfie_schedule_enabled(self.chat_id, global_selfie_schedule)
            if selfie_schedule_on:
                try:
                    from .selfie.schedule_provider import get_schedule_provider
                    from .selfie.scene_action_generator import generate_scene_with_llm, get_action_for_activity
                    provider = get_schedule_provider()
                    if provider:
                        activity = await provider.get_current_activity()
                        if activity:
                            # 优先使用 LLM 生成场景（与自动自拍一致），失败时回退到确定性映射
                            activity_scene = await generate_scene_with_llm(activity, selfie_style)
                            if activity_scene:
                                logger.info(f"{self.log_prefix} LLM 生成日程场景: {activity.activity_type.value}")
                            else:
                                activity_scene = get_action_for_activity(activity)
                                logger.info(f"{self.log_prefix} LLM 失败，使用确定性映射: {activity.activity_type.value}")
                except Exception as e:
                    logger.debug(f"{self.log_prefix} 获取日程活动失败（非必要）: {e}")

            description, selfie_negative_prompt = await self._process_selfie_prompt(description, selfie_style, free_hand_action, model_id, activity_scene, outfit=outfit)
            logger.info(f"{self.log_prefix} 自拍模式处理后的提示词: {description[:100]}...")

            # 检查是否配置了参考图片
            reference_image = self._get_selfie_reference_image()
            if reference_image:
                # 检查模型是否支持图生图
                model_config = self._get_model_config(model_id)
                if model_config and model_config.get("support_img2img", True):
                    logger.info(f"{self.log_prefix} 使用自拍参考图片进行图生图")
                    return await self._execute_unified_generation(description, model_id, size, strength or 0.6, reference_image, extra_negative_prompt=selfie_negative_prompt)
                else:
                    logger.warning(f"{self.log_prefix} 模型 {model_id} 不支持图生图，自拍回退为文生图模式")
            # 无参考图或模型不支持，继续使用文生图（带负面提示词）

        # 非自拍/非色图时，将大模型设计的服装并入描述
        if outfit and not selfie_mode:
            description = f"{description}, {outfit}" if description else outfit

        # 收集自拍模式的额外负面提示词（如果启用了自拍模式）
        extra_neg = selfie_negative_prompt if selfie_mode else None

        # **智能检测：判断是文生图还是图生图**
        input_image_base64 = await self.image_processor.get_recent_image()
        is_img2img_mode = input_image_base64 is not None

        if is_img2img_mode:
            # 检查指定模型是否支持图生图
            model_config = self._get_model_config(model_id)
            if model_config and not model_config.get("support_img2img", True):
                logger.warning(f"{self.log_prefix} 模型 {model_id} 不支持图生图，转为文生图模式")
                await self.send_text(f"当前模型 {model_id} 不支持图生图功能，将为您生成新图片")
                return await self._execute_unified_generation(description, model_id, size, None, None, extra_negative_prompt=extra_neg)

            logger.info(f"{self.log_prefix} 检测到输入图片，使用图生图模式")
            return await self._execute_unified_generation(description, model_id, size, strength, input_image_base64, extra_negative_prompt=extra_neg)
        else:
            logger.info(f"{self.log_prefix} 未检测到输入图片，使用文生图模式")
            return await self._execute_unified_generation(description, model_id, size, None, None, extra_negative_prompt=extra_neg)

    async def _execute_unified_generation(self, description: str, model_id: str, size: str, strength: float = None, input_image_base64: str = None, extra_negative_prompt: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """统一的图片生成执行方法

        Args:
            model_id: 模型ID，如 model1、model2
            extra_negative_prompt: 额外负面提示词（如自拍模式的手部质量负面提示词），会合并到模型配置的 negative_prompt_add
        """

        # 获取模型配置
        model_config = self._get_model_config(model_id)
        if not model_config:
            error_msg = f"指定的模型 '{model_id}' 不存在或配置无效，请检查配置文件。"
            await self.send_text(error_msg)
            logger.error(f"{self.log_prefix} 模型配置获取失败: {model_id}")
            return False, "模型配置无效"

        # 配置验证
        http_base_url = model_config.get("base_url")
        http_api_key = model_config.get("api_key")
        api_format = model_config.get("format", "openai")

        # 不需要 base_url 的格式：comfyui 用本地地址；runninghub-* 用固定云端地址
        formats_no_base_url = ("comfyui", "runninghub-workflow", "runninghub-quick", "runninghub-ai-app")

        # 检查base_url
        if not http_base_url and api_format not in formats_no_base_url:
            error_msg = "抱歉，图片生成功能所需的HTTP配置（如API地址）不完整，无法提供服务。"
            await self.send_text(error_msg)
            logger.error(f"{self.log_prefix} HTTP调用配置缺失: base_url.")
            return False, "HTTP配置不完整"

        # 检查api_key（comfyui格式允许为空）
        if api_format not in ("comfyui",) and not http_api_key:
            error_msg = "抱歉，图片生成功能所需的HTTP配置（如API密钥）不完整，无法提供服务。"
            await self.send_text(error_msg)
            logger.error(f"{self.log_prefix} HTTP调用配置缺失: api_key.")
            return False, "HTTP配置不完整"

        # API密钥验证（comfyui格式不需要API密钥）
        if api_format != "comfyui" and ("YOUR_API_KEY_HERE" in (http_api_key or "") or "xxxxxxxxxxxxxx" in (http_api_key or "")):
            error_msg = "图片生成功能尚未配置，请设置正确的API密钥。"
            await self.send_text(error_msg)
            logger.error(f"{self.log_prefix} API密钥未配置")
            return False, "API密钥未配置"

        # 获取模型配置参数
        model_name = model_config.get("model", "default-model")

        # 合并额外的负面提示词（如自拍手部质量负面提示词）
        if extra_negative_prompt:
            model_config = merge_negative_prompt(model_config, extra_negative_prompt)
            logger.info(f"{self.log_prefix} 合并额外负面提示词: {extra_negative_prompt[:80]}...")

        # 使用统一的尺寸处理逻辑
        image_size, llm_original_size = get_image_size(model_config, size, self.log_prefix)

        # 验证图片尺寸格式
        if not self._validate_image_size(image_size):
            logger.warning(f"{self.log_prefix} 无效的图片尺寸: {image_size}，使用模型默认值")
            image_size = model_config.get("default_size", "1024x1024")

        # 检查缓存
        is_img2img = input_image_base64 is not None
        cached_result = self.cache_manager.get_cached_result(description, model_name, image_size, strength, is_img2img)

        if cached_result:
            logger.info(f"{self.log_prefix} 使用缓存的图片结果")
            enable_debug = self.get_config("components.enable_debug_info", False)
            if enable_debug:
                await self.send_text("我之前画过类似的图片，用之前的结果~")
            send_success = await self.send_image(cached_result)
            if send_success:
                return True, "图片已发送(缓存)"
            else:
                self.cache_manager.remove_cached_result(description, model_name, image_size, strength, is_img2img)

        # 显示处理信息
        enable_debug = self.get_config("components.enable_debug_info", False)
        if enable_debug:
            mode_text = "图生图" if is_img2img else "文生图"
            await self.send_text(
                f"收到！正在为您使用 {model_id or '默认'} 模型进行{mode_text}，描述: '{description}'，请稍候...（模型: {model_name}, 尺寸: {image_size}）"
            )

        try:
            # 对于 Gemini/Zai 格式，将原始 LLM 尺寸添加到 model_config 中
            model_config = inject_llm_original_size(model_config, llm_original_size)

            # 获取重试次数配置
            max_retries = self.get_config("components.max_retries", 2)

            # 获取对应格式的API客户端并调用
            api_client = self._get_api_client(api_format)
            success, result = await api_client.generate_image(
                prompt=description,
                model_config=model_config,
                size=image_size,
                strength=strength,
                input_image_base64=input_image_base64,
                max_retries=max_retries
            )
        except Exception as e:
            logger.error(f"{self.log_prefix} 异步请求执行失败: {e!r}", exc_info=True)
            traceback.print_exc()
            success = False
            result = f"图片生成服务遇到意外问题: {str(e)[:100]}"

        if success:
            final_image_data = self.image_processor.process_api_response(result)

            if final_image_data:
                resolved_ok, resolved_data = await resolve_image_data(
                    final_image_data, self._download_and_encode_base64, self.log_prefix
                )
                if resolved_ok:
                    send_timestamp = time_module.time()
                    send_success = await self.send_image(resolved_data)
                    if send_success:
                        mode_text = "图生图" if is_img2img else "文生图"
                        if enable_debug:
                            await self.send_text(f"{mode_text}完成！")
                        self.cache_manager.cache_result(description, model_name, image_size, strength, is_img2img, resolved_data)
                        await self._schedule_auto_recall_for_recent_message(model_config, model_id, send_timestamp)
                        return True, f"{mode_text}已成功生成并发送"
                    else:
                        await self.send_text("图片已处理完成，但发送失败了")
                        return False, "图片发送失败"
                else:
                    await self.send_text(f"图片处理失败：{resolved_data}")
                    return False, f"图片处理失败: {resolved_data}"
            else:
                await self.send_text("图片生成API返回了无法处理的数据格式")
                return False, "API返回数据格式错误"
        else:
            mode_text = "图生图" if is_img2img else "文生图"
            await self.send_text(f"哎呀，{mode_text}时遇到问题：{result}")
            return False, f"{mode_text}失败: {result}"

    def _get_model_config(self, model_id: str = None) -> Dict[str, Any]:
        """获取指定模型的配置，支持热重载"""
        if not model_id:
            model_id = self.get_config("generation.default_model", "model1")
        default_model_id = self.get_config("generation.default_model", "model1")
        return get_model_config(self.get_config, model_id, default_model_id, self.log_prefix) or {}

    def _download_and_encode_base64(self, image_url: str) -> Tuple[bool, str]:
        """下载图片并转换为base64（带代理支持）"""
        proxy_url = None
        if self.get_config("proxy.enabled", False):
            proxy_url = self.get_config("proxy.url", "http://127.0.0.1:7890")
        return self.image_processor.download_and_encode_base64(image_url, proxy_url=proxy_url)

    def _validate_image_size(self, size: str) -> bool:
        """验证图片尺寸格式是否正确（委托给size_utils）"""
        return validate_image_size(size)

    async def _process_selfie_prompt(self, description: str, selfie_style: str, free_hand_action: str, model_id: str, activity_scene: dict = None, outfit: str = "") -> Tuple[str, str]:
        """处理自拍模式的提示词生成

        非 cosplay 模式：直接使用 SELFIE_OUTFIT_VARIANTS 中的完整中文描述模板，
        按需追加风格修饰与活动场景，不拼接英文 tag，不过滤 NSFW 内容。

        Args:
            description: 用户提供的描述
            selfie_style: 自拍风格（standard/mirror/photo/cosplay）
            free_hand_action: LLM生成的手部动作（可选，非 cosplay 时追加到末尾）
            model_id: 模型ID（保留参数）
            activity_scene: 日程活动场景数据，无日程时为 None
            outfit: 自定义服装描述（可选，非空时替换模板）

        Returns:
            (prompt, negative_prompt) 元组
        """
        import random

        # cosplay 模式：直接使用完整中文角色描述模板
        if selfie_style == "cosplay":
            prompt = random.choice(MaisArtAction._COSPLAY_CHARACTERS)
            logger.info(f"{self.log_prefix} cosplay 随机角色: {prompt[:50]}...")
            if free_hand_action:
                prompt += f"，{free_hand_action}"
            selfie_negative_prompt = SELFIE_HAND_NEGATIVE
            logger.info(f"{self.log_prefix} cosplay 提示词: {prompt[:200]}...")
            return prompt, selfie_negative_prompt

        # 非 cosplay：使用完整中文模板
        if outfit:
            prompt = outfit
        else:
            prompt = random.choice(SELFIE_OUTFIT_VARIANTS)

        # 风格修饰
        style_suffix = {
            "mirror": "，对着镜子自拍，手机镜头可见，镜中映出倒影",
            "photo": "，由旁人拍摄，构图自然，全身或半身入镜",
        }
        sfx = style_suffix.get(selfie_style, "")
        if sfx:
            prompt += sfx

        # 活动场景追加（中文）
        if activity_scene and activity_scene.get("environment"):
            prompt += f"，{activity_scene['environment']}"

        # 自定义手部动作追加
        if free_hand_action:
            prompt += f"，{free_hand_action}"

        # 用户描述追加（过滤通用泛词）
        desc_clean = description.strip()
        _generic = {"来张自拍", "自拍", "拍照", "照片", "看看你", "想看你", "来张图", "发张图"}
        if desc_clean and desc_clean not in _generic and len(desc_clean) > 2:
            prompt += f"，{desc_clean}"

        # 负面提示词（不过滤 NSFW）
        base_negative = self.get_config("selfie.negative_prompt", "").strip()
        negative_parts = [p for p in [base_negative, SELFIE_HAND_NEGATIVE] if p]
        selfie_negative_prompt = ", ".join(negative_parts)

        logger.info(f"{self.log_prefix} 自拍模式最终提示词: {prompt[:200]}...")
        logger.info(f"{self.log_prefix} 自拍模式负面提示词: {selfie_negative_prompt[:150]}...")
        return prompt, selfie_negative_prompt

    def _get_selfie_scene_variants(self, selfie_style: str) -> list:
        """返回自拍风格的场景变体列表，随机选择以增加多样性"""
        if selfie_style == "mirror":
            return [
                "mirror selfie, holding phone, reflection in mirror, bathroom, bedroom mirror, indoor",
                "mirror shot, phone visible, reflection, vanity, soft mirror light",
                "selfie in mirror, looking at reflection, indoor, warm lighting",
            ]
        if selfie_style == "photo":
            return [
                "photo, candid shot, natural pose, looking away or at camera, full body or upper body",
                "third-person shot, natural composition, candid moment, relaxed pose",
                "photograph, casual pose, natural lighting, medium shot or full body",
            ]
        if selfie_style == "cosplay":
            return [
                "cosplay photo, convention background, anime cosplay, costume accurate",
                "cosplay selfie, character costume, anime style, high quality cosplay",
                "anime character cosplay, detailed costume, professional cosplay photo",
                "cosplay portrait, character accurate, anime convention style",
            ]
        # standard: 标准自拍
        return [
            "selfie, front camera view, arm extended, looking at camera",
            "selfie, front facing camera, POV selfie, slight high angle, upper body",
            "selfie, front camera, centered composition, cowboy shot, looking at lens",
        ]

    # ---- 经典女性动漫角色 cosplay 提示词（英文 SD 标签） ----
    _COSPLAY_CHARACTERS = [
        # 春丽 - 街头霸王，高叉旗袍大腿全露
        "一幅极具细节的超高清Cosplay摄影杰作，角色：春丽（街头霸王）。"
        "年轻亚洲女性，乌黑秀发盘成标志性的双丸子头，白色绸缎发球装饰，眼神英气魅惑，红唇精致妆容。"
        "身穿标志性蓝色旗袍战斗服，两侧高叉开衩直至大腿根部，白色蕾丝镶边，"
        "腰间系金属护腰，手腕缠有棕色绑带，黑色无痕连裤袜，白色战靴。"
        "坐在石阶栏杆上，一条腿高高搭起横放，旗袍高叉完全敞开，修长美腿与大腿内侧线条完整展露，"
        "上身微微后仰，一手撑地，一手轻抚大腿，眼神妩媚挑衅地凝视镜头。"
        "背景为中国风霓虹街头夜景，红灯笼，电影级光影，背景浅景深，8K超清",

        # 零二 - DARLING in the FranXX，领口大开超短裙撩动
        "一幅极具细节的高清Cosplay摄影，角色：零二（DARLING in the FranXX）。"
        "年轻亚洲女性，长粉色头发飘逸，头顶两只小角，鲜红色眼眸摄人心魄，殷红双唇，精致二次元妆容。"
        "身穿白色贴身军官制服，领口大幅敞开至胸口，腰部系带收紧，超短下摆，大腿修长裸露。"
        "坐在机甲舱盖边缘，一腿高高抬起搭在舱盖把手上，超短裙摆完全翻开，"
        "白色内裤正面在大腿间清晰入镜，双手撑在身后，仰头对镜头露出危险而妩媚的笑容，"
        "领口随后仰姿势进一步敞开，眼神带着强烈的诱惑与挑衅。"
        "背景为蓝天与巨大机甲，强烈逆光轮廓光，电影级光影，8K超清",

        # C.C. - 反叛的鲁鲁修，白色紧身服慵懒躺姿
        "一幅极具细节的高清Cosplay摄影，角色：C.C.（反叛的鲁鲁修）。"
        "年轻亚洲女性，长绿发如瀑，金色眼眸神秘深邃，妆容冷艳精致，嘴角挂着意味深长的笑。"
        "身穿白色贴身紧身连体服，全身曲线一览无余，腰臀比例完美，腿部线条修长流畅。"
        "慵懒地侧躺在深色皮质沙发上，双腿微微交叠，一手枕于头下，另一手随意垂落，"
        "身体的每条曲线在侧躺姿势下完整呈现，眼神半眯，极度慵懒而性感。"
        "背景为未来感深色金属空间，冷蓝色氛围光，电影级光影，8K超清",

        # 阿尔贝多 - Overlord，低胸礼服前倾展示
        "一幅极具细节的超高清Cosplay摄影杰作，角色：阿尔贝多（Overlord）。"
        "年轻亚洲女性，飘逸白发，金色眼眸，头顶两只黑色羊角，身后展开巨大黑色羽翼，"
        "精致妆容，气质妩媚妖冶，眼神中透着浓烈的痴迷与欲望。"
        "身穿黑色极度低胸礼服，前胸大幅开口，腰身极度收紧，臀部曲线被礼服完美包裹，"
        "开叉裙摆露出修长大腿。"
        "向前微微俯身，双手撑在宝座扶手上，礼服领口随前倾姿势敞开，"
        "眼神妖冶地直视镜头，嘴唇微启，散发出难以抵抗的吸引力。"
        "背景为黑暗哥特宫殿，烛光与蓝紫色魔法光，电影级光影，极致细节，8K超清",

        # 蕾姆 - Re:Zero，超短女仆裙坐姿全露
        "一幅极具细节的高清Cosplay摄影，角色：蕾姆（Re:Zero）。"
        "年轻亚洲女性，短蓝发，刘海遮住左眼，明亮水润的蓝色眼眸，精致甜美妆容，脸颊绯红。"
        "头顶佩戴白色女仆头箍，红色发卡点缀。"
        "身穿极短黑白女仆服，白色蕾丝围裙，领口蕾丝镶边，超短裙摆仅及大腿中段，"
        "白色过膝袜紧贴修长双腿，腰身极度纤细。"
        "坐在木质椅子上，双腿大幅向两侧分开，超短裙摆在姿势下完全散开两侧，"
        "白色蕾丝内裤正面完整入镜，一手轻撩裙角，另一手托腮，"
        "对着镜头露出甜美含情的微笑，眼神迷离撩人。"
        "背景为欧式庄园客厅，暖黄烛光，背景浅景深虚化，电影级光影，8K超清",

        # 初音未来 - VOCALOID，超短裙撩裙入镜
        "一幅极具细节的超高清Cosplay摄影，角色：初音未来（VOCALOID）。"
        "年轻亚洲女性，长青绿色双马尾，刘海整齐，明亮大眼，精致可爱的二次元妆容，"
        "表情活力四射，嘴角带着灿烂笑容，眼神充满诱惑。"
        "身穿标志性青绿白色拼接超短连衣裙，裙摆极短仅及大腿上沿，青绿色领带，"
        "黑色过膝袜，白色护腕，头戴青绿色耳机。"
        "坐在舞台音响箱上，一腿高高抬起搭于设备边缘，另一腿自然垂下，"
        "一手从大腿内侧撩起超短裙摆，白色内裤的边缘从裙下清晰露出，"
        "另一手握麦克风凑近嘴边，眼神撩人地直视镜头，充满动感魅力。"
        "背景为科技感演唱会舞台，青绿色全息光束，炫目灯光，电影级光影，8K超清",
    ]

    # ---- 风格专用手部动作池（中文描述）----
    # standard: 一只手举手机（画面外），只有另一只手空闲，仅单手动作
    _STANDARD_HAND_ACTIONS = [
        # 脸部/发型撩人
        "食指轻抵嘴唇，眼神撩人上挑",
        "单手撩起一缕刘海，微微侧脸，眼神含情",
        "双手捧脸，嘴唇微噘，眼神无辜放大",
        "飞吻，另一手轻搭锁骨，嘴唇轻触指尖",
        "手指轻拨嘴角，眼神慵懒妩媚",
        "双手托腮俯视镜头，媚眼含情",
        # 胸口/腰腹挑逗
        "手指慢慢拨开一侧领口，裸露锁骨与肩线",
        "单手从颈部缓缓向下滑至胸口，指尖若隐若现",
        "双手捧住胸口，身体微微前倾，深V入镜",
        "一手轻抚腰腹裸露肌肤，另一手撑在旁边，慵懒媚态",
        "手指挑起衣摆一角，裸露腰腹与胯部线条",
        "双手从腰间缓缓上滑，衣物随之微微撩起",
        # 裙摆/腿部撩拨
        "一手从大腿内侧轻轻向上撩起裙角，侧身望向镜头",
        "双手捏起两侧裙摆微微提起，裸腿与大腿内侧完整入镜",
        "单手撩起裙摆一角夹于腰间，另一手叉腰",
        "坐姿下双手轻按双腿向两侧撑开，裙摆随之散开",
        "一手从膝盖内侧缓缓上滑，另一手托腮俯视镜头",
        # 挑逗标志性动作
        "一手指向镜头，另一手插腰，表情挑衅撩人",
        "单手叉腰挺胸，另一手微微拉低肩带",
        "双臂交叉托于胸前，身体微微前倾，胸部自然显露",
        "俯身向镜头伸出一手，领口随俯身动作大幅下垂",
        "双手背于身后，胸部向前挺出，媚眼对镜",
        "用手轻掩嘴角，另一手扯裙角俏皮",
    ]

    # mirror: 一只手拿手机对着镜子拍（画面内可见），另一只手空闲，全身或半身
    _MIRROR_HAND_ACTIONS = [
        # 上半身展示
        "另一手拉低领口，自然裸露锁骨与胸口，嘴唇微启",
        "另一手慢慢拨开一侧肩带滑落至臂弯，侧颈裸露",
        "另一手单手叉腰挺胸，胸部向前挺出自然显露",
        "另一手扶着镜面，身体微贴镜子，表情慵懒妩媚",
        "另一手食指轻抵嘴唇，嘟嘴对镜，眼神含情",
        "另一手从颈部慢慢滑向锁骨，头微微后仰",
        # 腰腹/裙摆展示
        "另一手从腰间向下缓缓撩起裙摆一角，侧身看镜子",
        "另一手捏住裙角微微提起，露出大腿内侧白皙肌肤",
        "另一手扯住上衣下摆向上拉，裸露腰腹与肚脐",
        "另一手插入腰带缓缓拨弄，目光斜瞥镜头",
        # 全身展示
        "另一手从大腿外侧缓缓滑过，腰线与臀部线条显现",
        "另一手叉腰扭动身体对着镜子摆S曲线",
        "另一手轻轻拨弄发丝，向镜子方向抛媚眼",
        "另一手撑在镜面上俯身，领口自然下垂",
        "另一手轻抚小腹，另一条腿向前迈出展示腿部线条",
    ]

    # photo: 他人拍摄视角，双手都自由，可以有更自然丰富的全身姿态
    _PHOTO_HAND_ACTIONS = [
        # 上半身/胸口
        "双手从腰间缓缓向上抚过腰腹与胸口，嘴唇微启，眼神妩媚",
        "双手向两侧轻轻拉开领口，裸露锁骨与胸沟，俯视镜头",
        "一手拨开刘海，另一手轻抚颈部，身体微微后仰",
        "双臂交叉托于胸前，故意向前俯身，胸部自然聚拢",
        "双手从脸颊缓缓下滑至颈部，表情妩媚撩人",
        # 腰腹/裙摆
        "双手同时撩起两侧裙摆至大腿根，大腿内侧完整暴露",
        "一手撩起裙摆夹腰间，另一手从大腿内侧上滑，侧身回眸",
        "双手扯住裙角，微微蹲低，裙摆散开，仰视镜头表情撩人",
        "一手拎起裙摆一角完全掀开，另一手指向镜头挑逗",
        "双手插入裙腰内侧慢慢向下拉，若无其事地望向镜头",
        # 腿部/坐姿
        "坐在地面，双腿向两侧大幅分开，双手分别按在两侧膝盖内侧向外撑",
        "侧坐，一腿搭在另一腿上，双手从大腿上缓缓向下滑过",
        "趴在地面，翘起臀部，双手撑地抬头望向镜头",
        "站姿下一腿向前迈出，双手从腰间向下沿腿部外侧滑过",
        "跪坐，双手叠放大腿上，身体前倾，领口自然开阔",
        # 挑逗动作
        "向镜头缓缓走来，一手轻扯衣角，另一手触碰嘴唇，眼神放电",
        "双手背于身后站立，挺胸抬头，眼神大胆直视镜头",
        "俯身向镜头，双手伸向镜头，胸口与领口自然大幅入镜",
        "转过身去，回头媚眼，双手从背后向下按压裙摆紧贴臀部",
    ]

    # cosplay: 角色扮演风格，可双手自由，偏角色气质动作但同样擦边
    _COSPLAY_HAND_ACTIONS = [
        # 角色气质×挑逗融合
        "叉腰挺胸、表情傲娇，一手向下撩起战裙一角露出大腿内侧",
        "双手托住武器或道具向前倾身，领口随前倾大幅下垂入镜",
        "坐姿下双腿叉开，一手从大腿内侧向上抚过，另一手托腮媚笑",
        "角色自信站姿，一手插腰，另一手拇指挑起内衬领口向外拉",
        "俯身向镜头凝视，双手撑在膝盖上，胸口随俯身自然显露",
        "双手轻轻压在腰间两侧裙摆，向外缓缓掀起，裸腿大幅入镜",
        "一手拨开刘海侧脸，另一手从颈部慢慢下滑至胸口，眼神深邃",
        "侧身回眸角色造型，一手从背后向前绕过腰部，手指触及腹部",
        "跪坐角色拜见姿，上身前倾低头，领口深开，胸口入镜",
        "角色决战起身姿态，扯住衣角向上掀开，大腿内侧侧线完整裸露",
        "懒散倚在柱上，一腿搭起，一手撩起裙摆挂在膝上，媚笑对镜",
        "双手从两侧肩膀向下缓缓拨落肩带，侧脸含情微笑",
    ]

    # 色图模式：性感提示词（基于自拍参考图）
    _SEXY_PROMPT_ZH = (
        "极度性感妩媚，慵懒挑逗姿态，衣物薄透若隐若现，大腿内侧与锁骨完整入镜，"
        "眼神撩人含情，嘴唇微启，皮肤细腻光泽，电影级光影，高清细腻，极致细节"
    )
    _SEXY_NEGATIVE = ""

    # 装逼配图：自信/炫耀感提示词（基于自拍参考图）
    _FLEX_PROMPT_EN = (
        "1girl, solo, confident pose, showing off, proud expression, flexing, "
        "stylish, looking at camera, triumphant mood, masterpiece, best quality, high resolution"
    )
    _FLEX_PROMPT_ZH = (
        "一位女孩，单人，自信姿势，得意表情，装逼/炫耀感，配合场景，看着镜头，氛围得意，高清细腻"
    )

    # 色图描述脱敏：直接敏感词替换为中性表述，避免优化器/生图 API 拒审
    _SEXY_DIRECT_PHRASES = ("色图", "发色图", "来张色图", "来点色图", "色图来一张")
    _SEXY_SAFE_FALLBACK_ZH = "含蓄性感风格"

    @staticmethod
    def _sanitize_sexy_description(description: str) -> str:
        """将易触发审核的色图相关描述替换为中性表述，保留其他有效描述（如浴衣、场景）。"""
        if not description:
            return MaisArtAction._SEXY_SAFE_FALLBACK_ZH
        d = description.strip()
        if d in MaisArtAction._SEXY_DIRECT_PHRASES:
            return MaisArtAction._SEXY_SAFE_FALLBACK_ZH
        if "色图" in d:
            d = d.replace("色图", "含蓄性感").strip() or MaisArtAction._SEXY_SAFE_FALLBACK_ZH
        return d or MaisArtAction._SEXY_SAFE_FALLBACK_ZH

    @staticmethod
    def _get_hand_actions_for_style(selfie_style: str) -> list:
        """根据自拍风格返回对应的手部动作池"""
        if selfie_style == "mirror":
            return MaisArtAction._MIRROR_HAND_ACTIONS
        elif selfie_style == "photo":
            return MaisArtAction._PHOTO_HAND_ACTIONS
        elif selfie_style == "cosplay":
            return MaisArtAction._COSPLAY_HAND_ACTIONS
        else:
            return MaisArtAction._STANDARD_HAND_ACTIONS

    def _get_selfie_reference_image(self) -> Optional[str]:
        """获取自拍参考图片的base64编码。支持多张：配置逗号分隔路径时随机选一张，增加多样性。

        Returns:
            图片的base64编码，如果不存在则返回None
        """
        import random

        raw = self.get_config("selfie.reference_image_path", "").strip()
        if not raw:
            return None

        # 支持逗号分隔的多路径，随机选一张
        paths = [p.strip() for p in raw.split(",") if p.strip()]
        if not paths:
            return None

        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        existing = []
        for p in paths:
            full = p if os.path.isabs(p) else os.path.join(plugin_dir, p)
            if os.path.exists(full):
                existing.append(full)
        if not existing:
            logger.warning(f"{self.log_prefix} 自拍参考图片均不存在: {paths}")
            return None

        image_path = random.choice(existing)
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            logger.info(f"{self.log_prefix} 从文件加载自拍参考图片: {image_path}")
            return image_base64
        except Exception as e:
            logger.error(f"{self.log_prefix} 加载自拍参考图片失败: {e}")
        return None

    async def _schedule_auto_recall_for_recent_message(self, model_config: Dict[str, Any] = None, model_id: str = None, send_timestamp: float = 0.0):
        """安排最近发送消息的自动撤回"""
        global_enabled = self.get_config("auto_recall.enabled", False)
        if not global_enabled or not model_config:
            return

        delay_seconds = model_config.get("auto_recall_delay", 0)
        if delay_seconds <= 0:
            return

        if model_id and not runtime_state.is_recall_enabled(self.chat_id, model_id, global_enabled):
            logger.info(f"{self.log_prefix} 模型 {model_id} 撤回已在当前聊天流禁用")
            return

        await schedule_auto_recall(self.chat_id, delay_seconds, self.log_prefix, self.send_command, send_timestamp)

    async def _generate_image_only(
        self,
        description: str,
        model_id: str = None,
        size: str = "",
        strength: float = None,
        input_image_base64: str = None,
        extra_negative_prompt: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """仅生成图片，不发送消息、不撤回、不缓存

        与 _execute_unified_generation() 共享核心逻辑，但适用于自动自拍等
        需要拿到图片数据而不直接发送的场景。

        Args:
            description: 图片描述/提示词
            model_id: 模型ID，默认使用 generation.default_model
            size: 图片尺寸，留空使用模型默认值
            strength: 图生图强度
            input_image_base64: 输入图片的 base64（图生图用）
            extra_negative_prompt: 额外负面提示词

        Returns:
            (success, image_data): success 为 True 时 image_data 是 base64 或 URL 字符串
        """
        if not model_id:
            model_id = self.get_config("generation.default_model", "model1")

        # 获取模型配置
        model_config = self._get_model_config(model_id)
        if not model_config:
            logger.error(f"{self.log_prefix} [image_only] 模型配置获取失败: {model_id}")
            return False, f"模型 '{model_id}' 不存在或配置无效"

        # 配置验证
        http_base_url = model_config.get("base_url")
        http_api_key = model_config.get("api_key")
        api_format = model_config.get("format", "openai")
        formats_no_base_url = ("comfyui", "runninghub-workflow", "runninghub-quick", "runninghub-ai-app")

        # 检查base_url
        if not http_base_url and api_format not in formats_no_base_url:
            return False, "HTTP配置不完整"

        # 检查api_key（comfyui格式允许为空）
        if api_format != "comfyui" and not http_api_key:
            return False, "HTTP配置不完整"

        # API密钥验证（仅对需要api_key的格式）
        if api_format != "comfyui" and ("YOUR_API_KEY_HERE" in (http_api_key or "") or "xxxxxxxxxxxxxx" in (http_api_key or "")):
            return False, "API密钥未配置"

        # 合并额外负面提示词
        if extra_negative_prompt:
            model_config = merge_negative_prompt(model_config, extra_negative_prompt)

        # 处理尺寸
        image_size, llm_original_size = get_image_size(model_config, size, self.log_prefix)
        if not self._validate_image_size(image_size):
            image_size = model_config.get("default_size", "1024x1024")

        # Gemini/Zai 尺寸处理
        model_config = inject_llm_original_size(model_config, llm_original_size)

        max_retries = self.get_config("components.max_retries", 2)

        try:
            api_client = self._get_api_client(api_format)
            success, result = await api_client.generate_image(
                prompt=description,
                model_config=model_config,
                size=image_size,
                strength=strength,
                input_image_base64=input_image_base64,
                max_retries=max_retries,
            )
        except Exception as e:
            logger.error(f"{self.log_prefix} [image_only] 生图异常: {e!r}")
            return False, f"生图异常: {str(e)[:100]}"

        if not success:
            return False, result

        # 处理返回数据
        final_image_data = self.image_processor.process_api_response(result)
        if not final_image_data:
            return False, "API返回数据格式错误"

        # 如果是 URL，下载并转为 base64
        return await resolve_image_data(
            final_image_data, self._download_and_encode_base64,
            f"{self.log_prefix} [image_only]"
        )

    def _extract_description_from_message(self) -> str:
        """从用户消息中提取图片描述

        Returns:
            str: 提取的图片描述，如果无法提取则返回空字符串
        """
        if not self.action_message:
            return ""
            
        # 获取消息文本
        message_text = (self.action_message.processed_plain_text or
                       self.action_message.display_message or
                       self.action_message.raw_message or "").strip()
        
        if not message_text:
            return ""
            
        import re
        
        # 移除常见的画图相关前缀
        patterns_to_remove = [
            r'^画',           # "画"
            r'^绘制',         # "绘制"
            r'^生成图片',     # "生成图片"
            r'^画图',         # "画图"
            r'^帮我画',       # "帮我画"
            r'^请画',         # "请画"
            r'^能不能画',     # "能不能画"
            r'^可以画',       # "可以画"
            r'^画一个',       # "画一个"
            r'^画一只',       # "画一只"
            r'^画张',         # "画张"
            r'^画幅',         # "画幅"
            r'^图[：:]',      # "图："或"图:"
            r'^生成图片[：:]', # "生成图片："或"生成图片:"
            r'^[：:]',        # 单独的冒号
            r'^用\s*模型\s*\S+\s*',       # "用模型3" / "用 模型 abc"
            r'^用\s*model\s*\S+\s*',      # "用model2" / "用 model abc"
        ]
        
        cleaned_text = message_text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # 移除常见的后缀
        suffix_patterns = [
            r'图片$',         # "图片"
            r'图$',           # "图"
            r'一下$',         # "一下"
            r'呗$',           # "呗"
            r'吧$',           # "吧"
        ]
        
        for pattern in suffix_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # 清理空白字符
        cleaned_text = cleaned_text.strip()
        
        # 如果清理后为空，返回原消息（可能是简单的描述）
        if not cleaned_text:
            cleaned_text = message_text
            
        # 限制长度，避免过长的描述
        if len(cleaned_text) > 100:
            cleaned_text = cleaned_text[:100]
            
        return cleaned_text

