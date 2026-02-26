"""提示词优化器模块

使用 MaiBot 主 LLM 将用户描述优化为专业的绘画提示词。
纯净调用，不带人设和回复风格。
"""
from typing import Tuple, Optional
from src.common.logger import get_logger
from src.plugin_system.apis import llm_api

logger = get_logger("mais_art.optimizer")

# 提示词优化系统提示词
OPTIMIZER_SYSTEM_PROMPT = """You are a professional AI art prompt engineer. Your task is to convert user descriptions into high-quality English prompts for image generation models (Stable Diffusion, DALL-E, etc.).

## Rules:
1. Output ONLY the English prompt, no explanations or translations
2. Use comma-separated tags/phrases
3. Follow structure: subject, action/pose, scene/background, lighting, style, quality tags
4. Use weight syntax for emphasis: (keyword:1.2) for important elements
5. Keep prompts concise but descriptive (50-150 words ideal)
6. Always end with quality tags: masterpiece, best quality, high resolution

## Examples:

Input: 海边的女孩
Output: 1girl, solo, standing on beach, ocean waves, sunset sky, orange and pink clouds, warm lighting, summer dress, wind blowing hair, peaceful expression, masterpiece, best quality, high resolution

Input: 可爱的猫咪睡觉
Output: cute cat, sleeping, curled up on soft blanket, fluffy fur, closed eyes, peaceful, warm indoor lighting, cozy atmosphere, detailed fur texture, masterpiece, best quality, high resolution

Input: 赛博朋克城市
Output: cyberpunk cityscape, neon lights, futuristic buildings, flying cars, rain, reflective wet streets, holographic advertisements, purple and blue color scheme, atmospheric, cinematic lighting, masterpiece, best quality, high resolution

Now convert the following description to an English prompt:"""

# 自拍场景专用提示词：只生成场景/环境/光线/氛围，不生成角色外观
# 参考 Seedream 提示词指南，鼓励多样性（光线、构图、时段、天气等）
SELFIE_SCENE_SYSTEM_PROMPT = """You are a scene description assistant for selfie image generation (Stable Diffusion / Seedream). The character's appearance is already defined separately. Your task is to convert the user's description into English tags describing ONLY the scene, environment, lighting, mood, and atmosphere.

## Rules:
1. Output ONLY English tags, no explanations
2. Use comma-separated tags/phrases
3. NEVER include character appearance (hair color, eye color, clothing, body type, etc.)
4. NEVER include character names or franchise references
5. Focus on: background, environment, lighting, weather, mood, atmosphere, time of day, composition hints
6. Keep it concise (20-60 words)
7. DIVERSITY: vary lighting (golden hour, soft diffused, dramatic, warm/cool), time of day (morning, afternoon, dusk), weather when applicable, mood (cozy, lively, serene, playful). Avoid repetitive structures.
8. If the description is just "selfie" or similar with no scene info, pick a varied generic scene (e.g. indoor soft light, cafe, park bench, cozy room)

## Examples:

Input: 在海边自拍
Output: beach background, ocean waves, golden sunset, warm sunlight, sand, gentle breeze, summer atmosphere

Input: 图书馆学习
Output: library interior, bookshelves, warm ambient lighting, quiet atmosphere, wooden desk, soft focus background

Input: 来张自拍
Output: casual indoor setting, soft natural lighting, clean background

Input: 下雨天在咖啡店
Output: coffee shop interior, rainy window, warm cozy atmosphere, soft indoor lighting, rain drops on glass, bokeh background

Input: 晚上在家
Output: cozy living room, warm lamp light, evening atmosphere, soft shadows, peaceful mood

Now convert the following description to English scene tags (use varied lighting/mood when possible):"""

# 豆包（火山方舟 Seedream）专用：中文自然语言提示词
# 参考：https://www.volcengine.com/docs/82379/1666946 支持中英文，建议不超过300汉字
OPTIMIZER_SYSTEM_PROMPT_DOUBAO = """你是一名专业的图像生成提示词助手，面向火山方舟豆包 Seedream 模型。请将用户的描述改写成一段简洁、画面感强的中文自然语言提示词，直接用于文生图/图生图。

## 要求：
1. 只输出中文提示词本身，不要解释、不要翻译成英文、不要加“提示词：”等前缀
2. 使用自然、通顺的中文句子或短语，描述画面内容（主体、动作、场景、光线、氛围、风格）
3. 建议 50–150 字，不超过 300 字，避免堆砌无关词
4. 不要使用英文 tag、不要用括号权重语法
5. 可适当补充画面细节（如光影、色调、构图感），使生成效果更稳定

## 示例：

输入：海边的女孩
输出：一位女孩站在海边，身后是海浪与夕阳，天空有橙粉色的云，穿着夏装，头发被风吹起，表情宁静，画面温暖柔和，高清细腻

输入：可爱的猫咪睡觉
输出：一只可爱的猫咪蜷缩在柔软的毯子上睡觉，毛发蓬松，闭着眼睛，室内暖光，氛围温馨宁静，细节清晰

输入：赛博朋克城市
输出：赛博朋克风格的城市街景，霓虹灯闪烁，未来感建筑，飞车与雨夜，地面反光，紫蓝色调，电影感氛围

请将以下用户描述改写成中文自然语言提示词："""

# 豆包自拍场景专用：仅场景/环境/光线/氛围，中文自然语言
SELFIE_SCENE_SYSTEM_PROMPT_DOUBAO = """你是自拍图像生成的场景描述助手，面向豆包 Seedream。角色外观已单独定义，你只需把用户描述改写成仅包含场景、环境、光线、氛围的中文自然语言，供图生图使用。

## 要求：
1. 只输出中文描述本身，不要解释或前缀
2. 不要包含角色外貌（发型、衣着、体型等）和角色名
3. 只写：背景、环境、光线、天气、氛围、时间感、构图感等
4. 20–80 字，自然通顺，可适当变化光线与氛围（如黄昏、室内柔光、咖啡店、雨天等）

## 示例：

输入：在海边自拍
输出：海边背景，海浪与沙滩，黄昏暖光，微风，夏日氛围

输入：图书馆学习
输出：图书馆内景，书架与书桌，暖色环境光，安静氛围，背景略虚化

输入：来张自拍
输出：室内日常场景，柔和自然光，简洁背景

输入：下雨天在咖啡店
输出：咖啡店内，窗外雨景，暖黄灯光，雨滴落在玻璃上，氛围温馨

请将以下描述改写成仅含场景与氛围的中文提示词："""


class PromptOptimizer:
    """提示词优化器

    使用 MaiBot 主 LLM 优化用户描述为专业绘画提示词
    """

    def __init__(self, log_prefix: str = "[PromptOptimizer]"):
        self.log_prefix = log_prefix
        self._model_config = None

    def _get_model_config(self):
        """获取可用的 LLM 模型配置"""
        if self._model_config is None:
            try:
                models = llm_api.get_available_models()
                # 使用 replyer 模型（首要回复模型）
                if "replyer" in models:
                    self._model_config = models["replyer"]
                else:
                    logger.warning(f"{self.log_prefix} 没有找到 replyer 模型")
                    return None
            except Exception as e:
                logger.error(f"{self.log_prefix} 获取模型配置失败: {e}")
                return None
        return self._model_config

    async def optimize(self, user_description: str, scene_only: bool = False, api_format: str = None) -> Tuple[bool, str]:
        """优化用户描述为专业绘画提示词

        Args:
            user_description: 用户原始描述（中文或英文）
            scene_only: 仅生成场景/环境描述（自拍模式用，不包含角色外观）
            api_format: 生图 API 格式，如 "doubao" 时使用中文自然语言提示词（火山方舟 Seedream）

        Returns:
            Tuple[bool, str]: (是否成功, 优化后的提示词或错误信息)
        """
        if not user_description or not user_description.strip():
            return False, "描述不能为空"

        model_config = self._get_model_config()
        if not model_config:
            # 降级：直接返回原始描述
            logger.warning(f"{self.log_prefix} 无可用模型，降级使用原始描述")
            return True, user_description

        try:
            # 豆包格式使用中文自然语言提示词（火山方舟 Seedream 支持中英文，效果更自然）
            use_doubao_style = (api_format or "").strip().lower() == "doubao"
            if scene_only:
                system_prompt = SELFIE_SCENE_SYSTEM_PROMPT_DOUBAO if use_doubao_style else SELFIE_SCENE_SYSTEM_PROMPT
            else:
                system_prompt = OPTIMIZER_SYSTEM_PROMPT_DOUBAO if use_doubao_style else OPTIMIZER_SYSTEM_PROMPT

            # 构建完整 prompt
            full_prompt = f"{system_prompt}\n\nInput: {user_description.strip()}\nOutput:"

            mode_label = "场景提示词" if scene_only else "提示词"
            style_label = "中文自然语言(豆包)" if use_doubao_style else "英文 tag"
            logger.info(f"{self.log_prefix} 开始优化{mode_label} ({style_label}): {user_description[:50]}...")

            # 调用 LLM（不传递 temperature 和 max_tokens，使用模型默认值）
            success, response, reasoning, model_name = await llm_api.generate_with_model(
                prompt=full_prompt,
                model_config=model_config,
                request_type="plugin.prompt_optimize",
            )

            if success and response:
                # 清理响应（移除可能的前缀/后缀）
                optimized = self._clean_response(response)
                logger.info(f"{self.log_prefix} 优化成功 (模型: {model_name}): {optimized[:80]}...")
                return True, optimized
            else:
                logger.warning(f"{self.log_prefix} LLM 返回空响应，降级使用原始描述: {user_description[:50]}...")
                return True, user_description

        except Exception as e:
            logger.error(f"{self.log_prefix} 优化失败: {e}，使用原始描述: {user_description[:50]}...")
            # 降级：返回原始描述
            return True, user_description

    def _clean_response(self, response: str) -> str:
        """清理 LLM 响应

        移除可能的前缀、后缀、引号等
        """
        result = response.strip()

        # 移除可能的 "Output:" / "输出:" 等前缀
        prefixes_to_remove = ["Output:", "output:", "Prompt:", "prompt:", "输出：", "输出:"]
        for prefix in prefixes_to_remove:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()

        # 移除首尾引号
        if (result.startswith('"') and result.endswith('"')) or \
           (result.startswith("'") and result.endswith("'")):
            result = result[1:-1]

        # 移除多余换行
        result = " ".join(result.split())

        return result


# 全局优化器实例
_optimizer_instance = None

def get_optimizer(log_prefix: str = "[PromptOptimizer]") -> PromptOptimizer:
    """获取提示词优化器实例（单例）"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = PromptOptimizer(log_prefix)
    else:
        _optimizer_instance.log_prefix = log_prefix
    return _optimizer_instance


async def optimize_prompt(
    user_description: str,
    log_prefix: str = "[PromptOptimizer]",
    scene_only: bool = False,
    api_format: str = None,
) -> Tuple[bool, str]:
    """便捷函数：优化提示词

    Args:
        user_description: 用户原始描述
        log_prefix: 日志前缀
        scene_only: 仅生成场景/环境描述（自拍模式用）
        api_format: 生图 API 格式，如 "doubao" 时输出中文自然语言提示词（火山方舟 Seedream）

    Returns:
        Tuple[bool, str]: (是否成功, 优化后的提示词)
    """
    optimizer = get_optimizer(log_prefix)
    return await optimizer.optimize(user_description, scene_only=scene_only, api_format=api_format)
