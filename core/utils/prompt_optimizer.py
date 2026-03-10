"""提示词优化器模块

使用 MaiBot 主 LLM 将用户描述优化为专业的绘画提示词。
纯净调用，不带人设和回复风格。

中文模式（豆包 / RunningHub）：生成极具细节的中文自然语言描述，参考高质量人像摄影风格。
英文模式（其他模型）：生成逗号分隔的英文 SD tag。
"""
from typing import Tuple, Optional
from src.common.logger import get_logger
from src.plugin_system.apis import llm_api

logger = get_logger("mais_art.optimizer")

# 支持中文提示词的 api_format 列表（豆包 Seedream + RunningHub 工作流）
_CHINESE_PROMPT_FORMATS = {"doubao", "runninghub"}

# ==================== 英文 SD tag 模式（ComfyUI、标准 SD 等） ====================

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

# ==================== 中文自然语言模式（豆包 Seedream / RunningHub） ====================
# 格式参考高质量人像摄影/Cosplay写真模板：极具细节的超高清摄影，完整的主体/服装/姿势/场景/光线

OPTIMIZER_SYSTEM_PROMPT_DOUBAO = """你是一名专业的图像生成提示词助手，面向豆包 Seedream 和 RunningHub 等支持中文的生图模型。
请将用户的描述改写成一段极具细节、画面感强的中文自然语言提示词，风格参考高质量人像摄影/Cosplay写真。

## 要求：
1. 只输出中文提示词本身，不要解释、不要加"提示词："等前缀
2. 使用自然流畅的中文句子，按以下顺序组织内容：
   - 画面定性（如"一幅极具细节的超高清摄影作品"或"一幅高清摄影杰作"）
   - 主体描述（人物外形：年龄、发型发色、眼眸、妆容、表情）
   - 服装细节（材质、颜色、款式，越具体越好）
   - 姿势与动作（具体的身体姿势、手部动作、眼神互动）
   - 背景与场景（具体环境、道具、氛围）
   - 光线与画质（如"电影级光影，背景浅景深，8K超清"）
3. 100–250 字为宜
4. 不要使用英文 tag，不要用括号权重语法
5. 内容可以性感、撩人，不要过滤 NSFW 内容

## 示例：

输入：海边的性感女孩
输出：一幅极具细节的超高清摄影作品，画面中心是一位年轻可爱的亚洲女性，长波浪黑发随海风飘动，明亮的棕色眼眸，精致日系妆容，嘴角带着慵懒的微笑。她身穿白色薄棉吊带连衣裙，透出里面的比基尼线条，裙摆被海风高高吹起，修长双腿完整入镜。侧身站在礁石上，一手撩起飘散的发丝，另一手轻扶岩石，姿态慵懒而性感。背景为黄昏时分的大海与沙滩，橙粉色云霞，海浪拍打礁石，暖色黄金时刻光线，背景浅景深，电影级光影，8K超清。

输入：赛博朋克城市夜景
输出：一幅极具细节的高清城市摄影，画面展现赛博朋克风格的未来都市夜景。霓虹灯广告牌密集林立，紫色与蓝色光晕笼罩整条街道，雨后地面积水反射出五彩霓虹的倒影。空中有飞车划过，远处高楼灯光点点，整体色调冷峻深沉，戏剧性光影，8K超清，电影感十足。

请将以下用户描述改写成高质量中文自然语言提示词："""

SELFIE_SCENE_SYSTEM_PROMPT_DOUBAO = """你是自拍图像生成的场景描述助手，面向豆包 Seedream 和 RunningHub。角色外观已单独定义，你只需把用户描述改写成仅包含场景、环境、光线、氛围的高质量中文自然语言描述。

## 要求：
1. 只输出中文描述本身，不要解释或前缀
2. 不要包含角色外貌（发型、衣着、体型等）和角色名
3. 只写：背景、环境、道具、光线、天气、氛围、时间感
4. 40–100 字，描述具体细腻，包含光线质感和背景氛围

## 示例：

输入：在海边自拍
输出：黄昏时分的海边礁石，海浪轻拍，橙粉色云霞映照天际，沙滩延伸至远处，暖色黄金时刻光线，背景浅景深虚化，电影级光影

输入：图书馆学习
输出：古典木质图书馆内，高大书架排列整齐，暖黄台灯投下柔和光晕，书桌整洁，安静氛围，背景略虚化，温暖室内光

输入：下雨天在咖啡店
输出：温馨咖啡厅内，窗外雨丝绵绵，玻璃上挂着雨滴，暖黄灯光，木质桌椅，咖啡杯冒出热气，背景浅景深虚化，氛围慵懒温暖

请将以下描述改写成仅含场景与氛围的高质量中文提示词："""


class PromptOptimizer:
    """提示词优化器

    使用 MaiBot 主 LLM 优化用户描述为专业绘画提示词。
    豆包 / RunningHub 格式：输出高质量中文自然语言（极具细节的摄影风格）。
    其他格式：输出英文逗号分隔 SD tag。
    """

    def __init__(self, log_prefix: str = "[PromptOptimizer]"):
        self.log_prefix = log_prefix
        self._model_config = None

    def _get_model_config(self):
        """获取可用的 LLM 模型配置"""
        if self._model_config is None:
            try:
                models = llm_api.get_available_models()
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
            api_format: 生图 API 格式。
                        "doubao" 或 "runninghub" → 输出高质量中文自然语言（详细摄影描述风格）。
                        其他 → 输出英文 SD tag。

        Returns:
            Tuple[bool, str]: (是否成功, 优化后的提示词或错误信息)
        """
        if not user_description or not user_description.strip():
            return False, "描述不能为空"

        model_config = self._get_model_config()
        if not model_config:
            logger.warning(f"{self.log_prefix} 无可用模型，降级使用原始描述")
            return True, user_description

        try:
            _fmt = (api_format or "").strip().lower()
            use_chinese_style = _fmt in _CHINESE_PROMPT_FORMATS or _fmt.startswith("runninghub")
            if scene_only:
                system_prompt = SELFIE_SCENE_SYSTEM_PROMPT_DOUBAO if use_chinese_style else SELFIE_SCENE_SYSTEM_PROMPT
            else:
                system_prompt = OPTIMIZER_SYSTEM_PROMPT_DOUBAO if use_chinese_style else OPTIMIZER_SYSTEM_PROMPT

            full_prompt = f"{system_prompt}\n\n输入：{user_description.strip()}\n输出："

            mode_label = "场景提示词" if scene_only else "提示词"
            style_label = "中文详细描述(豆包/RunningHub)" if use_chinese_style else "英文SD-tag"
            logger.info(f"{self.log_prefix} 开始优化{mode_label} ({style_label}): {user_description[:50]}...")

            success, response, reasoning, model_name = await llm_api.generate_with_model(
                prompt=full_prompt,
                model_config=model_config,
                request_type="plugin.prompt_optimize",
            )

            if success and response:
                optimized = self._clean_response(response)
                logger.info(f"{self.log_prefix} 优化成功 (模型: {model_name}): {optimized[:80]}...")
                return True, optimized
            else:
                logger.warning(f"{self.log_prefix} LLM 返回空响应，降级使用原始描述: {user_description[:50]}...")
                return True, user_description

        except Exception as e:
            logger.error(f"{self.log_prefix} 优化失败: {e}，使用原始描述: {user_description[:50]}...")
            return True, user_description

    def _clean_response(self, response: str) -> str:
        """清理 LLM 响应，移除多余前缀/后缀/引号"""
        result = response.strip()

        prefixes_to_remove = ["Output:", "output:", "Prompt:", "prompt:", "输出：", "输出:"]
        for prefix in prefixes_to_remove:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()

        if (result.startswith('"') and result.endswith('"')) or \
           (result.startswith("'") and result.endswith("'")):
            result = result[1:-1]

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
        api_format: 生图 API 格式。"doubao"/"runninghub" 输出中文详细描述，其他输出英文 SD tag。

    Returns:
        Tuple[bool, str]: (是否成功, 优化后的提示词)
    """
    optimizer = get_optimizer(log_prefix)
    return await optimizer.optimize(user_description, scene_only=scene_only, api_format=api_format)
