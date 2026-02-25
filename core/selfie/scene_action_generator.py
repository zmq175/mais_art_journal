"""
场景动作生成器

根据 ActivityInfo 生成符合情境的动作和 Stable Diffusion 提示词。

自动自拍：优先使用 LLM 根据活动描述生成英文 SD 场景标签，失败时取消。
手动自拍：优先使用 LLM 生成手部动作（generate_hand_action_with_llm），
         失败时回退到风格专属动作池随机选取。

多样性：活动映射含多个变体，随机选择；LLM prompt 鼓励多样化输出。
"""

import json
import random
import re
from typing import Dict, List, Optional

from src.common.logger import get_logger

from .schedule_provider import ActivityInfo
from ..utils import SELFIE_HAND_NEGATIVE, ANTI_DUAL_PHONE_PROMPT

logger = get_logger("auto_selfie.scene")


# ==================== 确定性映射（手动自拍 + LLM 兜底） ====================
# 每个活动类型有多个变体，随机选择以增加多样性（参考 Seedream 提示词指南）

# 活动类型到动作的映射（每种类型多个变体）
ACTIVITY_ACTIONS: Dict[str, List[str]] = {
    "sleeping": ["lying down, hugging pillow, cozy", "curled up, soft blanket, peaceful", "resting head on pillow, sleepy"],
    "waking_up": ["stretching, yawning, messy hair", "rubbing eyes, drowsy, morning light", "sitting on bed, stretching arms"],
    "eating": ["holding chopsticks, eating", "sipping drink, enjoying meal", "taking a bite, happy expression"],
    "working": ["typing on laptop, focused", "reading screen, concentrated", "resting chin on hand, thoughtful"],
    "studying": ["holding book, reading", "taking notes, focused", "highlighting text, engaged"],
    "exercising": ["stretching, athletic, holding water bottle", "warming up, energetic", "cooling down, relaxed posture"],
    "relaxing": ["lying on couch, relaxed, listening to music", "curled up with blanket", "sitting comfortably, legs crossed"],
    "socializing": ["making peace sign, happy, laughing", "waving hand, cheerful", "hands clasped, excited smile"],
    "commuting": ["holding bag, walking, wearing earbuds", "standing, holding strap", "waiting, looking around"],
    "hobby": ["holding camera, creative", "focused on craft, concentrated", "holding tools, engaged"],
    "self_care": ["applying makeup, mirror", "skincare routine, gentle", "fixing hair, mirror reflection"],
    "other": ["standing, casual pose, natural", "relaxed stance, comfortable", "leaning slightly, easy-going"],
}

# 活动类型到场景环境的映射（多样变体）
ACTIVITY_ENVIRONMENTS: Dict[str, List[str]] = {
    "sleeping": ["bedroom, dim lighting, cozy atmosphere, bed", "soft pillows, warm blankets", "quiet room, night ambience"],
    "waking_up": ["bedroom, morning light, curtains, warm sunlight", "window light, soft glow", "dawn atmosphere, gentle rays"],
    "eating": ["dining room, table setting", "cozy cafe, warm interior", "kitchen counter, casual meal"],
    "working": ["office desk, computer screen", "study room, clean setup", "workspace, minimal background"],
    "studying": ["library, bookshelves, desk lamp", "study desk, warm lamp", "quiet corner, books around"],
    "exercising": ["gym, fitness equipment", "home workout space", "outdoor park, green background"],
    "relaxing": ["living room, sofa, afternoon sun", "cozy corner, soft cushions", "balcony, city view"],
    "socializing": ["outdoor cafe, bright atmosphere", "restaurant, warm lighting", "park bench, natural scenery"],
    "commuting": ["city street, urban", "subway platform, soft light", "bus stop, morning scene"],
    "hobby": ["art studio, creative space", "workshop, organized chaos", "creative corner, inspiring"],
    "self_care": ["bathroom, mirror, vanity", "dressing table, soft light", "vanity area, clean aesthetic"],
    "other": ["indoor, natural lighting", "casual setting, soft focus", "neutral background, clean"],
}

# 活动类型到表情的映射（多样变体）
ACTIVITY_EXPRESSIONS: Dict[str, List[str]] = {
    "sleeping": ["peaceful expression, closed eyes", "serene, restful", "content smile, dreaming"],
    "waking_up": ["drowsy expression, half-open eyes", "sleepy smile, gentle", "soft expression, waking slowly"],
    "eating": ["happy expression, enjoying food", "content smile, satisfied", "delighted, savoring"],
    "working": ["focused expression, serious", "concentrated, thoughtful", "determined, engaged"],
    "studying": ["focused, thoughtful expression", "absorbed, curious", "intent, learning"],
    "exercising": ["energetic expression, determined", "bright smile, motivated", "healthy glow, active"],
    "relaxing": ["relaxed smile, content", "peaceful, at ease", "gentle expression, comfortable"],
    "socializing": ["bright smile, happy", "laughing, joyful", "warm expression, friendly"],
    "commuting": ["calm expression", "peaceful, contemplative", "relaxed, observing"],
    "hobby": ["excited, passionate", "enthusiastic, engaged", "bright eyes, inspired"],
    "self_care": ["gentle smile, self-care", "serene, pampered", "soft expression, relaxed"],
    "other": ["natural smile", "warm expression", "pleasant, approachable"],
}

# 活动类型到光线的映射（多样变体）
ACTIVITY_LIGHTING: Dict[str, List[str]] = {
    "sleeping": ["dim warm light, night lamp", "soft ambient glow", "moonlight, gentle shadows"],
    "waking_up": ["soft morning light, golden hour", "warm sunrise, gentle rays", "dawn light, soft diffusion"],
    "eating": ["warm indoor lighting", "soft overhead light", "candlelight, cozy"],
    "working": ["office lighting, even illumination", "desk lamp, focused light", "natural window light"],
    "studying": ["desk lamp, focused light", "warm study lamp", "soft reading light"],
    "exercising": ["bright natural light", "gym lighting, energetic", "outdoor sunshine, dynamic"],
    "relaxing": ["soft afternoon light, warm ambient light", "golden hour, cozy", "dappled light, peaceful"],
    "socializing": ["bright cheerful lighting", "warm cafe lights", "natural daylight, lively"],
    "commuting": ["morning sunlight", "soft urban light", "overcast, diffused"],
    "hobby": ["creative studio lighting", "natural light, inspiring", "warm lamp, focused"],
    "self_care": ["bathroom lighting, mirror reflection", "vanity lights, soft", "natural mirror light"],
    "other": ["natural lighting", "soft diffused light", "balanced illumination"],
}


# ==================== LLM 场景生成（自动自拍专用） ====================

_SCENE_LLM_PROMPT_BASE = """You are a selfie scene tag generator for anime image generation (Stable Diffusion / Seedream).
Given a character's current activity description, output a JSON object with 4 keys:
- action: physical pose/gesture/hand position (3-8 English tags)
- environment: background and surroundings (3-8 English tags)
- expression: facial expression (2-5 English tags)
- lighting: light conditions (2-4 English tags)

Rules:
1. Output ONLY valid JSON, no markdown, no explanations
2. All values must be English tags suitable for Stable Diffusion / Seedream
3. Do NOT include character appearance (hair, eyes, clothing)
4. Tags should feel natural for the scenario
5. Keep tags concise and descriptive
6. IMPORTANT for action: prefer simple, AI-friendly gestures. AVOID complex multi-finger details (e.g. heart shape with hands, interlocked fingers) as they cause generation artifacts
7. DIVERSITY: vary camera angles (close-up, medium shot, cowboy shot), lighting (golden hour, soft diffused, dramatic), mood (cheerful, serene, playful), time of day, and weather. Avoid repetitive structures across generations."""

# 按风格补充的约束
_SCENE_STYLE_HINTS = {
    "standard": """
7. STYLE CONSTRAINT - Standard selfie: one hand is holding the phone (OFF-SCREEN). Only the OTHER hand is free. Action MUST be a single-hand gesture (e.g. peace sign, touching hair, hand on chin, waving). NEVER use two-hand actions.""",

    "mirror": """
7. STYLE CONSTRAINT - Mirror selfie: one hand holds the phone (VISIBLE in mirror). Only the OTHER hand is free. Action should be single-hand poses suitable for mirror reflection (e.g. hand on hip, adjusting hair, fixing collar, hand in pocket).""",

    "photo": """
7. STYLE CONSTRAINT - Third-person photo: both hands are FREE (someone else is taking the photo). Action can use both hands naturally (e.g. hands behind back, walking casually, holding a cup, leaning on railing, sitting). Prefer natural full-body poses.""",
}

_SCENE_LLM_EXAMPLES = """
Examples (vary lighting, mood, composition for diversity):

Activity: 在书房看轻小说
{"action": "holding book, reading, relaxed pose", "environment": "study room, bookshelf, warm interior", "expression": "content smile, absorbed", "lighting": "desk lamp, warm indoor light"}

Activity: 在厨房做早饭
{"action": "holding spatula, cooking", "environment": "kitchen, stove, morning atmosphere", "expression": "happy smile, focused on cooking", "lighting": "morning light through window, bright kitchen"}

Activity: 在公园散步
{"action": "walking, casual stroll", "environment": "park, trees, pathway, flowers", "expression": "peaceful smile, relaxed", "lighting": "soft natural sunlight, dappled light"}

Activity: 在咖啡厅休息
{"action": "holding cup, sipping coffee", "environment": "cafe interior, warm atmosphere, window seat", "expression": "gentle smile, relaxed", "lighting": "afternoon light, soft shadows, cozy ambience"}

Activity: 下雨天在家
{"action": "curled up, blanket", "environment": "living room, rainy window, cozy", "expression": "content, peaceful", "lighting": "overcast light, rainy day glow, soft diffusion"}

Now generate for the following activity (use varied lighting/mood when possible):"""


def _build_scene_llm_prompt(selfie_style: str) -> str:
    """组装带风格约束的 LLM 场景生成 prompt"""
    style_hint = _SCENE_STYLE_HINTS.get(selfie_style, _SCENE_STYLE_HINTS["standard"])
    return f"{_SCENE_LLM_PROMPT_BASE}{style_hint}{_SCENE_LLM_EXAMPLES}"


async def generate_scene_with_llm(activity_info: ActivityInfo, selfie_style: str = "standard") -> Optional[Dict[str, str]]:
    """使用 LLM 根据活动描述生成英文 SD 场景标签

    Args:
        activity_info: 活动信息
        selfie_style: 自拍风格，用于约束 LLM 生成的动作类型

    Returns:
        包含 action, environment, expression, lighting 的字典，失败返回 None
    """
    try:
        from src.plugin_system.apis import llm_api

        models = llm_api.get_available_models()
        model = models.get("replyer")
        if not model:
            logger.warning("未找到 replyer 模型，LLM 场景生成失败")
            return None

        system_prompt = _build_scene_llm_prompt(selfie_style)
        prompt = f"{system_prompt}\n\nActivity: {activity_info.description}"

        success, response, _, model_name = await llm_api.generate_with_model(
            prompt=prompt,
            model_config=model,
            request_type="plugin.auto_selfie_scene",
            temperature=0.7,
            max_tokens=8192,
        )

        if not success or not response:
            logger.warning("LLM 场景生成返回空响应")
            return None

        # 清理响应（移除可能的 markdown 代码块）
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        scene = json.loads(cleaned)

        # 验证必要字段
        required_keys = {"action", "environment", "expression", "lighting"}
        if not required_keys.issubset(scene.keys()):
            missing = required_keys - set(scene.keys())
            logger.warning(f"LLM 场景缺少字段: {missing}")
            return None

        # 确保所有值都是字符串
        for key in required_keys:
            if not isinstance(scene[key], str) or not scene[key].strip():
                logger.warning(f"LLM 场景字段 {key} 无效: {scene.get(key)}")
                return None

        logger.info(f"LLM 场景生成成功 (模型: {model_name}): action={scene['action'][:50]}")
        return {
            "hand_action": scene["action"],
            "environment": scene["environment"],
            "expression": scene["expression"],
            "lighting": scene["lighting"],
        }

    except json.JSONDecodeError as e:
        logger.warning(f"LLM 场景 JSON 解析失败: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM 场景生成异常: {e}")
        return None


async def generate_hand_action_with_llm(description: str, selfie_style: str = "standard") -> Optional[str]:
    """使用与自动自拍同一套 LLM prompt 生成手部动作

    复用 _build_scene_llm_prompt（风格感知），将用户描述作为 Activity 输入，
    解析完整 JSON 后只提取 action 字段返回。

    用于手动自拍无日程数据时，动作池兜底之前。

    Args:
        description: 用户的场景描述
        selfie_style: 自拍风格，约束动作类型

    Returns:
        英文手部动作标签字符串，失败返回 None
    """
    try:
        from src.plugin_system.apis import llm_api

        models = llm_api.get_available_models()
        model = models.get("replyer")
        if not model:
            logger.warning("未找到 replyer 模型，手部动作生成失败")
            return None

        system_prompt = _build_scene_llm_prompt(selfie_style)
        prompt = f"{system_prompt}\n\nActivity: {description}"

        success, response, _, model_name = await llm_api.generate_with_model(
            prompt=prompt,
            model_config=model,
            request_type="plugin.selfie_hand_action",
            temperature=0.7,
            max_tokens=8192,
        )

        if not success or not response:
            logger.warning("手部动作 LLM 返回空响应")
            return None

        # 清理响应
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        scene = json.loads(cleaned)

        action = scene.get("action")
        if not isinstance(action, str) or not action.strip():
            logger.warning(f"手部动作字段无效: {action}")
            return None

        logger.info(f"LLM 手部动作生成成功 (模型: {model_name}): {action[:60]}")
        return action.strip()

    except json.JSONDecodeError as e:
        logger.warning(f"手部动作 JSON 解析失败: {e}")
        return None
    except Exception as e:
        logger.error(f"手部动作 LLM 生成异常: {e}")
        return None


# ==================== 公共函数 ====================

def _get_selfie_scene_for_style(selfie_style: str) -> List[str]:
    """返回自拍风格的场景变体列表，随机选择以增加多样性"""
    if selfie_style == "mirror":
        return [
            "mirror selfie, reflection in mirror, holding phone in hand, phone visible, looking at mirror, indoor scene",
            "mirror shot, vanity, reflection, warm mirror light, indoor",
            "selfie in mirror, bathroom or bedroom mirror, soft reflection",
        ]
    if selfie_style == "photo":
        return [
            "photo, candid shot, natural pose, full body or upper body, looking away or at camera, (natural composition:1.2)",
            "third-person photograph, casual pose, natural lighting, medium shot",
            "candid photo, relaxed composition, natural stance",
        ]
    return [
        "selfie, front camera view, POV selfie, (front facing selfie camera angle:1.3), looking at camera, slight high angle selfie, upper body shot, cowboy shot, (centered composition:1.2)",
        "selfie, front camera, arm extended, centered, upper body, looking at lens",
        "POV selfie, front facing, slight high angle, cowboy shot, warm lighting",
    ]


def get_action_for_activity(activity_info: ActivityInfo) -> Dict[str, str]:
    """
    根据活动类型获取场景数据（手动自拍使用），从变体池随机选择以增加多样性。

    Args:
        activity_info: 活动信息

    Returns:
        包含 hand_action, environment, expression, lighting 的字典
    """
    def _pick(key: str, mapping: Dict[str, List[str]]) -> str:
        pool = mapping.get(activity_info.activity_type.value, mapping["other"])
        return random.choice(pool) if isinstance(pool, list) else pool

    return {
        "hand_action": _pick("hand_action", ACTIVITY_ACTIONS),
        "environment": _pick("environment", ACTIVITY_ENVIRONMENTS),
        "expression": _pick("expression", ACTIVITY_EXPRESSIONS),
        "lighting": _pick("lighting", ACTIVITY_LIGHTING),
    }


async def convert_to_selfie_prompt(
    activity_info: ActivityInfo,
    selfie_style: str = "standard",
    bot_appearance: str = "",
) -> Optional[str]:
    """
    将活动信息转换为完整的自拍 SD 提示词（自动自拍专用）

    使用 LLM 根据活动描述生成场景标签，LLM 失败时返回 None。

    Args:
        activity_info: 活动信息
        selfie_style: 自拍风格 ("standard"、"mirror" 或 "photo")
        bot_appearance: Bot 外观描述（从配置读取的 selfie.prompt_prefix）

    Returns:
        完整的 SD 提示词，LLM 失败时返回 None
    """
    # 使用 LLM 生成场景（传入风格以约束动作类型）
    scene = await generate_scene_with_llm(activity_info, selfie_style)
    if not scene:
        logger.warning("LLM 场景生成失败，取消本次自拍提示词生成")
        return None

    prompt_parts: List[str] = []

    # 1. 强制主体（含手部质量引导）
    prompt_parts.append("(1girl:1.4), (solo:1.3), (perfect hands:1.2), (correct anatomy:1.1)")

    # 2. Bot 外观
    if bot_appearance:
        prompt_parts.append(bot_appearance)

    # 3. 表情
    prompt_parts.append(f"({scene['expression']}:1.2)")

    # 4. 手部/身体动作
    hand_action = scene["hand_action"]

    # standard 自拍禁止手机类词汇
    if selfie_style == "standard" and hand_action:
        if re.search(r"\b(phone|smartphone|mobile|device)\b", hand_action, flags=re.IGNORECASE):
            hand_action = "resting head on hand"

    if hand_action:
        if selfie_style == "standard":
            hand_prompt = (
                f"(visible free hand {hand_action}:1.4), "
                "(only one hand visible in frame:1.5), "
                "(single hand gesture:1.3)"
            )
        elif selfie_style == "photo":
            # 第三人称照片：自然动作，不需要手部强调
            hand_prompt = f"({hand_action}:1.2)"
        else:
            hand_prompt = f"({hand_action}:1.3)"
        prompt_parts.append(hand_prompt)

    # 5. 环境
    prompt_parts.append(scene["environment"])

    # 6. 光线
    prompt_parts.append(scene["lighting"])

    # 7. 自拍风格（多种变体随机，增加多样性）
    selfie_scenes = _get_selfie_scene_for_style(selfie_style)
    prompt_parts.append(random.choice(selfie_scenes))

    # 8. 过滤空值、去重、拼接
    prompt_parts = [p for p in prompt_parts if p and p.strip()]
    keywords = [kw.strip() for kw in ", ".join(prompt_parts).split(",")]
    seen = set()
    unique = []
    for kw in keywords:
        kw_lower = kw.strip().lower()
        if kw_lower and kw_lower not in seen:
            seen.add(kw_lower)
            unique.append(kw.strip())

    final_prompt = ", ".join(unique)
    logger.info(f"生成自拍提示词: {final_prompt[:150]}...")
    return final_prompt


def get_negative_prompt_for_style(selfie_style: str, base_negative: str = "") -> str:
    """
    获取指定自拍风格的负面提示词

    Args:
        selfie_style: 自拍风格
        base_negative: 基础负面提示词（从配置读取）

    Returns:
        完整的负面提示词
    """
    parts = []
    if base_negative:
        parts.append(base_negative)

    # 所有风格都加手部质量负面提示词
    parts.append(SELFIE_HAND_NEGATIVE)

    # standard 额外加防双手拿手机
    if selfie_style == "standard":
        parts.append(ANTI_DUAL_PHONE_PROMPT)

    return ", ".join(parts)
