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
from ..utils import SELFIE_HAND_NEGATIVE, SELFIE_OUTFIT_VARIANTS

logger = get_logger("auto_selfie.scene")


# ==================== 确定性映射（手动自拍 + LLM 兜底） ====================
# 每个活动类型有多个变体，随机选择以增加多样性（参考 Seedream 提示词指南）

# 活动类型到动作的映射（每种类型多个变体）
ACTIVITY_ACTIONS: Dict[str, List[str]] = {
    "sleeping": ["侧躺抱枕，蜷缩慵懒", "裹紧毛毯，安静入眠", "枕着手臂睡着，安详表情"],
    "waking_up": ["伸懒腰打哈欠，发型微乱", "揉眼睛半睁，刚醒神态", "坐在床上伸展双臂"],
    "eating": ["手持筷子夹菜", "端起杯子轻抿", "咬一口食物，满足表情"],
    "working": ["敲击键盘，专注神情", "盯着屏幕，若有所思", "托腮思考，眼神专注"],
    "studying": ["手持书本阅读", "提笔做笔记，认真", "手指划过书页，专注"],
    "exercising": ["拉伸热身，精力充沛，手持水瓶", "做准备活动，充满活力", "放松冷却，姿态舒缓"],
    "relaxing": ["窝在沙发上听音乐，慵懒", "蜷缩进毛毯里", "舒适地盘腿而坐"],
    "socializing": ["比V字手势欢笑", "挥手打招呼，开心", "双手交握，兴奋微笑"],
    "commuting": ["手提包，戴耳机漫步", "站立抓住扶手", "等待，四处张望"],
    "hobby": ["手持相机，创意十足", "专注手工，神情投入", "手持道具，全神贯注"],
    "self_care": ["对着镜子化妆", "护肤中，轻柔动作", "整理发型，看着镜子"],
    "other": ["站立自然姿势", "轻松站姿，舒适", "微微侧身，随性"],
}

# 活动类型到场景环境的映射（多样变体）
ACTIVITY_ENVIRONMENTS: Dict[str, List[str]] = {
    "sleeping": ["卧室，暖色夜灯，温馨氛围", "柔软枕头，暖色毛毯", "安静房间，夜晚氛围"],
    "waking_up": ["卧室，清晨阳光，窗帘半开", "窗前，柔和晨光", "晨曦氛围，光线轻柔"],
    "eating": ["餐厅，桌上摆好餐具", "温馨咖啡厅，暖色内饰", "厨房吧台，随性用餐"],
    "working": ["办公桌，屏幕背光", "书房，整洁布置", "工作台，干净背景"],
    "studying": ["图书馆，书架，台灯", "书桌，暖色台灯", "安静角落，书籍环绕"],
    "exercising": ["健身房，器械背景", "家中运动区", "户外公园，绿色背景"],
    "relaxing": ["客厅，沙发，午后阳光", "温馨角落，软垫", "阳台，城市远景"],
    "socializing": ["户外咖啡厅，明亮氛围", "餐厅，暖色灯光", "公园长椅，自然风景"],
    "commuting": ["城市街道，都市感", "地铁站台，柔和光线", "公交站，清晨场景"],
    "hobby": ["艺术工作室，创意空间", "工坊，有序陈列", "创意角落，充满灵感"],
    "self_care": ["浴室，镜子，梳妆台", "梳妆台，柔和灯光", "化妆区，清洁美感"],
    "other": ["室内，自然采光", "随性场景，柔和虚化", "中性背景，干净简洁"],
}

# 活动类型到表情的映射（多样变体）
ACTIVITY_EXPRESSIONS: Dict[str, List[str]] = {
    "sleeping": ["安详表情，眼睛闭合", "宁静，安睡中", "满足的微笑，梦境中"],
    "waking_up": ["迷糊表情，眼睛半开", "睡眼惺忪，淡淡微笑", "柔和表情，刚刚醒来"],
    "eating": ["开心表情，享受美食", "满足微笑", "愉快，细细品味"],
    "working": ["专注表情，神情认真", "若有所思，眼神投入", "坚定，全情投入"],
    "studying": ["专注，若有所思", "沉浸其中，好奇", "认真，求知若渴"],
    "exercising": ["充满活力，坚定", "灿烂笑容，斗志昂扬", "健康光泽，运动感"],
    "relaxing": ["放松微笑，惬意", "平静，舒适自在", "温柔表情，舒缓"],
    "socializing": ["灿烂微笑，开心", "大笑，喜悦", "温暖表情，友善"],
    "commuting": ["平静表情", "宁静，若有所思", "放松，环顾四周"],
    "hobby": ["兴奋，充满热情", "投入，全神贯注", "眼神发光，充满灵感"],
    "self_care": ["温柔微笑，美好时光", "宁静，宠爱自己", "柔和表情，放松"],
    "other": ["自然微笑", "温柔表情", "亲切随和"],
}

# 活动类型到光线的映射（多样变体）
ACTIVITY_LIGHTING: Dict[str, List[str]] = {
    "sleeping": ["暖色夜灯，昏黄", "柔和环境光", "月光，淡淡阴影"],
    "waking_up": ["柔和晨光，金色时刻", "温暖日出，轻柔光线", "黎明光线，柔和漫射"],
    "eating": ["暖色室内灯光", "柔和顶灯", "烛光，温馨"],
    "working": ["办公灯光，均匀照明", "台灯，集中光线", "窗边自然光"],
    "studying": ["台灯，聚焦光线", "暖色学习灯", "柔和阅读光"],
    "exercising": ["明亮自然光", "健身房灯光，充满活力", "户外阳光，动感"],
    "relaxing": ["柔和午后光线，暖色环境光", "黄金时刻，温馨", "斑驳光影，安静"],
    "socializing": ["明亮欢快灯光", "暖色咖啡厅灯", "自然日光，生动"],
    "commuting": ["清晨阳光", "柔和都市光线", "阴天，漫射光"],
    "hobby": ["创意工作室灯光", "自然光，充满灵感", "暖色台灯，聚焦"],
    "self_care": ["浴室灯光，镜面反射", "梳妆台灯，柔和", "自然镜前光"],
    "other": ["自然采光", "柔和漫射光", "均衡照明"],
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

    "cosplay": """
7. STYLE CONSTRAINT - Cosplay photo: character cosplay pose. Both hands are FREE. Action should fit anime character style (e.g. confident pose, peace sign, hand on hip, character stance, holding prop). Prefer poses that match the character's personality.""",
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
    if selfie_style == "cosplay":
        return [
            "cosplay photo, anime character cosplay, convention style, detailed costume",
            "anime cosplay portrait, character accurate, high quality cosplay",
            "cosplay selfie, anime style, professional cosplay photo",
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
    将活动信息转换为完整的自拍提示词（自动自拍专用）

    非 cosplay 模式：直接从 SELFIE_OUTFIT_VARIANTS 中选取完整中文描述模板，
    追加活动描述作为情境补充，不再调用 LLM 生成场景。
    cosplay 模式：保持原有英文 SD tag 逻辑，LLM 场景生成失败时返回 None。

    Args:
        activity_info: 活动信息
        selfie_style: 自拍风格 ("standard"、"mirror"、"photo" 或 "cosplay")
        bot_appearance: Bot 外观描述（从配置读取的 selfie.prompt_prefix，非 cosplay 时忽略）

    Returns:
        完整提示词，异常时返回 None
    """
    # cosplay 模式：直接使用完整中文角色描述模板
    if selfie_style == "cosplay":
        from ..pic_action import MaisArtAction
        prompt = random.choice(MaisArtAction._COSPLAY_CHARACTERS)
        if activity_info and activity_info.description:
            prompt += f"，她刚{activity_info.description}回来"
        logger.info(f"生成 cosplay 自拍提示词: {prompt[:150]}...")
        return prompt

    # 非 cosplay 模式：完整中文模板 + 活动情境追加
    prompt = random.choice(SELFIE_OUTFIT_VARIANTS)

    # 按风格追加修饰
    style_suffix = {
        "mirror": "，对着镜子自拍，手机镜头可见，镜中映出倒影",
        "photo": "，由旁人拍摄，构图自然，全身或半身入镜",
    }
    suffix = style_suffix.get(selfie_style, "")
    if suffix:
        prompt = prompt + suffix

    # 追加活动描述作为情境补充
    if activity_info and activity_info.description:
        prompt = prompt + f"，她刚{activity_info.description}回来"

    logger.info(f"生成自拍提示词: {prompt[:150]}...")
    return prompt


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

    parts.append(SELFIE_HAND_NEGATIVE)

    return ", ".join(parts)
