"""
自拍提示词构建器（Command 与 Action 共用）

供 /dr 来张自拍 等 Command 路径使用。
非 cosplay 模式直接使用 SELFIE_OUTFIT_VARIANTS 中的完整中文描述模板作为 prompt，
确保高质量图片输出，不过滤 NSFW 内容。
"""

import random
from typing import Callable, Optional, Tuple

from ..utils import (
    SELFIE_HAND_NEGATIVE,
    SELFIE_OUTFIT_VARIANTS,
)
from ..pic_action import MaisArtAction

# 按自拍风格追加的中文场景修饰语
_STYLE_SUFFIX = {
    "mirror": "，对着镜子自拍，手机镜头可见，镜中映出倒影",
    "photo": "，由旁人拍摄，构图自然，全身或半身入镜",
}


async def build_selfie_prompt(
    description: str,
    selfie_style: str,
    config_getter: Callable[[str, any], any],
    activity_scene: Optional[dict] = None,
    outfit: str = "",
    free_hand_action: str = "",
) -> Tuple[str, str]:
    """构建自拍模式完整提示词

    非 cosplay 模式：直接选取 SELFIE_OUTFIT_VARIANTS 中的完整中文描述模板，
    按需追加风格修饰与活动场景，不拼接英文 tag。

    Args:
        description: 用户描述（如「来张自拍」）
        selfie_style: standard/mirror/photo/cosplay
        config_getter: (key, default) -> value，如 plugin.get_config
        activity_scene: 日程活动场景（可选）
        outfit: 自定义服装描述（可选，非空时替换模板）
        free_hand_action: 自定义手部动作（可选，追加到结尾）

    Returns:
        (prompt, negative_prompt)
    """
    # cosplay 模式：使用经典动漫角色形象，保持原有英文 SD tag 逻辑
    if selfie_style == "cosplay":
        prompt = random.choice(MaisArtAction._COSPLAY_CHARACTERS)
        if free_hand_action:
            prompt += f"，{free_hand_action}"
        return prompt, SELFIE_HAND_NEGATIVE

    # 非 cosplay 模式：使用完整中文模板
    if outfit:
        prompt = outfit
    else:
        prompt = random.choice(SELFIE_OUTFIT_VARIANTS)

    # 追加风格修饰
    suffix = _STYLE_SUFFIX.get(selfie_style, "")
    if suffix:
        prompt = prompt + suffix

    # 追加活动场景（中文）
    if activity_scene and activity_scene.get("environment"):
        prompt = prompt + f"，场景：{activity_scene['environment']}"

    # 追加自定义手部动作
    if free_hand_action:
        prompt = prompt + f"，{free_hand_action}"

    negative = SELFIE_HAND_NEGATIVE
    return prompt, negative
