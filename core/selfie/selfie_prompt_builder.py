"""
自拍提示词构建器（Command 与 Action 共用）

供 /dr 来张自拍 等 Command 路径使用，确保与 Action 自拍流程输出一致的提示词
（含 SELFIE_OUTFIT_VARIANTS、手部动作、场景等）
"""

import random
from typing import Callable, Optional, Tuple

from ..utils import (
    SELFIE_HAND_NEGATIVE, ANTI_DUAL_PHONE_PROMPT,
    SELFIE_OUTFIT_VARIANTS, SELFIE_OUTFIT_NEGATIVE,
)
from ..pic_action import MaisArtAction


def _get_selfie_scene_variants(selfie_style: str) -> list:
    """返回自拍风格的场景变体列表"""
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
    return [
        "selfie, front camera view, arm extended, looking at camera",
        "selfie, front facing camera, POV selfie, slight high angle, upper body",
        "selfie, front camera, centered composition, cowboy shot, looking at lens",
    ]


def _get_hand_actions_for_style(selfie_style: str) -> list:
    """根据自拍风格返回手部动作池"""
    return MaisArtAction._get_hand_actions_for_style(selfie_style)


async def build_selfie_prompt(
    description: str,
    selfie_style: str,
    config_getter: Callable[[str, any], any],
    activity_scene: Optional[dict] = None,
    outfit: str = "",
    free_hand_action: str = "",
) -> Tuple[str, str]:
    """构建自拍模式完整提示词（与 pic_action._process_selfie_prompt 逻辑一致）

    Args:
        description: 用户描述（如「来张自拍」），用于 LLM 生成手部动作或作为场景补充
        selfie_style: standard/mirror/photo/cosplay
        config_getter: (key, default) -> value，如 plugin.get_config
        activity_scene: 日程活动场景（可选）
        outfit: LLM 设计的服装（可选）
        free_hand_action: LLM 生成的手部动作（可选）

    Returns:
        (prompt, negative_prompt)
    """
    forced_subject = "(1girl:1.4), (solo:1.3), (perfect hands:1.2), (correct anatomy:1.1)"

    if selfie_style == "cosplay":
        bot_appearance = random.choice(MaisArtAction._COSPLAY_CHARACTERS)
    else:
        bot_appearance = (config_getter("selfie.prompt_prefix", "") or "").strip()

    selfie_scenes = _get_selfie_scene_variants(selfie_style)
    selfie_scene = random.choice(selfie_scenes)

    if free_hand_action:
        hand_action = free_hand_action
    elif activity_scene and activity_scene.get("hand_action"):
        hand_action = activity_scene["hand_action"]
    else:
        hand_action = None
        desc_clean = description.strip().strip(",. 、，。")
        desc_long_enough = len(desc_clean) > 3 if any('\u4e00' <= c <= '\u9fff' for c in desc_clean) else len(desc_clean) > 6
        if desc_long_enough:
            try:
                from .scene_action_generator import generate_hand_action_with_llm
                hand_action = await generate_hand_action_with_llm(description, selfie_style)
            except Exception:
                pass
        if not hand_action:
            hand_action = random.choice(_get_hand_actions_for_style(selfie_style))

    prompt_parts = [forced_subject]
    if bot_appearance:
        prompt_parts.append(bot_appearance)
    if outfit:
        prompt_parts.append(outfit)
    elif selfie_style != "cosplay":
        prompt_parts.append(random.choice(SELFIE_OUTFIT_VARIANTS))

    if activity_scene:
        if activity_scene.get("expression"):
            prompt_parts.append(f"({activity_scene['expression']}:1.2)")
        if activity_scene.get("lighting"):
            prompt_parts.append(activity_scene["lighting"])

    prompt_parts.append(hand_action)
    if activity_scene and activity_scene.get("environment"):
        prompt_parts.append(activity_scene["environment"])
    prompt_parts.append(selfie_scene)
    # 通用自拍短词用英文场景补充，避免中文干扰 SD/RunningHub
    scene_desc = description.strip()
    generic_selfie = scene_desc in ("来张自拍", "自拍", "拍照", "照片", "看看你", "想看你", "来张图", "发张图") or (
        len(scene_desc) <= 8 and any(k in scene_desc for k in ("自拍", "拍照", "照片", "看看", "想看你"))
    )
    if generic_selfie:
        scene_desc = "casual selfie, natural lighting, soft focus"
    prompt_parts.append(scene_desc)

    final_prompt = ", ".join(p for p in prompt_parts if p)
    keywords = [kw.strip() for kw in final_prompt.split(",")]
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
    final_prompt = ", ".join(unique_keywords)

    base_negative = (config_getter("selfie.negative_prompt", "") or "").strip()
    negative_parts = []
    if base_negative:
        negative_parts.append(base_negative)
    negative_parts.append(SELFIE_HAND_NEGATIVE)
    if selfie_style != "cosplay":
        negative_parts.append(SELFIE_OUTFIT_NEGATIVE)
    if selfie_style == "standard":
        negative_parts.append(ANTI_DUAL_PHONE_PROMPT)
    negative_prompt = ", ".join(negative_parts)

    return final_prompt, negative_prompt
