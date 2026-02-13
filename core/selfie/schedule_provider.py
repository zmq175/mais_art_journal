"""
日程适配层

提供统一的日程/活动信息接口：
- PlanningPluginProvider: 读取 autonomous_planning 插件的 SQLite 数据库
- get_schedule_provider(): 工厂函数，自动选择可用的 provider
"""

import datetime
import json
import os
import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

from src.common.logger import get_logger

logger = get_logger("auto_selfie.schedule")


class ActivityType(Enum):
    """活动类型枚举"""
    SLEEPING = "sleeping"
    WAKING_UP = "waking_up"
    EATING = "eating"
    WORKING = "working"
    STUDYING = "studying"
    EXERCISING = "exercising"
    RELAXING = "relaxing"
    SOCIALIZING = "socializing"
    COMMUTING = "commuting"
    HOBBY = "hobby"
    SELF_CARE = "self_care"
    OTHER = "other"


@dataclass
class ActivityInfo:
    """活动信息数据类 - 统一的活动描述格式"""
    activity_type: ActivityType
    description: str          # 活动描述（中文）
    mood: str = "neutral"     # 情绪
    time_point: str = ""      # 时间点 "HH:MM"


class ScheduleProvider:
    """日程提供者基类"""

    async def get_current_activity(self) -> Optional[ActivityInfo]:
        """获取当前时间对应的活动信息"""
        raise NotImplementedError


class PlanningPluginProvider(ScheduleProvider):
    """
    从 autonomous_planning 插件的 SQLite 数据库读取日程

    goals 表结构关键字段：
    - goal_id, name, description, goal_type, status, created_at
    - parameters (JSON): 包含 time_window: [start_minutes, end_minutes]

    查询逻辑：
    1. 优先获取今天的 active 目标
    2. 匹配 time_window 包含当前时间的条目
    3. 无精确匹配则返回最新记录
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        logger.info(f"PlanningPluginProvider 初始化, db: {db_path}")

    async def get_current_activity(self) -> Optional[ActivityInfo]:
        try:
            if not os.path.exists(self.db_path):
                return None

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            now = datetime.datetime.now()
            current_time_str = now.strftime("%H:%M")
            today_str = now.strftime("%Y-%m-%d")
            current_minutes = now.hour * 60 + now.minute

            # 优先获取今天的活跃目标
            # 用 substr 截取 ISO 日期前10字符，避免 date() 做 UTC 时区转换
            rows = []
            try:
                cursor.execute("""
                    SELECT * FROM goals
                    WHERE status = 'active'
                    AND substr(created_at, 1, 10) = ?
                    ORDER BY created_at DESC
                    LIMIT 20
                """, (today_str,))
                rows = cursor.fetchall()
            except Exception:
                pass

            # 今天没有结果，回退到最新的活跃记录
            if not rows:
                logger.debug("今天没有活跃目标，回退到最近的活跃记录")
                cursor.execute("""
                    SELECT * FROM goals
                    WHERE status = 'active'
                    ORDER BY created_at DESC
                    LIMIT 20
                """)
                rows = cursor.fetchall()

            conn.close()

            if not rows:
                return None

            # 尝试匹配当前时间窗口（time_window 在 parameters JSON 中）
            for row in rows:
                row_dict = dict(row)
                time_window = self._extract_time_window(row_dict)

                if time_window and len(time_window) == 2:
                    start_min, end_min = int(time_window[0]), int(time_window[1])
                    if self._is_minutes_in_range(current_minutes, start_min, end_min):
                        return self._row_to_activity(row_dict, current_time_str)

            # 如果没有精确匹配，返回最新的活跃记录
            first = dict(rows[0])
            return self._row_to_activity(first, current_time_str)

        except Exception as e:
            logger.error(f"PlanningPluginProvider 查询失败: {e}")
            return None

    @staticmethod
    def _extract_time_window(row: dict) -> Optional[List[int]]:
        """从 parameters JSON 中提取 time_window

        goals 表的 parameters 列存储为 JSON 文本，
        time_window 格式为 [start_minutes, end_minutes]
        """
        params_raw = row.get("parameters")
        if not params_raw:
            return None
        try:
            params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
            return params.get("time_window")
        except (json.JSONDecodeError, TypeError):
            return None

    def _row_to_activity(self, row: dict, current_time: str) -> ActivityInfo:
        """将数据库行转换为 ActivityInfo"""
        description = row.get("description", "") or row.get("name", "") or "日常活动"
        goal_type = (row.get("goal_type", "") or "").lower()

        # 类型映射（覆盖 autonomous_planning 常用的 goal_type 值）
        type_map = {
            # 英文关键词
            "work": ActivityType.WORKING,
            "study": ActivityType.STUDYING,
            "exercise": ActivityType.EXERCISING,
            "eat": ActivityType.EATING,
            "meal": ActivityType.EATING,
            "rest": ActivityType.RELAXING,
            "relax": ActivityType.RELAXING,
            "social": ActivityType.SOCIALIZING,
            "hobby": ActivityType.HOBBY,
            "sleep": ActivityType.SLEEPING,
            "self_care": ActivityType.SELF_CARE,
            "commut": ActivityType.COMMUTING,
            # 中文关键词
            "工作": ActivityType.WORKING,
            "办公": ActivityType.WORKING,
            "会议": ActivityType.WORKING,
            "学习": ActivityType.STUDYING,
            "阅读": ActivityType.STUDYING,
            "读书": ActivityType.STUDYING,
            "审阅": ActivityType.STUDYING,
            "看书": ActivityType.STUDYING,
            "研究": ActivityType.STUDYING,
            "运动": ActivityType.EXERCISING,
            "锻炼": ActivityType.EXERCISING,
            "健身": ActivityType.EXERCISING,
            "散步": ActivityType.EXERCISING,
            "吃": ActivityType.EATING,
            "餐": ActivityType.EATING,
            "料理": ActivityType.EATING,
            "烹饪": ActivityType.EATING,
            "休息": ActivityType.RELAXING,
            "放松": ActivityType.RELAXING,
            "泡澡": ActivityType.RELAXING,
            "泡浴": ActivityType.RELAXING,
            "聊天": ActivityType.SOCIALIZING,
            "交流": ActivityType.SOCIALIZING,
            "社交": ActivityType.SOCIALIZING,
            "睡": ActivityType.SLEEPING,
            "梦": ActivityType.SLEEPING,
            "入眠": ActivityType.SLEEPING,
            "午休": ActivityType.SLEEPING,
            "小憩": ActivityType.SLEEPING,
            "梳妆": ActivityType.SELF_CARE,
            "打扮": ActivityType.SELF_CARE,
            "化妆": ActivityType.SELF_CARE,
            "护肤": ActivityType.SELF_CARE,
            "通勤": ActivityType.COMMUTING,
            "赶路": ActivityType.COMMUTING,
            "出行": ActivityType.COMMUTING,
        }
        activity_type = ActivityType.OTHER
        for key, atype in type_map.items():
            if key in goal_type or key in description.lower():
                activity_type = atype
                break

        return ActivityInfo(
            activity_type=activity_type,
            description=description,
            mood="neutral",
            time_point=current_time,
        )

    @staticmethod
    def _is_minutes_in_range(current: int, start: int, end: int) -> bool:
        """检查分钟数是否在范围内（支持跨午夜）"""
        if end < start:
            return current >= start or current <= end
        return start <= current <= end


def get_schedule_provider(
    planning_db_search_dirs: Optional[list] = None,
) -> Optional[ScheduleProvider]:
    """
    工厂函数：查找 autonomous_planning 数据库并返回 provider

    找不到数据库时返回 None，由调用方决定如何处理。

    Args:
        planning_db_search_dirs: 搜索 autonomous_planning 数据库的目录列表

    Returns:
        PlanningPluginProvider 实例，或 None
    """
    if planning_db_search_dirs is None:
        # __file__ = plugins/mais_art_journal/core/selfie/schedule_provider.py
        # 需要回到 plugins/ 目录（4层 dirname）
        plugins_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        planning_db_search_dirs = [
            os.path.join(plugins_dir, "autonomous_planning_plugin"),
            os.path.join(plugins_dir, "autonomous_planning"),
        ]

    for search_dir in planning_db_search_dirs:
        if not os.path.isdir(search_dir):
            continue
        # 搜索插件根目录和 data/ 子目录（GoalManager 默认把 db 放在 data/ 下）
        check_dirs = [search_dir, os.path.join(search_dir, "data")]
        for check_dir in check_dirs:
            if not os.path.isdir(check_dir):
                continue
            for fname in os.listdir(check_dir):
                if fname.endswith((".db", ".sqlite", ".sqlite3")):
                    db_path = os.path.join(check_dir, fname)
                    try:
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='goals'")
                        if cursor.fetchone():
                            conn.close()
                            logger.info(f"找到 autonomous_planning 数据库: {db_path}")
                            return PlanningPluginProvider(db_path)
                        conn.close()
                    except Exception:
                        pass

    logger.warning("未找到 autonomous_planning 数据库")
    return None
