"""虚拟越线计数：每个 track_id 仅第一次有效穿越计一次（规则甲）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


def _side(px: float, py: float, x1: float, y1: float, x2: float, y2: float, eps: float = 1e-6) -> int:
    """点相对有向直线 (x1,y1)->(x2,y2) 的侧：+1 / -1 / 0（在线上）。"""
    v = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if abs(v) < eps:
        return 0
    return 1 if v > 0 else -1


@dataclass
class LineCounter:
    x1: float
    y1: float
    x2: float
    y2: float
    total_crossings: int = 0
    _last_side: Dict[int, int] = field(default_factory=dict)
    _counted_ids: Set[int] = field(default_factory=set)

    def update(self, track_ids: Iterable[int], centers: Iterable[Tuple[float, float]]) -> None:
        for tid, (cx, cy) in zip(track_ids, centers):
            s = _side(cx, cy, self.x1, self.y1, self.x2, self.y2)
            prev = self._last_side.get(tid)
            if prev is None:
                if s != 0:
                    self._last_side[tid] = s
                continue
            if s == 0:
                continue
            if prev * s < 0 and tid not in self._counted_ids:
                self.total_crossings += 1
                self._counted_ids.add(tid)
            self._last_side[tid] = s

    def reset(self) -> None:
        self.total_crossings = 0
        self._last_side.clear()
        self._counted_ids.clear()


def parse_line_arg(values: Optional[List[float]]) -> Optional[Tuple[float, float, float, float]]:
    """命令行传入 [x1,y1,x2,y2]；未传或长度不对则返回 None（不画线、不计数）。"""
    if not values:
        return None
    if len(values) != 4:
        raise ValueError("越线参数需要 4 个数：x1 y1 x2 y2（像素坐标）")
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])
