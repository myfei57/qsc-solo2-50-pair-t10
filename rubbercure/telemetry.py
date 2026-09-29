"""趋势采样与补点。

模温与蒸汽压力的历史趋势必须连续：硫化曲线基线比对、过硫判定和机群调度
都要按时间窗读取趋势。现场总线抖动会丢点，趋势缓冲按固定周期回填（用前
一个有效值），并把补出来的点标记为 ``filled`` / ``source="filled"``。

补齐点只用于保持曲线连续，均值、最新实测值等后续判定只读取 ``measured``
采样，避免把前值保持出来的数据当成现场实测。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

SOURCE_MEASURED = "measured"
SOURCE_FILLED = "filled"


@dataclass(frozen=True)
class Sample:
    """一条趋势采样。"""

    timestamp: float
    value: float
    filled: bool = False

    @property
    def source(self) -> str:
        """采样来源：``measured`` 为实测，``filled`` 为补齐。"""

        return SOURCE_FILLED if self.filled else SOURCE_MEASURED

    @property
    def usable_for_judgement(self) -> bool:
        """该采样是否允许参与均值、最新值等后续工艺判定。"""

        return not self.filled

    def as_dict(self) -> dict:
        return {
            "ts": round(self.timestamp, 3),
            "value": round(self.value, 4),
            "source": self.source,
            "filled": self.filled,
            "usable_for_judgement": self.usable_for_judgement,
        }


class TrendBuffer:
    """按分量维护的定周期趋势缓冲。"""

    def __init__(self, period_s: float, capacity: int = 2048) -> None:
        if period_s <= 0:
            raise ValueError("采样周期必须为正")
        self._period = float(period_s)
        self._capacity = int(capacity)
        self._series: dict[str, list[Sample]] = {}
        self._lock = threading.RLock()

    def series_names(self) -> list[str]:
        with self._lock:
            return sorted(self._series)

    def append(self, name: str, timestamp: float, value: float) -> Sample:
        """追加实测点。"""

        sample = Sample(timestamp=float(timestamp), value=float(value))
        with self._lock:
            series = self._series.setdefault(name, [])
            series.append(sample)
            if len(series) > self._capacity:
                del series[: len(series) - self._capacity]
        return sample

    def fill_gaps(self, name: str) -> int:
        """在缺失采样周期的位置补点，返回补出的点数。"""

        with self._lock:
            series = self._series.get(name)
            if not series or len(series) < 2:
                return 0
            filled: list[Sample] = [series[0]]
            for current in series[1:]:
                previous = filled[-1]
                gap = current.timestamp - previous.timestamp
                steps = int(round(gap / self._period)) - 1
                for step in range(1, steps + 1):
                    filled.append(
                        Sample(
                            timestamp=previous.timestamp + step * self._period,
                            value=previous.value,
                            filled=True,
                        )
                    )
                filled.append(current)
            added = len(filled) - len(series)
            self._series[name] = filled[-self._capacity :]
        return added

    def window(
        self, name: str, start: float, end: float, include_filled: bool = True
    ) -> list[Sample]:
        """返回 ``[start, end]`` 区间内的采样。

        ``include_filled=False`` 时只返回实测点，供后续工艺判定使用。
        """

        with self._lock:
            series = list(self._series.get(name, ()))
        return [
            sample
            for sample in series
            if start <= sample.timestamp <= end
            and (include_filled or not sample.filled)
        ]

    def latest(self, name: str) -> Sample | None:
        with self._lock:
            series = self._series.get(name)
            return series[-1] if series else None

    def latest_measured(self, name: str) -> Sample | None:
        """返回最新实测点；最新补点不参与判定。"""

        with self._lock:
            series = self._series.get(name, ())
            for sample in reversed(series):
                if not sample.filled:
                    return sample
        return None

    def average(self, name: str, start: float, end: float) -> float | None:
        """区间实测均值；区间内无实测点返回 ``None``。"""

        samples = self.window(name, start, end, include_filled=False)
        if not samples:
            return None
        return sum(sample.value for sample in samples) / len(samples)

    def snapshot(self) -> dict:
        """转成控制台趋势接口使用的结构。"""

        with self._lock:
            return {
                name: [sample.as_dict() for sample in samples]
                for name, samples in self._series.items()
            }

    def origin_segments(self, name: str, start: float, end: float) -> list[dict]:
        """按连续来源合并窗口内采样，返回实测段／补齐段。"""

        segments: list[dict] = []
        for sample in self.window(name, start, end):
            if segments and segments[-1]["source"] == sample.source:
                segments[-1]["end_ts"] = round(sample.timestamp, 3)
                segments[-1]["count"] += 1
            else:
                segments.append(
                    {
                        "source": sample.source,
                        "filled": sample.filled,
                        "usable_for_judgement": sample.usable_for_judgement,
                        "start_ts": round(sample.timestamp, 3),
                        "end_ts": round(sample.timestamp, 3),
                        "count": 1,
                    }
                )
        return segments
