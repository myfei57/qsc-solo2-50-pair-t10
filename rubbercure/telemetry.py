"""趋势采样与补点。

模温与蒸汽压力的历史趋势必须连续：硫化曲线基线比对、过硫判定和机群调度
都要按时间窗读取趋势。现场总线抖动会丢点，趋势缓冲按固定周期回填（用前
一个有效值）。每个采样必须带来源标记，报表与下游判定据此区分：

* ``measured``：现场总线上真实采到的点；
* ``filled``：按周期回填的点，值沿用上一个实测点，并记录该实测点的时间戳。

补齐点只用于让曲线连续，绝不能当实测点参与均值等后续判定——因此默认的
窗口聚合只统计 ``measured`` 点，需要含补齐点的展示口径必须显式要求。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

SOURCE_MEASURED = "measured"
SOURCE_FILLED = "filled"


@dataclass(frozen=True)
class Sample:
    """一条趋势采样。

    ``source`` 标明采样来源；补齐点额外用 ``basis_ts`` 记录取值所依据的
    那个实测点时间戳，方便在报表上追溯每一个补齐值的出处。``is_filled``
    是由 ``source`` 派生的布尔标志，避免调用方反复比较字符串。
    """

    timestamp: float
    value: float
    source: str = SOURCE_MEASURED
    basis_ts: float | None = None
    is_filled: bool = False

    def __post_init__(self) -> None:
        if self.source not in (SOURCE_MEASURED, SOURCE_FILLED):
            raise ValueError(f"未知的采样来源: {self.source}")
        if self.source == SOURCE_FILLED and self.basis_ts is None:
            raise ValueError("补齐点必须记录取值依据的实测点时间戳")
        if self.source == SOURCE_MEASURED and self.basis_ts is not None:
            raise ValueError("实测点不能携带补齐依据时间戳")
        # 标志位只允许由来源派生，杜绝构造时自相矛盾的标记。
        object.__setattr__(self, "is_filled", self.source == SOURCE_FILLED)

    @property
    def filled(self) -> bool:
        """是否为补齐点（语义化别名）。"""

        return self.is_filled

    @classmethod
    def measured(cls, timestamp: float, value: float) -> "Sample":
        return cls(timestamp=float(timestamp), value=float(value), source=SOURCE_MEASURED)

    @classmethod
    def make_filled(cls, timestamp: float, basis_ts: float, value: float) -> "Sample":
        """以一个实测点的时间戳与取值为依据构造补齐点。"""

        return cls(
            timestamp=float(timestamp),
            value=float(value),
            source=SOURCE_FILLED,
            basis_ts=float(basis_ts),
        )

    def as_dict(self) -> dict:
        payload = {
            "ts": round(self.timestamp, 3),
            "value": round(self.value, 4),
            "source": self.source,
            "filled": self.is_filled,
        }
        if self.basis_ts is not None:
            payload["basis_ts"] = round(self.basis_ts, 3)
        return payload


class TrendBuffer:
    """按分量维护的定周期趋势缓冲。"""

    def __init__(self, period_s: float, capacity: int = 2048) -> None:
        if period_s <= 0:
            raise ValueError("采样周期必须为正")
        self._period = float(period_s)
        self._capacity = int(capacity)
        self._series: dict[str, list[Sample]] = {}
        self._lock = threading.RLock()

    @property
    def period_s(self) -> float:
        return self._period

    def series_names(self) -> list[str]:
        with self._lock:
            return sorted(self._series)

    def append(self, name: str, timestamp: float, value: float) -> Sample:
        """追加实测点。"""

        sample = Sample.measured(timestamp=timestamp, value=value)
        with self._lock:
            series = self._series.setdefault(name, [])
            series.append(sample)
            if len(series) > self._capacity:
                del series[: len(series) - self._capacity]
        return sample

    def fill_gaps(self, name: str) -> int:
        """在缺失采样周期的位置补点，返回当前整条序列里补齐点总数。

        补齐点插在两个实测点之间，取值沿用上一个实测点；重复调用幂等：
        已经补齐过的缺口不会再插点。
        """

        with self._lock:
            series = self._series.get(name)
            if not series or len(series) < 2:
                return 0
            # 遍历原始序列时始终携带"最近实测点"的时间戳与取值：新补点
            # 一律锚定它，而不是锚定序列里的前一个点（前一个点可能本身
            # 就是补齐点）。容量裁剪后若头部只剩补齐点，或遍历时遇到既有
            # 补齐点（重复补点的幂等情形），就沿它们记录的 basis 继续传递。
            first = series[0]
            basis_ts = first.basis_ts if first.filled else first.timestamp
            basis_value = first.value
            merged: list[Sample] = [first]
            for current in series[1:]:
                previous = merged[-1]
                gap = current.timestamp - previous.timestamp
                steps = int(round(gap / self._period)) - 1
                for step in range(1, steps + 1):
                    merged.append(
                        Sample.make_filled(
                            timestamp=previous.timestamp + step * self._period,
                            basis_ts=basis_ts,
                            value=basis_value,
                        )
                    )
                merged.append(current)
                if current.filled:
                    basis_ts = current.basis_ts
                else:
                    basis_ts = current.timestamp
                basis_value = current.value
            self._series[name] = merged[-self._capacity :]
            return sum(1 for sample in self._series[name] if sample.filled)

    def window(
        self,
        name: str,
        start: float,
        end: float,
        sources: tuple[str, ...] = (SOURCE_MEASURED, SOURCE_FILLED),
    ) -> list[Sample]:
        """返回 ``[start, end]`` 区间内、来源符合要求的采样。"""

        with self._lock:
            series = list(self._series.get(name, ()))
        return [
            sample
            for sample in series
            if sample.source in sources and start <= sample.timestamp <= end
        ]

    def latest(self, name: str) -> Sample | None:
        """返回最新一个采样（可能是补齐点，仅用于展示）。"""

        with self._lock:
            series = self._series.get(name)
            return series[-1] if series else None

    def latest_measured(self, name: str) -> Sample | None:
        """返回最新一个实测点；下游判定只允许读这个口径。"""

        with self._lock:
            series = self._series.get(name)
            if not series:
                return None
            for sample in reversed(series):
                if not sample.filled:
                    return sample
        return None

    def average(self, name: str, start: float, end: float, include_filled: bool = False) -> float:
        """区间均值；区间内无可用点返回 0.0。

        默认只统计实测点，补齐点不参与判定口径；展示层确需含补齐点的
        连续曲线均值时，必须显式传 ``include_filled=True``。
        """

        sources = (
            (SOURCE_MEASURED, SOURCE_FILLED) if include_filled else (SOURCE_MEASURED,)
        )
        samples = self.window(name, start, end, sources=sources)
        if not samples:
            return 0.0
        return sum(sample.value for sample in samples) / len(samples)

    def segments(self, name: str, start: float, end: float) -> list[dict]:
        """把窗口内的采样按来源切成连续段，供报表分段标注。

        每段给出来源、起止时间戳与点数，相邻同类来源合并为一段。
        """

        samples = self.window(name, start, end)
        runs: list[dict] = []
        for sample in samples:
            if runs and runs[-1]["source"] == sample.source:
                runs[-1]["end"] = round(sample.timestamp, 3)
                runs[-1]["points"] += 1
                continue
            runs.append(
                {
                    "source": sample.source,
                    "start": round(sample.timestamp, 3),
                    "end": round(sample.timestamp, 3),
                    "points": 1,
                }
            )
        return runs

    def snapshot(self) -> dict:
        """转成控制台趋势接口使用的结构。"""

        with self._lock:
            return {
                name: [sample.as_dict() for sample in samples]
                for name, samples in self._series.items()
            }
