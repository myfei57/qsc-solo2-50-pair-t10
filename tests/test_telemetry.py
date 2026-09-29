"""趋势采样：实测/补齐来源标记与判定口径。"""

from __future__ import annotations

import pytest

from rubbercure.console.api import ConsoleAPI
from rubbercure.telemetry import (
    SOURCE_FILLED,
    SOURCE_MEASURED,
    Sample,
    TrendBuffer,
)


def test_appended_samples_are_marked_measured():
    trend = TrendBuffer(period_s=5.0)
    sample = trend.append("mold_temp", 0.0, 150.0)
    assert sample.source == SOURCE_MEASURED
    assert sample.filled is False
    assert sample.basis_ts is None
    assert sample.as_dict() == {
        "ts": 0.0,
        "value": 150.0,
        "source": "measured",
        "filled": False,
    }


def test_filled_points_carry_basis_and_previous_value():
    trend = TrendBuffer(period_s=5.0)
    trend.append("mold_temp", 0.0, 150.0)
    trend.append("mold_temp", 15.0, 152.0)

    added = trend.fill_gaps("mold_temp")
    assert added == 2
    samples = trend.window("mold_temp", 0.0, 15.0)
    assert [s.timestamp for s in samples] == [0.0, 5.0, 10.0, 15.0]
    first, gap1, gap2, last = samples
    assert not first.filled and not last.filled
    for gap in (gap1, gap2):
        assert gap.source == SOURCE_FILLED
        assert gap.filled is True
        assert gap.value == 150.0
        assert gap.basis_ts == 0.0
        assert gap.as_dict()["basis_ts"] == 0.0


def test_fill_gaps_is_idempotent_and_counts_filled_total():
    trend = TrendBuffer(period_s=5.0)
    trend.append("mold_temp", 0.0, 150.0)
    trend.append("mold_temp", 15.0, 152.0)
    assert trend.fill_gaps("mold_temp") == 2
    # 第二次调用不应再插入补齐点，返回的仍是序列里补齐点总数。
    assert trend.fill_gaps("mold_temp") == 2
    assert len(trend.window("mold_temp", 0.0, 15.0)) == 4


def test_consecutive_gaps_anchor_to_latest_measured_point():
    trend = TrendBuffer(period_s=5.0)
    trend.append("mold_temp", 0.0, 150.0)
    trend.append("mold_temp", 10.0, 160.0)  # 5.0 一个缺口
    trend.append("mold_temp", 25.0, 170.0)  # 15、20 两个缺口
    trend.fill_gaps("mold_temp")
    samples = trend.window("mold_temp", 0.0, 25.0)
    by_ts = {s.timestamp: s for s in samples}
    assert by_ts[5.0].basis_ts == 0.0
    assert by_ts[5.0].value == 150.0
    assert by_ts[15.0].basis_ts == 10.0
    assert by_ts[20.0].basis_ts == 10.0
    assert by_ts[15.0].value == by_ts[20.0].value == 160.0


def test_average_excludes_filled_points_by_default():
    trend = TrendBuffer(period_s=5.0)
    trend.append("mold_temp", 0.0, 150.0)
    trend.append("mold_temp", 10.0, 160.0)
    trend.fill_gaps("mold_temp")

    # 判定口径：只算 150 和 160 两个实测点，补齐的 150 不重复计入。
    assert trend.average("mold_temp", 0.0, 10.0) == pytest.approx(155.0)
    # 展示口径显式含补齐点：三个点两个 150、一个 160。
    assert trend.average("mold_temp", 0.0, 10.0, include_filled=True) == pytest.approx(
        153.3333, abs=1e-3
    )


def test_window_can_filter_by_source():
    trend = TrendBuffer(period_s=5.0)
    trend.append("mold_temp", 0.0, 150.0)
    trend.append("mold_temp", 10.0, 160.0)
    trend.fill_gaps("mold_temp")
    measured = trend.window("mold_temp", 0.0, 10.0, sources=(SOURCE_MEASURED,))
    assert [s.timestamp for s in measured] == [0.0, 10.0]
    assert all(not s.filled for s in measured)


def test_latest_measured_ignores_trail_of_filled_points():
    trend = TrendBuffer(period_s=5.0)
    trend.append("mold_temp", 0.0, 150.0)
    trend.append("mold_temp", 10.0, 160.0)
    trend.fill_gaps("mold_temp")
    # 末尾是补齐点时，判定方只能拿到最后一个实测点。
    assert trend.latest("mold_temp").timestamp == 10.0
    assert trend.latest_measured("mold_temp").timestamp == 10.0

    trend.append("mold_temp", 20.0, 162.0)
    assert trend.latest_measured("mold_temp").value == 162.0


def test_segments_split_measured_and_filled_runs():
    trend = TrendBuffer(period_s=5.0)
    trend.append("mold_temp", 0.0, 150.0)
    trend.append("mold_temp", 5.0, 151.0)
    trend.append("mold_temp", 20.0, 153.0)  # 10、15 缺口
    trend.fill_gaps("mold_temp")
    segments = trend.segments("mold_temp", 0.0, 20.0)
    assert segments == [
        {"source": "measured", "start": 0.0, "end": 5.0, "points": 2},
        {"source": "filled", "start": 10.0, "end": 15.0, "points": 2},
        {"source": "measured", "start": 20.0, "end": 20.0, "points": 1},
    ]


def test_sample_constructor_validates_source_and_basis():
    Sample(0.0, 1.0, source=SOURCE_MEASURED)
    with pytest.raises(ValueError):
        Sample(0.0, 1.0, source="guessed")
    with pytest.raises(ValueError):
        Sample(0.0, 1.0, source=SOURCE_FILLED)  # 缺 basis_ts
    with pytest.raises(ValueError):
        Sample(0.0, 1.0, source=SOURCE_MEASURED, basis_ts=0.0)
    measured = Sample.measured(0.0, 1.0)
    filled = Sample.make_filled(5.0, measured.timestamp, measured.value)
    assert filled.filled and filled.basis_ts == 0.0 and filled.value == 1.0


def test_capacity_trim_keeps_filled_basis_chain():
    trend = TrendBuffer(period_s=5.0, capacity=6)
    for index, value in enumerate([100, 110, 120, 130, 140, 150, 160, 170]):
        trend.append("mold_temp", float(index * 10), value)
    # 8 个实测点、每两个之间一个缺口；裁剪到最后 6 个点后头部是补齐点，
    # 它仍须带着被裁掉的实测点依据（ts=40, value=140）。
    assert trend.fill_gaps("mold_temp") == 3
    assert trend.fill_gaps("mold_temp") == 3  # 幂等
    samples = trend.window("mold_temp", 0.0, 1e9)
    assert len(samples) == 6
    assert samples[0].timestamp == 45.0
    assert samples[0].filled
    assert samples[0].basis_ts == 40.0
    assert samples[0].value == 140.0

    # 裁剪后继续补新缺口，仍锚定真实实测点而不是前一个补齐点。
    trend.append("mold_temp", 80.0, 180.0)
    trend.fill_gaps("mold_temp")
    tail = {s.timestamp: s for s in trend.window("mold_temp", 0.0, 1e9)}
    assert tail[75.0].filled
    assert tail[75.0].basis_ts == 70.0
    assert tail[75.0].value == 170.0


def test_trend_api_reports_segments_and_measured_only_average(platform):
    api = ConsoleAPI(platform)
    trend = platform.trend
    now = platform.clock.now()
    trend.append("mold_temp", now - 20.0, 150.0)
    trend.append("mold_temp", now - 10.0, 160.0)
    trend.append("mold_temp", now, 158.0)
    # 前两个实测点之间有一个 5 秒缺口
    payload = api.trend(series="mold_temp", window_s=60.0)["series"]["mold_temp"]

    assert payload["measured_points"] == 3
    assert payload["filled_points"] == 2
    sources = [s["source"] for s in payload["samples"]]
    assert sources == ["measured", "filled", "measured", "filled", "measured"]
    assert payload["average"] == round((150.0 + 160.0 + 158.0) / 3, 4)
    assert payload["average_including_filled"] == round(
        (150.0 + 150.0 + 160.0 + 160.0 + 158.0) / 5, 4
    )
    assert payload["latest_measured"]["source"] == "measured"
    run_kinds = [segment["source"] for segment in payload["segments"]]
    assert run_kinds == ["measured", "filled", "measured", "filled", "measured"]
    assert all(
        segment["points"] == 1
        for segment in payload["segments"]
        if segment["source"] == "filled"
    )
