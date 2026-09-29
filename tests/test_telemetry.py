"""趋势采样来源与补齐点隔离。"""

from __future__ import annotations

from rubbercure.console.api import ConsoleAPI
from rubbercure.telemetry import TrendBuffer


def test_samples_are_marked_with_origin_and_average_uses_measured_only():
    trend = TrendBuffer(period_s=5.0)
    trend.append("mold_temp", 0.0, 100.0)
    trend.append("mold_temp", 15.0, 120.0)

    assert trend.fill_gaps("mold_temp") == 2
    samples = trend.window("mold_temp", 0.0, 15.0)

    assert [sample.source for sample in samples] == [
        "measured",
        "filled",
        "filled",
        "measured",
    ]
    assert samples[0].as_dict()["filled"] is False
    assert samples[1].as_dict()["filled"] is True
    assert samples[0].as_dict()["usable_for_judgement"] is True
    assert samples[1].as_dict()["usable_for_judgement"] is False
    assert trend.average("mold_temp", 0.0, 15.0) == 110.0
    assert trend.window("mold_temp", 0.0, 15.0, include_filled=False) == [
        samples[0],
        samples[-1],
    ]
    assert trend.latest_measured("mold_temp") == samples[-1]
    assert trend.origin_segments("mold_temp", 0.0, 15.0) == [
        {
            "source": "measured",
            "filled": False,
            "usable_for_judgement": True,
            "start_ts": 0.0,
            "end_ts": 0.0,
            "count": 1,
        },
        {
            "source": "filled",
            "filled": True,
            "usable_for_judgement": False,
            "start_ts": 5.0,
            "end_ts": 10.0,
            "count": 2,
        },
        {
            "source": "measured",
            "filled": False,
            "usable_for_judgement": True,
            "start_ts": 15.0,
            "end_ts": 15.0,
            "count": 1,
        },
    ]


def test_trend_report_counts_sources_and_exposes_measured_latest(platform):
    now = platform.clock.now()
    platform.trend.append("mold_temp", now, 100.0)
    platform.trend.append("mold_temp", now + 10.0, 120.0)
    platform.clock.advance(10.0)

    report = ConsoleAPI(platform).trend(series="mold_temp", window_s=60.0)
    item = report["series"]["mold_temp"]

    assert item["filled_points"] == 1
    assert item["sample_counts"] == {"total": 3, "measured": 2, "filled": 1}
    assert item["segments"] == [
        {
            "source": "measured",
            "filled": False,
            "usable_for_judgement": True,
            "start_ts": round(now, 3),
            "end_ts": round(now, 3),
            "count": 1,
        },
        {
            "source": "filled",
            "filled": True,
            "usable_for_judgement": False,
            "start_ts": round(now + 5.0, 3),
            "end_ts": round(now + 5.0, 3),
            "count": 1,
        },
        {
            "source": "measured",
            "filled": False,
            "usable_for_judgement": True,
            "start_ts": round(now + 10.0, 3),
            "end_ts": round(now + 10.0, 3),
            "count": 1,
        },
    ]
    assert [sample["source"] for sample in item["samples"]] == [
        "measured",
        "filled",
        "measured",
    ]
    assert item["average"] == 110.0
    assert item["average_source"] == "measured"
    assert item["latest"]["source"] == "measured"
    assert item["latest_measured"]["source"] == "measured"
    assert "measured" in report["origins"]


def test_trend_window_without_measured_points_reports_null_judgement_value():
    trend = TrendBuffer(period_s=5.0)
    trend.append("steam_pressure", 0.0, 600.0)
    trend.fill_gaps("steam_pressure")

    assert trend.average("steam_pressure", 5.0, 20.0) is None
    assert trend.latest_measured("steam_pressure").timestamp == 0.0
