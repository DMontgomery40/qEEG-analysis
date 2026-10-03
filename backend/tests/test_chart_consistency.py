"""A chart whose contents disagree is named the same day, never weeks later.

Each case is a wrong-chart filing the clinic actually had: a woman's report
(printing "Female, 9/5/1954") on a man born 1982; a test upload filed under an
invented DM_03-14-1981 whose printed ages (42, 43) fit DM_09-23-1982; the same
bytes on two charts; one visit split across two charts.
"""

from __future__ import annotations

import hashlib
import importlib

import pytest

from backend import chart_consistency as cc

SUMMARY = (
    "Physical Reaction Time {rt} (+20) ms 280 (+18) ms 255-367 ms\n"
    "Trail Making Test A 41 sec 38 sec 38-64 sec\n"
    "Trail Making Test B 77 sec 70 sec 43-84 sec\n"
    "Audio P300 Delay 300 ms 290 ms 257-333 ms\n"
    "Audio P300 Voltage 9.1 uV 10.2 uV 6-14 uV\n"
    "CZ Eyes Closed Theta/Beta (Power) 1.8 2.1 0.9-1.9\n"
    "F3/F4 Eyes Closed Alpha (Power) 0.7 1.0 0.9-1.1\n"
    "Frontal Theta/Beta Ratio 2.0 2.2 1.2-2.4\n"
)


def wavi(header=None, sessions=(("3/4/2025", None), ("6/5/2025", None)), rt=283, extra="", summary=True):
    """A WAVi report's OCR text: header line, session table, page-1 summary."""
    lines = ["=== PAGE 1 / 2 ===", "WAVi Wellness Basic Report " + extra]
    if header:
        lines.append(f"fF — {header} — ID: N/A — Generated: 2/18/2026 10:28 AM")
    for n, (when, age) in enumerate(sessions, 1):
        lines.append(f"Session {n} ({when}) Followup N/A N/A N/A N/A" + (f" {age} yrs" if age else ""))
    text = "\n".join(lines) + "\n" + (SUMMARY.format(rt=rt) if summary else "")
    return text + "=== PAGE 2 / 2 ===\nappendix\n"


def chart(label):
    uuid = hashlib.md5(label.encode()).hexdigest()
    return uuid, {"uuid": uuid, "label": label, "birthdate": cc._chart_birthdate(label, None),
                  "clinic": cc._is_clinic_id(label)}


def report(report_id, uuid, text, filename=None):
    return {"id": report_id, "patient_uuid": uuid, "filename": filename or f"{report_id}.pdf",
            "sha256": hashlib.sha256(text.encode()).hexdigest(), "facts": cc.parse_report_text(text)}


def violations(*rows):
    charts = dict(chart(label) for label in {label for label, *_ in rows})
    reports = [report(rid, chart(label)[0], text) for label, rid, text in rows]
    return cc.find_violations(charts, reports)


def kinds(found):
    return sorted(v["kind"] for v in found)


def test_consistent_charts_have_no_violation():
    assert violations(
        ("DM_09-23-1982", "r1", wavi("Male, 9/23/1982", (("8/11/2025", 42), ("11/11/2025", 43)))),
        ("DM_09-23-1982", "r2", wavi(None, (("8/11/2025", 42), ("11/11/2025", 43)), extra="v2")),
        ("MF_09-05-1954", "r3", wavi("Female, 9/5/1954", (("12/15/2025", 71),), rt=301)),
    ) == []


def test_a_report_printing_another_birthday_is_named():
    found = violations(("DM_09-23-1982", "d16db596", wavi("Female, 9/5/1954", (("12/15/2025", 71),))))
    assert kinds(found) == ["birthday"], "its ages follow from the birthday; one line says it"
    assert found[0]["patients"] == ["DM_09-23-1982"]
    assert found[0]["reports"][0]["id"] == "d16db596"
    assert "9/5/1954" in found[0]["evidence"] and "09-23-1982" in found[0]["evidence"]


def test_one_misread_ocr_stream_is_not_a_wrong_birthday():
    text = wavi("Male, 9/23/1982") + "fF — Male, 9/28/1982 — ID: N/A\n"
    assert violations(("DM_09-23-1982", "r1", text)) == []


def test_printed_ages_that_do_not_fit_the_birthday_are_named():
    # Tonight's test upload: no printed birthday, ages 42 and 43.
    text = wavi(None, (("8/11/2025", 42), ("11/11/2025", 43)))
    found = violations(("DM_03-14-1981", "06d2c4c0", text))
    assert kinds(found) == ["age"]
    assert "42" in found[0]["evidence"] and "44" in found[0]["evidence"]
    assert violations(("DM_09-23-1982", "06d2c4c0", text)) == []


@pytest.mark.parametrize("label,when,age", [
    ("LM_12-02-1985", "11/27/2024", 39),  # live false positive: five days early
    ("AN_04-08-1986", "4/3/2026", 40),
])
def test_wavi_365_day_age_turnover_is_not_a_misfiling(label, when, age):
    assert violations((label, "r1", wavi(None, ((when, age),)))) == []
    assert kinds(violations((label, "r1", wavi(None, ((when, age + 1),))))) == ["age"]


def test_one_chart_printing_two_sexes_is_named_once_and_keeps_its_key():
    found = violations(
        ("AB_02-02-1990", "r1", wavi("Female, 2/2/1990")),
        ("AB_02-02-1990", "r2", wavi("Male, 2/2/1990", rt=290)),
    )
    assert kinds(found) == ["sex"]
    again = violations(
        ("AB_02-02-1990", "r1", wavi("Female, 2/2/1990")),
        ("AB_02-02-1990", "r2", wavi("Male, 2/2/1990", rt=290)),
        ("AB_02-02-1990", "r3", wavi("Female, 2/2/1990", rt=310)),
    )
    assert [v["key"] for v in again] == [found[0]["key"]], "a clean report adds no new alarm"


def test_the_same_bytes_on_two_charts_are_named_but_twice_on_one_chart_are_not():
    text = wavi(None, (("8/11/2025", 42), ("11/11/2025", 43)))
    found = violations(("DM_09-23-1982", "a", text), ("DM_09-23-1982", "b", text), ("DM_03-14-1981", "c", text))
    assert kinds(found) == ["age", "duplicate"]
    dup = next(v for v in found if v["kind"] == "duplicate")
    assert dup["patients"] == ["DM_03-14-1981", "DM_09-23-1982"]
    assert {r["id"] for r in dup["reports"]} == {"a", "b", "c"}
    assert violations(("DM_09-23-1982", "a", text), ("DM_09-23-1982", "b", text)) == []


def test_one_visit_on_two_charts_is_named_by_its_measured_values():
    found = violations(
        ("XX_03-05-2010", "a", wavi(None, extra="first export")),
        ("CV_02-12-2012", "b", wavi(None, extra="second export")),
    )
    assert kinds(found) == ["same_visit"]
    assert found[0]["patients"] == ["CV_02-12-2012", "XX_03-05-2010"]
    assert "3/4/2025" in found[0]["evidence"] and "6/5/2025" in found[0]["evidence"]
    # One measured value differs on 3/4/2025: that is a different visit, and
    # only the identical 6/5/2025 one is named.
    partly = violations(
        ("XX_03-05-2010", "a", wavi(None, extra="first export")),
        ("CV_02-12-2012", "b", wavi(None, rt=284, extra="second export")),
    )
    assert len(partly) == 1
    assert "6/5/2025" in partly[0]["evidence"] and "3/4/2025" not in partly[0]["evidence"]
    assert partly[0]["key"] != found[0]["key"], "a different set of shared visits is a new line"


def test_too_few_measured_values_never_make_a_same_visit():
    short = "\n".join(SUMMARY.splitlines()[:7]) + "\n"
    cut = [wavi(None, extra=e, summary=False).replace("=== PAGE 2", short.format(rt=283) + "=== PAGE 2")
           for e in ("one", "two")]
    assert violations(("XX_03-05-2010", "a", cut[0]), ("CV_02-12-2012", "b", cut[1])) == []


def test_legacy_and_test_charts_are_not_checked():
    text = wavi("Male, 10/7/1963")
    assert violations(("4-8-1997", "a", text), ("BB_10-07-1963", "b", text)) == []


def test_the_cache_is_keyed_by_the_text_not_the_pdf(tmp_path):
    original = tmp_path / "original.pdf"
    original.write_bytes(b"%PDF same bytes")
    (tmp_path / "extracted_enhanced.txt").write_text(wavi("Male, 9/23/1982"), encoding="utf-8")
    row = {"id": "r", "original": original, "text": tmp_path / "extracted_enhanced.txt", "sha256": None}
    cache = tmp_path / "cache.json"
    assert cc.read_reports([row], cache_path=cache) == []
    assert row["facts"]["birthdates"] == {"1982-09-23": 1}
    assert row["sha256"] == hashlib.sha256(b"%PDF same bytes").hexdigest()
    # The OCR is redone under the same PDF: the new text is read, not the cached one.
    (tmp_path / "extracted_enhanced.txt").write_text(wavi("Female, 9/5/1954"), encoding="utf-8")
    again = dict(row, sha256=None)
    cc.read_reports([again], cache_path=cache)
    assert again["facts"]["birthdates"] == {"1954-09-05": 1}


# --- at filing time --------------------------------------------------------


class _Events(list):
    def warning(self, event, **fields):
        self.append({"event": event, **fields})


def _file(key, first, last, birthdate, text, name):
    intake = importlib.import_module("backend.clinic_intake")
    return intake.submit_upload(
        key=key,
        identity={"firstName": first, "lastName": last, "birthdate": birthdate},
        files=[(name, text.encode(), "text/plain")],
        file_meta=[{"documentKind": "report"}],
        actor="Staff",
    )["upload"]


def test_filing_names_a_new_violation_once_and_a_clean_filing_says_nothing(temp_data_dir, monkeypatch):
    events = _Events()
    monkeypatch.setattr(cc, "LOGGER", events)
    clean = _file("f1", "Dan", "Moore", "09-23-1982",
                  wavi("Male, 9/23/1982", (("8/11/2025", 42),)), "dm.txt")
    assert clean["items"][0]["status"] == "registered"
    assert events == [], "a clean filing writes no note"

    misfiled = _file("f2", "Dan", "Moore", "09-23-1982",
                     wavi("Female, 9/5/1954", (("12/15/2025", 71),), rt=301), "MF_MCI_20 tx_final.txt")
    assert misfiled["items"][0]["status"] == "registered", "the filing is never blocked"
    named = [e for e in events if e["event"] == "chart_consistency_violation"]
    assert len(named) == 2, events  # her birthday, and a second sex on his chart
    assert {e["kind"] for e in named} == {"birthday", "sex"}
    assert all(e["patient_id"] == "DM_09-23-1982" for e in named)
    assert all(e["report_id"] == misfiled["items"][0]["sourceId"] for e in named)
    assert "9/5/1954" in next(e for e in named if e["kind"] == "birthday")["evidence"]

    events.clear()
    _file("f3", "Dan", "Moore", "09-23-1982",
          wavi("Male, 9/23/1982", (("1/5/2026", 43),), rt=299), "dm-later.txt")
    assert events == [], "a clean report on an already-flagged chart adds no note"


def test_a_detector_failure_never_fails_the_filing(temp_data_dir, monkeypatch):
    events = _Events()
    monkeypatch.setattr(cc, "LOGGER", events)

    def broken(*args, **kwargs):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(cc, "find_violations", broken)
    filed = _file("g1", "Ada", "Baker", "02-02-1900", wavi("Female, 9/5/1954"), "a.txt")
    assert filed["items"][0]["status"] == "registered"
    assert [e["event"] for e in events] == ["chart_consistency_check_failed"]
