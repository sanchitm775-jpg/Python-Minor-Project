"""
Tests for the LIVE-data code paths of the mini-projects.
Author : Sanchit Kumar Mishra

Why this file exists
--------------------
Every project tries a live source first (seaborn download, World Bank API, Open Library API,
Open-Meteo API, UCI download, Wikipedia) and falls back to bundled data if that fails.
The fallback path is easy to test (no network needed).  The live path is not - unless we
FAKE the network.  Here `requests.get` and `seaborn.load_dataset` are replaced with fakes that
return data in the same format the real services use, so the parsing code is really executed.

Run from the repository root:
    python -m unittest tests.test_live_paths_with_mocks -v
    (or)  python tests/test_live_paths_with_mocks.py

NOTE: these tests prove that the parsers handle the documented formats. They do not prove
that a real server is reachable or has not changed its format - run each project once with
internet to confirm that.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]


def load_project(folder: str):
    """Import a project's main.py under a unique module name (the folder names contain hyphens)."""
    path = ROOT / folder / "main.py"
    spec = importlib.util.spec_from_file_location(f"mod_{folder.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quiet(function, *args, **kwargs):
    """Call a function while hiding its printed output (keeps test logs readable)."""
    with contextlib.redirect_stdout(io.StringIO()) as buffer:
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, json_data=None, text="", content=b"", status=200):
        self._json, self.text, self.content, self.status_code = json_data, text, content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json


# --------------------------------------------------------------------------- #
class Project01Seaborn(unittest.TestCase):
    def test_live_path_with_real_schema(self):
        module = load_project("project-01-mpg-regression")
        rng = np.random.default_rng(0)
        n = 120
        weight = rng.uniform(1600, 5000, n)
        frame = pd.DataFrame({
            "mpg": 55 - weight * 0.008 + rng.normal(0, 2, n),
            "cylinders": rng.choice([4, 6, 8], n), "displacement": weight * 0.07 + rng.normal(0, 10, n),
            "horsepower": weight * 0.04 + rng.normal(0, 8, n), "weight": weight,
            "acceleration": rng.uniform(8, 25, n), "model_year": rng.integers(70, 83, n),
            "origin": rng.choice(["usa", "japan", "europe"], n), "name": ["car"] * n,
        })
        frame.loc[[3, 9], "horsepower"] = np.nan  # the real dataset has 6 missing horsepower values
        with mock.patch("seaborn.load_dataset", return_value=frame):
            (df, source), _ = quiet(module.load_data, False)
            self.assertIn("seaborn", source)
            self.assertEqual(len(df), n - 2)  # NaN rows dropped
            result, _ = quiet(module.run, False)
        self.assertIn("seaborn", result["source"])


class Project03WorldBank(unittest.TestCase):
    @staticmethod
    def fake_world_bank(url, params=None, **kwargs):
        meta = {"page": 1, "pages": 1}
        if url.endswith("/country"):
            rows = [{"id": "WLD", "name": "World", "region": {"id": "NA", "value": "Aggregates"},
                     "incomeLevel": {"id": "NA", "value": "Aggregates"}}]
            for i in range(40):
                rows.append({"id": f"C{i:02d}", "name": f"Country {i}",
                             "region": {"id": "XX", "value": "Some Region "},
                             "incomeLevel": {"id": "H" if i >= 25 else "L", "value": "High income" if i >= 25 else "Low income"}})
            return FakeResponse(json_data=[meta, rows])
        code = url.split("/indicator/")[1]
        rows = [{"countryiso3code": "WLD", "date": "2021", "value": 1.0}]  # aggregate row must be ignored later
        for i in range(40):
            iso = f"C{i:02d}"
            rich = i >= 25
            base = {"NY.GDP.PCAP.CD": 30000 + 800 * i if rich else 400 + 60 * i,
                    "SP.DYN.LE00.IN": 80 - (40 - i) * 0.05 if rich else 55 + i * 0.3,
                    "SE.ADT.LITR.ZS": None if rich else 40 + i * 2,        # rich countries: literacy missing
                    "SP.DYN.IMRT.IN": 4 if rich else 60 - i}[code]
            if base is None:
                continue
            rows.append({"countryiso3code": iso, "date": "2021", "value": float(base)})
            rows.append({"countryiso3code": iso, "date": "2022", "value": None})   # latest year empty -> must use 2021
        return FakeResponse(json_data=[meta, rows])

    def test_parsers_and_full_run(self):
        module = load_project("project-03-country-clustering")
        with mock.patch("requests.get", side_effect=self.fake_world_bank):
            raw = module.fetch_world_bank()
            self.assertNotIn("World", raw["country"].tolist())          # aggregates removed
            self.assertEqual(len(raw), 40)
            self.assertEqual(raw.loc[raw["country"] == "Country 30", "adult_literacy"].iloc[0], 99.0)  # rich-country imputation
            self.assertAlmostEqual(raw.loc[raw["country"] == "Country 0", "gdp_per_capita"].iloc[0], 400.0)  # 2021, not the null 2022
            result, _ = quiet(module.run, False)
        self.assertIn("live", result["source"])


class Project04Spam(unittest.TestCase):
    def test_zip_and_mirror_paths(self):
        module = load_project("project-04-spam-classifier")
        sample = pd.read_csv(ROOT / "project-04-spam-classifier" / "data" / "sms_sample.csv")
        tsv = "\n".join(f"{r.label}\t{r.text}" for r in sample.itertuples())
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("SMSSpamCollection", tsv)

        # (a) UCI zip works
        with mock.patch("requests.get", return_value=FakeResponse(content=buffer.getvalue())):
            df = module.download_real_data()
        self.assertEqual(len(df), len(sample.drop_duplicates()))
        self.assertEqual(set(df["label"]), {"ham", "spam"})

        # (b) zip fails -> GitHub mirror is used
        def zip_down_mirror_up(url, **kwargs):
            if "archive.ics.uci.edu" in url:
                raise requests.ConnectionError("down")
            return FakeResponse(text=tsv)
        with mock.patch("requests.get", side_effect=zip_down_mirror_up):
            df2 = module.download_real_data()
        self.assertEqual(len(df2), len(df))

        # (c) full run on the "downloaded" data
        with mock.patch("requests.get", return_value=FakeResponse(content=buffer.getvalue())):
            result, _ = quiet(module.run, False)
        self.assertIn("UCI", result["source"])


class Project05Diamonds(unittest.TestCase):
    def test_live_path_with_zero_size_rows(self):
        module = load_project("project-05-diamonds-dashboard")
        frame = module.simulate_diamonds(1500)
        frame.loc[[0, 1], ["x", "y", "z"]] = 0  # the real dataset contains a few zero-size rows
        with mock.patch("seaborn.load_dataset", return_value=frame):
            result, _ = quiet(module.run, False)
        self.assertFalse(result["simulated"])
        self.assertIn("seaborn", result["source"])


class Project07OpenLibrary(unittest.TestCase):
    def test_parser_and_full_run(self):
        module = load_project("project-07-book-recommender")
        sample = pd.read_csv(ROOT / "project-07-book-recommender" / "data" / "books_sample.csv")

        def fake_open_library(url, params=None, **kwargs):
            genre = params["q"].split('"')[1]
            docs = []
            for i, r in enumerate(sample[sample["genre"] == genre].itertuples()):
                docs.append({
                    "title": r.title, "author_name": [r.author], "first_publish_year": int(r.year),
                    "edition_count": 100 - i,
                    "subject": [genre.title(), "Accessible book", "Protected DAISY", "Fiction, general"] + [t.title() for t in r.tags.split("; ")],
                })
            docs.append({"title": "Too few subjects", "subject": ["Fiction"], "edition_count": 1})  # must be skipped
            return FakeResponse(json_data={"numFound": len(docs), "docs": docs})

        with mock.patch("requests.get", side_effect=fake_open_library), mock.patch("time.sleep"):
            df = module.fetch_open_library()
            self.assertEqual(len(df), len(sample))
            self.assertNotIn("Too few subjects", df["title"].tolist())
            all_tags = " ; ".join(df["tags"])
            self.assertNotIn("accessible", all_tags)       # catalogue noise removed
            self.assertNotIn("daisy", all_tags)
            result, _ = quiet(module.run, False)
        self.assertIn("Open Library API", result["source"])

    def test_clean_subjects(self):
        module = load_project("project-07-book-recommender")
        tags = module.clean_subjects(["Fantasy", "Dragons", "Large type books", "A very long subject heading with many words", "dragons", "Magic"], "fantasy")
        self.assertEqual(tags, ["dragons", "magic"])


class Project08Wikipedia(unittest.TestCase):
    ALTERNATIVE_LAYOUT = """
    <html><body>
    <table class="wikitable sortable"><tr><th>Rank</th><th>State/UT</th><th>Population (2011)</th>
    <th>Decadal growth</th><th>Area (km2)</th><th>Density (/km2)</th></tr>
    {rows}
    </table></body></html>"""

    def test_alternative_table_layout_and_noise(self):
        module = load_project("project-08-wikipedia-table-scraper")
        rows = "".join(
            f'<tr><td>{i}</td><td><a>State {i}</a><sup>[3]</sup></td><td>{1_000_000 * (40 - i):,}</td>'
            f'<td>{10 + i}.5%</td><td>{50_000 + i * 100:,}</td><td>{300 + i}</td></tr>' for i in range(1, 31))
        rows += "<tr><td></td><td>Total</td><td>999</td><td>-</td><td>1</td><td>1</td></tr>"
        df = module.html_to_dataframe(self.ALTERNATIVE_LAYOUT.format(rows=rows))
        self.assertEqual(len(df), 30)                         # total row dropped
        self.assertEqual(df.loc[0, "state"], "State 1")       # citation mark [3] removed
        self.assertEqual(df.loc[0, "population"], 39_000_000)  # commas parsed; 'decadal growth' NOT mistaken for population
        self.assertEqual(df.loc[0, "area_km2"], 50_100)
        self.assertTrue(df["sex_ratio"].isna().all())         # absent column -> NaN, no crash

    def test_live_path_and_missing_table_fallback(self):
        module = load_project("project-08-wikipedia-table-scraper")
        html = (ROOT / "project-08-wikipedia-table-scraper" / "data" / "sample_page.html").read_text(encoding="utf-8")
        with mock.patch("requests.get", return_value=FakeResponse(text=html)):
            df, source = module.load_table(offline=False)
        self.assertEqual(source, "live Wikipedia page")
        self.assertEqual(len(df), 36)
        with mock.patch("requests.get", return_value=FakeResponse(text="<html><body><p>no tables</p></body></html>")):
            _, source = quiet(module.load_table, False)[0]
        self.assertIn("bundled", source)                       # page changed -> graceful fallback


class Project10OpenMeteo(unittest.TestCase):
    def test_parser_keeps_nulls_and_full_run(self):
        module = load_project("project-10-weather-eda")
        sim = module.simulate_weather()
        daily = {"time": sim["date"].dt.strftime("%Y-%m-%d").tolist()}
        for api_name, column in module.API_VARIABLES.items():
            values = sim[column].astype(object).tolist()
            values[-3:] = [None, None, None]  # the newest days are often null in the real archive
            daily[api_name] = values
        payload = {"latitude": 21.25, "longitude": 81.63, "daily": daily}
        with mock.patch("requests.get", return_value=FakeResponse(json_data=payload)):
            df = module.fetch_open_meteo()
            self.assertEqual(int(df["rain_mm"].isna().sum()), 3)  # nulls preserved as NaN, not hidden
            result, output = quiet(module.run, False)
        self.assertFalse(result["simulated"])
        self.assertIn("21 missing cells", output)               # 3 null days x 7 weather columns


class Project09Organizer(unittest.TestCase):
    def setUp(self):
        self.module = load_project("project-09-file-organizer")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicates_collisions_and_undo(self):
        (self.root / "a.txt").write_text("same")
        (self.root / "b.txt").write_text("same")                 # duplicate of a.txt
        (self.root / "c.txt").write_text("diff")                 # same size, different content
        (self.root / "pic.png").write_bytes(b"123456789")
        (self.root / ".hidden").write_text("skip me")
        files = self.module.list_files(self.root)
        self.assertNotIn(".hidden", [f.name for f in files])
        groups, stats = self.module.find_duplicates(files)
        self.assertEqual(stats["duplicate_files"], 1)
        self.assertEqual(stats["same_size_but_different_content"], 0)  # a,b,c all 4 bytes: 1 group of 3 -> hashed -> 1 dup group
        moves = self.module.plan_moves(self.root, files, groups, by_date=False)
        self.assertEqual(sum(m["reason"] == "duplicate" for m in moves), 1)
        before = {p.name: p.read_bytes() for p in files}
        self.module.execute_moves(self.root, moves)
        self.assertFalse((self.root / "b.txt").exists())
        restored = self.module.undo(self.root)
        self.assertEqual(restored, 4)
        after = {p.name: p.read_bytes() for p in self.module.list_files(self.root)}
        self.assertEqual(before, after)

    def test_dry_run_changes_nothing(self):
        (self.root / "x.jpg").write_bytes(b"1")
        with contextlib.redirect_stdout(io.StringIO()):
            code = self.module.main([str(self.root)])
        self.assertEqual(code, 0)
        self.assertTrue((self.root / "x.jpg").exists())


if __name__ == "__main__":
    os.chdir(ROOT)
    unittest.main(verbosity=2)
