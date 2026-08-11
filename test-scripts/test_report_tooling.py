#!/usr/bin/env python3
"""
Tests for the report/paper tooling: shared_lib.paper_lint, typst_report
scaffolding and detection, publish CLI derivation, and the R2 key plumbing.

Stdlib unittest, no pytest — this has to run on a machine with no venv. Nothing
here touches the network, R2 credentials, or a GPU. The few checks that need
the Typst compiler skip themselves when it is absent, so the suite is green on
the Mac and covers more on the GPU box.

    python test-scripts/test_report_tooling.py
    python test-scripts/test_report_tooling.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_lib import media, paper_lint, publish, typst_report
from shared_lib.results import latest, load_results, run_order
from shared_lib.paper_lint import (
    _is_uninteresting, _iter_scalars, _matches_any, _precision, _prose,
    check_paper, load_allow, load_results_values,
)
from shared_lib.typst_report import (
    _cross_section_stubs, _fill, _typ_str, content_hashes, scaffold_paper,
    show_rule_fn,
)

try:
    import typst  # noqa: F401
    HAS_TYPST = True
except ImportError:
    HAS_TYPST = False


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TreeCase(unittest.TestCase):
    """Base for tests that need a throwaway report tree on disk."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.project = Path(self._tmp.name)
        self.root = self.project / "paper"
        self.root.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def make_tree(self, sections: dict[str, str], *, entry: str | None = None,
                  results: list[dict] | None = None, bib: str | None = None) -> Path:
        for stem, body in sections.items():
            write(self.root / "sections" / f"{stem}.typ", body)
        includes = "\n".join(f'#include "/sections/{s}.typ"' for s in sections)
        write(self.root / "main.typ",
              entry if entry is not None else f'#show: paper.with(title: "T")\n{includes}\n')
        if results is not None:
            write(self.project / "results.jsonl",
                  "".join(json.dumps(r) + "\n" for r in results))
        if bib is not None:
            write(self.root / "refs.bib", bib)
        return self.root


# ───────────────────────────── numeric matching ─────────────────────────────


class TestPrecision(unittest.TestCase):
    def test_decimal_places(self):
        self.assertEqual(_precision("0.228"), 3)
        self.assertEqual(_precision("1.5"), 1)
        self.assertEqual(_precision("128"), 0)

    def test_scientific_notation(self):
        """The bug this caught: float() round-trips lose the written precision,
        so 3e-4 was being compared at 0 decimal places and never matched."""
        self.assertEqual(_precision("3e-4"), 4)
        self.assertEqual(_precision("1E-6"), 6)
        self.assertEqual(_precision("1e3"), -3)

    def test_garbage_does_not_raise(self):
        self.assertEqual(_precision("not-a-number"), 0)


class TestMatching(unittest.TestCase):
    def test_matches_at_written_precision(self):
        self.assertTrue(_matches_any("0.228", {0.22814}))
        self.assertTrue(_matches_any("0.2281", {0.22814}))

    def test_rejects_beyond_written_precision(self):
        self.assertFalse(_matches_any("0.229", {0.22814}))
        self.assertFalse(_matches_any("0.238", {0.22814}))

    def test_percentages(self):
        self.assertTrue(_matches_any("22.8", {0.22814}))

    def test_scientific_literal_matches_stored_float(self):
        self.assertTrue(_matches_any("3e-4", {0.0003}))

    def test_unit_scales_are_opt_in(self):
        self.assertFalse(_matches_any("3.38", {3378186.0}))
        self.assertTrue(_matches_any("3.38", {3378186.0}, unit_scales=True))

    def test_empty_pool(self):
        self.assertFalse(_matches_any("0.228", set()))


class TestUninteresting(unittest.TestCase):
    def test_small_integers_and_years_skipped(self):
        for lit in ("0", "7", "20", "2026", "1999"):
            self.assertTrue(_is_uninteresting(lit), lit)

    def test_real_measurements_checked(self):
        for lit in ("0.228", "21", "128", "0.5"):
            self.assertFalse(_is_uninteresting(lit), lit)


class TestResultsPool(unittest.TestCase):
    ROW = {"acc": 0.228, "n": 5, "history": {"acc": [0.1 * i for i in range(40)]},
           "seeds": [1, 2, 3]}

    def test_curves_excluded_by_default(self):
        vals = set(_iter_scalars(self.ROW))
        self.assertIn(0.228, vals)
        self.assertIn(3.0, vals, "short lists are summary data and stay in")
        self.assertNotIn(1.5, vals, "a 40-long curve is not part of the pool")

    def test_curves_included_when_asked(self):
        vals = set(_iter_scalars(self.ROW, max_list=None))
        self.assertIn(1.5, vals)

    def test_booleans_are_not_numbers(self):
        self.assertEqual(set(_iter_scalars({"ok": True, "bad": False})), set())

    def test_non_finite_dropped(self):
        self.assertEqual(set(_iter_scalars({"x": float("nan"), "y": float("inf")})), set())

    def test_load_from_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = write(Path(td) / "results.jsonl",
                      '{"acc": 0.5}\n\n{"acc": 0.75}\nnot json\n')
            self.assertEqual(load_results_values(p), {0.5, 0.75})

    def test_missing_file(self):
        self.assertEqual(load_results_values("/nonexistent/results.jsonl"), set())


# ──────────────────────────────── prose ─────────────────────────────────────


class TestResultsLoading(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "results.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def test_keyed_by_experiment(self):
        write(self.path, '{"experiment": "exp1", "acc": 0.5}\n'
                         '{"experiment": "exp2", "acc": 0.7}\n')
        rows = load_results(self.path)
        self.assertEqual(set(rows), {"exp1", "exp2"})
        self.assertEqual(rows["exp2"]["acc"], 0.7)

    def test_first_duplicate_wins(self):
        """Two runners that both read the skip-if-done set before either wrote
        each append the same experiment; the first finished cleanly."""
        write(self.path, '{"experiment": "exp1", "acc": 0.5}\n'
                         '{"experiment": "exp1", "acc": 0.9}\n')
        self.assertEqual(load_results(self.path)["exp1"]["acc"], 0.5)

    def test_malformed_lines_skipped(self):
        """A file truncated mid-write by a killed job must still yield the
        runs that completed."""
        write(self.path, '{"experiment": "exp1", "acc": 0.5}\n'
                         'not json at all\n'
                         '{"experiment": "exp2", "acc":\n')
        self.assertEqual(set(load_results(self.path)), {"exp1"})

    def test_blank_lines_and_missing_file(self):
        write(self.path, '\n\n{"experiment": "exp1"}\n\n')
        self.assertEqual(set(load_results(self.path)), {"exp1"})
        self.assertEqual(load_results(self.path.parent / "nope.jsonl"), {})

    def test_non_dict_rows_ignored(self):
        write(self.path, '[1, 2, 3]\n"a string"\n{"experiment": "exp1"}\n')
        self.assertEqual(set(load_results(self.path)), {"exp1"})

    def test_run_order_is_numeric(self):
        """Plain sorted() puts exp10 between exp1 and exp2, silently
        scrambling every axis built from it."""
        self.assertEqual(
            run_order(["exp10", "exp2", "exp1", "exp13"]),
            ["exp1", "exp2", "exp10", "exp13"],
        )

    def test_run_order_without_digits(self):
        self.assertEqual(run_order(["baseline", "exp2"]), ["baseline", "exp2"])

    def test_latest_picks_highest_numbered(self):
        rows = {"exp1": {"chance": 0.5}, "exp10": {"chance": 0.2}}
        self.assertEqual(latest(rows, "chance"), 0.2)

    def test_latest_falls_back_past_rows_missing_the_field(self):
        rows = {"exp1": {"chance": 0.5}, "exp10": {"other": 1}}
        self.assertEqual(latest(rows, "chance"), 0.5)
        self.assertIsNone(latest(rows, "absent"))


class TestProse(unittest.TestCase):
    def test_line_comments_stripped(self):
        """Section stubs carry their obligations in comments, and those contain
        numbers. Checking them would make every fresh scaffold noisy."""
        self.assertNotIn("0.999", _prose("// see 0.999 in the notes\ntext 0.5"))
        self.assertIn("0.5", _prose("// see 0.999 in the notes\ntext 0.5"))

    def test_raw_blocks_stripped(self):
        self.assertNotIn("3.11", _prose("prose ```python\nversion 3.11\n``` more"))
        self.assertNotIn("42", _prose("inline `x = 42` here"))

    def test_block_comments_stripped(self):
        self.assertNotIn("7.7", _prose("/* 7.7 */ kept 1.1"))


# ─────────────────────────────── structure ──────────────────────────────────


class TestStructure(TreeCase):
    def test_clean_tree(self):
        self.make_tree({"01-a": "= A\n"}, results=[{"acc": 0.5}])
        report = check_paper(self.root)
        self.assertEqual(report.errors, [])

    def test_orphan_section(self):
        """The live case: sparse-attn-emergence renamed 06-reading to
        07-reading and left the old file tracked in git."""
        self.make_tree({"01-a": "= A\n"}, results=[{"acc": 0.5}])
        write(self.root / "sections" / "99-orphan.typ", "= Orphan\n")
        kinds = [f.kind for f in check_paper(self.root).findings]
        self.assertIn("orphan-section", kinds)

    def test_missing_include_is_an_error(self):
        write(self.root / "main.typ", '#include "/sections/nope.typ"\n')
        report = check_paper(self.root)
        self.assertTrue(any(f.kind == "missing-include" for f in report.errors))

    def test_missing_entry(self):
        report = check_paper(self.root)
        self.assertTrue(any(f.kind == "missing-include" for f in report.errors))

    def test_missing_figure_is_an_error(self):
        self.make_tree({"01-a": '#fig(include "/figures/gone.typ")\n'}, results=[{"a": 1.0}])
        report = check_paper(self.root)
        self.assertTrue(any(f.kind == "missing-figure" for f in report.errors))

    def test_missing_asset_is_an_error(self):
        self.make_tree({"01-a": '#fig(include "/figures/f.typ")\n'}, results=[{"a": 1.0}])
        write(self.root / "figures" / "f.typ", '#plot(data: json("/assets/f.json"))\n')
        report = check_paper(self.root)
        self.assertTrue(any(f.kind == "missing-asset" for f in report.errors))

    def test_figure_with_asset_is_clean(self):
        self.make_tree({"01-a": '#fig(include "/figures/f.typ")\n'}, results=[{"a": 1.0}])
        write(self.root / "figures" / "f.typ", '#plot(data: json("/assets/f.json"))\n')
        write(self.root / "assets" / "f.json", "{}")
        self.assertEqual(check_paper(self.root).errors, [])

    def test_todo_reported_as_warning(self):
        self.make_tree({"01-a": "#todo[write this]\n"}, results=[{"a": 1.0}])
        report = check_paper(self.root)
        self.assertTrue(any(f.kind == "unwritten" for f in report.findings))
        self.assertEqual(report.errors, [], "an unfinished draft is not a broken one")


class TestCitations(TreeCase):
    BIB = '@article{smith2020,\n  title = {T},\n  author = {S},\n  year = {2020},\n}\n'

    def test_known_citation_is_clean(self):
        self.make_tree({"01-a": "as shown @smith2020\n"}, results=[{"a": 1.0}], bib=self.BIB)
        kinds = [f.kind for f in check_paper(self.root).findings]
        self.assertNotIn("unknown-citation", kinds)
        self.assertNotIn("uncited-reference", kinds)

    def test_trailing_period_not_part_of_key(self):
        """`@smith2020.` at the end of a sentence was being read as the key
        `smith2020.`, reporting both an unknown citation and an uncited entry."""
        self.make_tree({"01-a": "as shown @smith2020.\n"}, results=[{"a": 1.0}], bib=self.BIB)
        kinds = [f.kind for f in check_paper(self.root).findings]
        self.assertNotIn("unknown-citation", kinds)
        self.assertNotIn("uncited-reference", kinds)

    def test_label_reference_is_not_a_citation(self):
        """Typst spells cross-references and citations identically."""
        self.make_tree(
            {"01-a": "#table()<tab:runs>\nsee @tab:runs\n"},
            results=[{"a": 1.0}], bib=self.BIB,
        )
        kinds = [f.kind for f in check_paper(self.root).findings]
        self.assertNotIn("unknown-citation", kinds)

    def test_unknown_citation_reported(self):
        self.make_tree({"01-a": "see @ghost2019\n"}, results=[{"a": 1.0}], bib=self.BIB)
        self.assertTrue(any(f.kind == "unknown-citation" for f in check_paper(self.root).findings))

    def test_uncited_entry_reported(self):
        self.make_tree({"01-a": "no citations here\n"}, results=[{"a": 1.0}], bib=self.BIB)
        self.assertTrue(any(f.kind == "uncited-reference" for f in check_paper(self.root).findings))

    def test_no_bib_means_no_citation_checks(self):
        self.make_tree({"01-a": "see @anything\n"}, results=[{"a": 1.0}])
        kinds = [f.kind for f in check_paper(self.root).findings]
        self.assertNotIn("unknown-citation", kinds)


# ────────────────────────── numbers, end to end ─────────────────────────────


class TestNumberChecking(TreeCase):
    def test_matching_number_is_silent(self):
        self.make_tree({"01-a": "accuracy reached 0.228\n"}, results=[{"acc": 0.22814}])
        self.assertEqual(
            [f for f in check_paper(self.root).findings if f.kind == "unmatched-number"], [])

    def test_stale_number_is_caught(self):
        """The regression this whole check exists for: a rerun moved the value
        and the prose kept the old one."""
        self.make_tree({"01-a": "accuracy reached 0.228\n"}, results=[{"acc": 0.31}])
        self.assertTrue(
            any(f.kind == "unmatched-number" for f in check_paper(self.root).findings))

    def test_allow_list_silences(self):
        self.make_tree({"01-a": "trained on 1623 characters\n"}, results=[{"acc": 0.5}])
        write(self.root / ".lint-allow", "1623  # Omniglot character count, not a result\n")
        self.assertEqual(
            [f for f in check_paper(self.root).findings if f.kind == "unmatched-number"], [])

    def test_allow_list_parsing(self):
        write(self.root / ".lint-allow", "\n# a comment line\n1623 # why\n0.5\n\n")
        self.assertEqual(load_allow(self.root), {"1623", "0.5"})

    def test_missing_allow_file(self):
        self.assertEqual(load_allow(self.root), set())

    def test_repeated_literal_reported_once(self):
        self.make_tree(
            {"01-a": "value 0.777 here\n", "02-b": "value 0.777 again\n"},
            results=[{"acc": 0.5}],
        )
        hits = [f for f in check_paper(self.root).findings if f.kind == "unmatched-number"]
        self.assertEqual(len(hits), 1)
        self.assertIn("01-a.typ", hits[0].where)
        self.assertIn("02-b.typ", hits[0].where)

    def test_numbers_in_comments_ignored(self):
        self.make_tree({"01-a": "// obligation: report 0.999\ntext\n"}, results=[{"acc": 0.5}])
        self.assertEqual(
            [f for f in check_paper(self.root).findings if f.kind == "unmatched-number"], [])

    def test_empty_results_reports_rather_than_crashing(self):
        self.make_tree({"01-a": "value 0.777\n"})
        self.assertTrue(any(f.kind == "no-results" for f in check_paper(self.root).findings))

    def test_curve_point_needs_the_flag(self):
        rows = [{"acc": 0.5, "history": {"acc": [round(0.01 * i, 4) for i in range(40)]}}]
        self.make_tree({"01-a": "peaked at 0.37 mid-run\n"}, results=rows)
        self.assertTrue(
            any(f.kind == "unmatched-number" for f in check_paper(self.root).findings))
        self.assertFalse(
            any(f.kind == "unmatched-number"
                for f in check_paper(self.root, include_curves=True).findings))

    def test_check_numbers_can_be_disabled(self):
        self.make_tree({"01-a": "value 0.777\n"}, results=[{"acc": 0.5}])
        kinds = [f.kind for f in check_paper(self.root, check_numbers=False).findings]
        self.assertNotIn("unmatched-number", kinds)


# ──────────────────────────── typst_report bits ─────────────────────────────


class TestFill(unittest.TestCase):
    def test_substitutes_known_keys(self):
        self.assertEqual(_fill("title: {title}", title='"X"'), 'title: "X"')

    def test_leaves_typst_code_blocks_alone(self):
        """str.format would raise on the first `{` somebody adds to a scaffold
        file, which is a trap for whoever edits the template next."""
        src = "#let f(x) = { x + 1 }\ntitle: {title}\n#if a { b } else { c }\n"
        out = _fill(src, title='"X"')
        self.assertIn("{ x + 1 }", out)
        self.assertIn("{ b }", out)
        self.assertIn('title: "X"', out)

    def test_unknown_placeholder_untouched(self):
        self.assertEqual(_fill("{other}", title="X"), "{other}")


class TestTypStrings(unittest.TestCase):
    def test_non_ascii_stays_literal(self):
        """Typst escapes are \\u{2014}, not JSON's \\u2014 — ensure_ascii would
        emit an em dash as six visible characters."""
        self.assertEqual(_typ_str("a — b"), '"a — b"')

    def test_quotes_escaped(self):
        self.assertEqual(_typ_str('say "hi"'), '"say \\"hi\\""')


class TestShowRuleDetection(TreeCase):
    def test_detects_paper(self):
        write(self.root / "main.typ", '#import "/template.typ": *\n#show: paper.with(title: "T")\n')
        self.assertEqual(show_rule_fn(self.root), "paper")

    def test_detects_report(self):
        write(self.root / "main.typ", '#show: report.with(title: "T")\n')
        self.assertEqual(show_rule_fn(self.root), "report")

    def test_falls_back_when_absent(self):
        write(self.root / "main.typ", "no show rule here\n")
        self.assertEqual(show_rule_fn(self.root), "report")

    def test_falls_back_when_no_entry(self):
        self.assertEqual(show_rule_fn(self.root), "report")


class TestCrossSectionStubs(TreeCase):
    BIB = '@article{smith2020, title={T}, author={S}, year={2020},}\n'

    def test_no_references_no_stubs(self):
        self.assertEqual(_cross_section_stubs(self.root, "plain prose\n"), "")

    def test_sibling_label_is_stubbed(self):
        out = _cross_section_stubs(self.root, "as shown in @sec:analysis\n")
        self.assertIn("<sec:analysis>", out)
        self.assertIn("height: 0pt", out)

    def test_link_form_is_a_reference(self):
        """The form the real papers actually use — `@key` is the minority
        spelling, and only handling it left every section preview broken."""
        out = _cross_section_stubs(self.root, "see §#link(<sec:analysis>)[7.4] there\n")
        self.assertIn("<sec:analysis>", out)

    def test_link_form_definition_not_confused_with_reference(self):
        text = "= Analysis <sec:analysis>\n\nsee §#link(<sec:other>)[3]\n"
        defined, referenced = typst_report.label_uses(text)
        self.assertEqual(defined, {"sec:analysis"})
        self.assertEqual(referenced, {"sec:other"})

    def test_own_label_not_stubbed(self):
        """Stubbing a label the section defines would be a duplicate."""
        out = _cross_section_stubs(self.root, "= H <sec:here>\nsee @sec:here\n")
        self.assertEqual(out, "")

    def test_citation_not_stubbed(self):
        """A bibliography key resolves via refs.bib, not a label."""
        write(self.root / "refs.bib", self.BIB)
        self.assertEqual(_cross_section_stubs(self.root, "see @smith2020\n"), "")

    def test_citation_and_label_together(self):
        write(self.root / "refs.bib", self.BIB)
        out = _cross_section_stubs(self.root, "see @smith2020 and @tab:runs\n")
        self.assertIn("<tab:runs>", out)
        self.assertNotIn("smith2020", out)

    def test_stubs_carry_their_own_numbering(self):
        """Typst refuses to reference an unnumbered heading, and the short
        report template does not number headings at all."""
        out = _cross_section_stubs(self.root, "see @sec:x\n")
        self.assertIn('set heading(numbering: "1.1")', out)


class TestScaffoldPaper(TreeCase):
    def test_creates_the_tree(self):
        scaffold_paper(self.root, title="A Title", subtitle="sub", date="2026-08-12")
        for rel in ("main.typ", "template.typ", "refs.bib", ".gitignore",
                    "sections/01-introduction.typ", "sections/09-conclusion.typ"):
            self.assertTrue((self.root / rel).exists(), rel)
        self.assertTrue((self.root / "figures").is_dir())
        self.assertTrue((self.root / "assets").is_dir())

    def test_no_placeholders_survive(self):
        """Mirrors the grep check project_start.md asks for by hand."""
        scaffold_paper(self.root, title="A Title")
        text = (self.root / "main.typ").read_text()
        for token in ("{title}", "{subtitle}", "{date}", "{includes}", "{paper_dir}"):
            self.assertNotIn(token, text)
        self.assertIn('title: "A Title"', text)
        self.assertIn("subtitle: none", text)

    def test_scaffolded_tree_declares_paper_show_rule(self):
        scaffold_paper(self.root, title="T")
        self.assertEqual(show_rule_fn(self.root), "paper")

    def test_every_section_is_included(self):
        scaffold_paper(self.root, title="T")
        report = check_paper(self.root, check_numbers=False)
        self.assertEqual([f.kind for f in report.findings if f.kind == "orphan-section"], [])

    def test_scaffold_is_a_draft_with_todos(self):
        scaffold_paper(self.root, title="T")
        report = check_paper(self.root, check_numbers=False)
        self.assertTrue(any(f.kind == "unwritten" for f in report.findings))
        self.assertEqual(report.errors, [], "a fresh scaffold must not be an error")

    def test_does_not_overwrite_by_default(self):
        scaffold_paper(self.root, title="T")
        edited = self.root / "sections" / "01-introduction.typ"
        edited.write_text("= Mine\n", encoding="utf-8")
        scaffold_paper(self.root, title="T")
        self.assertEqual(edited.read_text(), "= Mine\n")

    def test_overwrite_restores(self):
        scaffold_paper(self.root, title="T")
        edited = self.root / "sections" / "01-introduction.typ"
        edited.write_text("= Mine\n", encoding="utf-8")
        scaffold_paper(self.root, title="T", overwrite=True)
        self.assertNotEqual(edited.read_text(), "= Mine\n")

    def test_custom_section_gets_a_stub(self):
        scaffold_paper(self.root, title="T", sections=("01-introduction", "50-invented"))
        body = (self.root / "sections" / "50-invented.typ").read_text()
        self.assertIn("= Invented", body)
        self.assertIn("#todo", body)


class TestContentHashes(TreeCase):
    def test_tracks_content_ignores_derived(self):
        write(self.root / "main.typ", "x\n")
        write(self.root / "assets" / "d.json", "{}")
        write(self.root / "out.pdf", "derived")
        write(self.root / ".build.json", "{}")
        write(self.root / ".gitignore", "out.*")
        keys = set(content_hashes(self.root))
        self.assertEqual(keys, {"main.typ", "assets/d.json"})

    def test_hash_changes_with_content(self):
        write(self.root / "main.typ", "a\n")
        before = content_hashes(self.root)
        write(self.root / "main.typ", "b\n")
        self.assertNotEqual(before, content_hashes(self.root))


# ─────────────────────────────── publish CLI ────────────────────────────────


class TestPublishDerivation(TreeCase):
    def test_name_from_path(self):
        self.assertEqual(publish.derive_name(self.root), f"{self.project.name}_paper")

    def test_title_from_entry(self):
        write(self.root / "main.typ", '#show: paper.with(\n  title: "What is hard to learn?",\n)\n')
        self.assertEqual(publish.derive_title(self.root), "What is hard to learn?")

    def test_title_absent(self):
        write(self.root / "main.typ", "#show: paper.with()\n")
        self.assertIsNone(publish.derive_title(self.root))

    def test_title_no_entry(self):
        self.assertIsNone(publish.derive_title(self.root))


class TestPublishCLI(TreeCase):
    def test_rejects_non_directory(self):
        self.assertEqual(publish.main([str(self.root / "nope")]), 2)

    def test_rejects_tree_without_entry(self):
        self.assertEqual(publish.main([str(self.root)]), 2)

    def test_check_returns_zero_when_clean(self):
        self.make_tree({"01-a": "value 0.5\n"}, results=[{"acc": 0.5}])
        self.assertEqual(publish.main([str(self.root), "--check"]), 0)

    def test_check_returns_nonzero_on_error(self):
        write(self.root / "main.typ", '#include "/sections/gone.typ"\n')
        self.assertEqual(publish.main([str(self.root), "--check"]), 1)

    def test_status_does_not_publish(self):
        self.make_tree({"01-a": "value 0.5\n"}, results=[{"acc": 0.5}])
        self.assertEqual(publish.main([str(self.root), "--status"]), 0)
        self.assertFalse((self.root / typst_report.BUILD_FILE).exists())


# ──────────────────────────── R2 key plumbing ───────────────────────────────


class TestSaveMediaKeys(unittest.TestCase):
    """save_media must not reach the network; the fake records what it got."""

    def setUp(self):
        self.calls = []
        self._real = media.upload_media

        def fake(r2_key, data, content_type=None, cache_control=None):
            self.calls.append({"key": r2_key, "content_type": content_type,
                               "cache_control": cache_control})
            return {"success": True, "url": f"https://example.test/{r2_key}"}

        media.upload_media = fake

    def tearDown(self):
        media.upload_media = self._real

    def test_default_key_is_date_prefixed(self):
        media.save_media("f.svg", b"x", "image/svg+xml")
        key = self.calls[0]["key"]
        self.assertTrue(key.startswith("seewhy/"))
        self.assertRegex(key, r"^seewhy/\d{2}-\d{2}-\d{2}/f\.svg$")
        self.assertIsNone(self.calls[0]["cache_control"])

    def test_key_dir_replaces_the_date(self):
        media.save_media("p.pdf", b"x", "application/pdf",
                         key_dir="paper", cache_control="public, max-age=300")
        self.assertEqual(self.calls[0]["key"], "seewhy/paper/p.pdf")
        self.assertEqual(self.calls[0]["cache_control"], "public, max-age=300")

    def test_stable_key_is_reused_across_calls(self):
        """The whole point: a shared link must resolve to the current version."""
        media.save_media("p.pdf", b"one", "application/pdf", key_dir="paper")
        media.save_media("p.pdf", b"two", "application/pdf", key_dir="paper")
        self.assertEqual(self.calls[0]["key"], self.calls[1]["key"])


class TestSignedCacheControl(unittest.TestCase):
    """Cache-Control has to be signed, not merely sent: SigV4 covers every
    header in the dict, so adding it after signing would 403."""

    def setUp(self):
        from shared_lib import r2
        self.r2 = r2
        self._saved = (r2.R2_ACCESS_KEY_ID, r2.R2_SECRET_ACCESS_KEY,
                       r2.R2_ENDPOINT_URL, r2.R2_BUCKET_NAME)
        r2.R2_ACCESS_KEY_ID = "test-key"
        r2.R2_SECRET_ACCESS_KEY = "test-secret"
        r2.R2_ENDPOINT_URL = "https://example.test"
        r2.R2_BUCKET_NAME = "bucket"
        self._validate = r2._validate_config
        r2._validate_config = lambda: None

    def tearDown(self):
        (self.r2.R2_ACCESS_KEY_ID, self.r2.R2_SECRET_ACCESS_KEY,
         self.r2.R2_ENDPOINT_URL, self.r2.R2_BUCKET_NAME) = self._saved
        self.r2._validate_config = self._validate

    def test_cache_control_is_in_signed_headers(self):
        headers = {"Content-Type": "application/pdf", "Cache-Control": "public, max-age=300",
                   "Host": "example.test"}
        auth = self.r2._get_aws_signature_v4("PUT", "/bucket/k", headers, b"x")
        self.assertIn("cache-control", auth["Authorization"])

    def test_signature_changes_with_cache_control(self):
        base = {"Content-Type": "application/pdf", "Host": "example.test"}
        without = self.r2._get_aws_signature_v4("PUT", "/bucket/k", dict(base), b"x")
        with_cc = self.r2._get_aws_signature_v4(
            "PUT", "/bucket/k", {**base, "Cache-Control": "no-cache"}, b"x")
        self.assertNotEqual(without["Authorization"], with_cc["Authorization"])


# ───────────────────────── compiler-dependent checks ────────────────────────


@unittest.skipUnless(HAS_TYPST, "typst not installed (expected on the Mac)")
class TestCompilation(TreeCase):
    def test_scaffolded_paper_compiles(self):
        scaffold_paper(self.root, title="T", subtitle="s", date="2026-08-12")
        pdf = typst_report.compile_report(self.root, fmt="pdf")
        self.assertTrue(pdf.startswith(b"%PDF"))

    def test_final_status_rejects_unwritten_todo(self):
        """#todo is a red box in a draft and a hard error in a final build."""
        scaffold_paper(self.root, title="T")
        entry = self.root / "main.typ"
        entry.write_text(entry.read_text().replace('status: "draft"', 'status: "final"'),
                         encoding="utf-8")
        with self.assertRaises(Exception):
            typst_report.compile_report(self.root, fmt="pdf")

    def test_render_figure_is_a_single_page(self):
        from shared_lib.typst_plot import line_chart, long_form
        fig = line_chart("probe", long_form([1, 2, 3], {"a": [0.1, 0.5, 0.9]}),
                         x="x", y="y", colour="series")
        svg = typst_report.render_figure(fig, fmt="svg").decode()
        self.assertTrue(svg.lstrip().startswith("<svg"))
        self.assertEqual(svg.count("<text"), 0, "text must be outlined for <img> rendering")

    def test_preview_section_with_cross_reference(self):
        """A section referring to a label in a sibling used to fail the whole
        compile with `label <sec:x> does not exist`."""
        scaffold_paper(self.root, title="T")
        write(self.root / "sections" / "06-results.typ",
              '#import "/template.typ": *\n= Results\n\nsee @appendix-repro and @sec:gone\n')
        out = typst_report.preview_section(self.root, "06-results",
                                           self.root / "out.pdf", fmt="pdf")
        self.assertTrue(out.exists() and out.stat().st_size > 0)

    def test_preview_section_with_citation(self):
        """A citing section could not be previewed: the temp entry had no
        bibliography, so Typst could not resolve the key."""
        self.make_tree(
            {"01-a": '#import "/template.typ": *\n= A\n\nas shown @smith2020\n'},
            bib='@article{smith2020, title={T}, author={S}, year={2020},}\n',
        )
        scaffold_paper(self.root, title="T", sections=("01-a",))
        out = typst_report.preview_section(self.root, "01-a", self.root / "out.pdf", fmt="pdf")
        self.assertTrue(out.exists() and out.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
