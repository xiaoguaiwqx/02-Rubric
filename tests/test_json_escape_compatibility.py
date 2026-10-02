import json
import unittest

from structured_rubrics.utils import parse_json


class TestJsonEscapeCompatibility(unittest.TestCase):
    def test_invalid_escapes_are_literal(self):
        raw = r'{"thought":"\sqrt{5} + \pi", "answer":"A"}'
        self.assertEqual(parse_json(raw, allow_invalid_escapes=True),
                         {"thought": r"\sqrt{5} + \pi", "answer": "A"})
        with self.assertRaises(ValueError):
            parse_json(raw)

    def test_valid_escapes_unchanged(self):
        value = {"thought": 'newline\n tab\t quote" slash\\sqrt{5}', "answer": "None"}
        raw = json.dumps(value)
        self.assertEqual(parse_json(raw, allow_invalid_escapes=True), value)

    def test_other_errors_still_fail(self):
        for raw in ('{"answer": A}', '{"answer":"A",}',
                    r'{"thought":"\uXYZW","answer":"B"}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_json(raw, allow_invalid_escapes=True)

    def test_even_backslashes_not_repaired(self):
        raw = r'{"thought":"\\sqrt{5} then \pi", "answer":"B"}'
        self.assertEqual(parse_json(raw, allow_invalid_escapes=True)["thought"],
                         r"\sqrt{5} then \pi")
