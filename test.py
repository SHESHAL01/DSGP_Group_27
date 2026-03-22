import unittest
import pandas as pd
import numpy as np
import ast
import re

# ─────────────────────────────────────────────────────────────
#  Copy the functions under test here so the test file is
#  self-contained.  In a real project you would import them:
#      from your_module import is_compound_abbreviation, ...
# ─────────────────────────────────────────────────────────────

STOPWORDS = {
    "introduction", "intro", "fundamentals", "fundamental",
    "overview", "basics", "basic", "advanced",
    "a", "an", "the", "and", "but", "or", "nor", "if",
    "to", "of", "in", "for", "on", "at", "by", "from",
    "with", "into", "through", "via", "using", "about", "as",
    "until", "while", "during", "before", "after", "above",
    "below", "up", "down", "out", "over", "under", "again",
    "its", "their", "our", "your", "my",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "it", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "they", "them",
    "so", "than", "too", "very", "just", "only", "same",
    "each", "more", "most", "other", "some", "such", "no",
    "not", "both", "own", "between", "here", "there",
    "when", "where", "how", "all", "few", "then", "once",
}

def is_compound_abbreviation(token: str) -> bool:
    parts = token.split("/")
    pattern = re.compile(r"^[A-Z][A-Z0-9]*(-[A-Z0-9]+)*$")
    return all(pattern.match(p) for p in parts)

def split_comma_outside_parens(text: str) -> list:
    parts, current, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1; current.append(ch)
        elif ch == ")":
            depth -= 1; current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip()); current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]

def expand_parentheses(text: str) -> list:
    results = []
    remaining = text
    while "(" in remaining and ")" in remaining:
        open_i  = remaining.index("(")
        close_i = remaining.index(")")
        before  = remaining[:open_i].strip().rstrip(",").strip()
        inside  = remaining[open_i + 1 : close_i].strip()
        after   = remaining[close_i + 1 :].strip().lstrip(",").strip()
        if before:
            results.append(before)
        for chunk in re.split(r",", inside):
            chunk = chunk.strip()
            if not chunk:
                continue
            for part in chunk.split("/"):
                part = part.strip()
                if part:
                    results.append(part)
        remaining = after
    if remaining.strip():
        results.append(remaining.strip())
    return [r for r in results if r]

def split_slash(skill: str) -> list:
    if "/" not in skill:
        return [skill]
    def replace_slash_token(match):
        token = match.group(0)
        if is_compound_abbreviation(token):
            return token
        return token.replace("/", "§")
    processed = re.sub(r"\S+/\S*", replace_slash_token, skill)
    if "§" not in processed:
        return [skill]
    return [p.strip() for p in processed.split("§") if p.strip()]

def remove_stopwords(skill: str) -> str:
    words = skill.split()
    cleaned = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(cleaned)

def clean_skills_cell(cell) -> str:
    if pd.isna(cell):
        return cell
    raw_skills = split_comma_outside_parens(cell)
    final_skills = []
    for skill in raw_skills:
        skill = skill.strip()
        if not skill:
            continue
        expanded = expand_parentheses(skill)
        for part in expanded:
            split_parts = split_slash(part)
            for sp in split_parts:
                sp = remove_stopwords(sp)
                sp = re.sub(r"\s+", " ", sp).strip().lower()
                if sp:
                    final_skills.append(sp)
    return ", ".join(final_skills)

def parse_market_skills(raw: str) -> set:
    try:
        items = ast.literal_eval(raw)
        return {str(t).strip().lower() for t in items if str(t).strip()}
    except (ValueError, SyntaxError):
        raw = raw.strip("[]").replace("'", "").replace('"', "")
        return {t.strip().lower() for t in raw.split(",") if t.strip()}

def phrase_to_tokens(phrase: str) -> set:
    phrase = phrase.lower()
    return {word for word in re.findall(r"[a-z]+", phrase)}

def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return round(len(set_a & set_b) / len(set_a | set_b), 4)


# ═══════════════════════════════════════════════════════════════
#  TEST CLASSES
# ═══════════════════════════════════════════════════════════════

class TestIsCompoundAbbreviation(unittest.TestCase):
    """Tests for is_compound_abbreviation()"""

    def test_single_uppercase_abbreviation(self):
        """IO is a pure uppercase token → compound abbreviation."""
        self.assertTrue(is_compound_abbreviation("IO"))

    def test_classic_io_slash(self):
        """I/O — both parts are single uppercase letters."""
        self.assertTrue(is_compound_abbreviation("I/O"))

    def test_cicd(self):
        """CI/CD is a well-known compound abbreviation."""
        self.assertTrue(is_compound_abbreviation("CI/CD"))

    def test_osi_tcpip(self):
        """OSI/TCP-IP — hyphenated second part is still uppercase."""
        self.assertTrue(is_compound_abbreviation("OSI/TCP-IP"))

    def test_ddl_dml_dcl(self):
        """SQL dialect group DDL/DML/DCL — three uppercase parts."""
        self.assertTrue(is_compound_abbreviation("DDL/DML/DCL"))

    def test_mixed_case_is_not_compound(self):
        """Azure contains lowercase → not a compound abbreviation."""
        self.assertFalse(is_compound_abbreviation("AWS/Azure"))

    def test_lowercase_word(self):
        """Plain lowercase word is never a compound abbreviation."""
        self.assertFalse(is_compound_abbreviation("python"))

    def test_single_uppercase_no_slash(self):
        """AWS without a slash — still qualifies as an abbreviation token."""
        self.assertTrue(is_compound_abbreviation("AWS"))

    def test_number_in_abbreviation(self):
        """S3 (uppercase + digit) is a valid abbreviation part."""
        self.assertTrue(is_compound_abbreviation("S3"))

    def test_empty_string(self):
        """Empty string produces an empty part which does NOT match the pattern."""
        self.assertFalse(is_compound_abbreviation(""))


class TestSplitCommaOutsideParens(unittest.TestCase):
    """Tests for split_comma_outside_parens()"""

    def test_simple_comma_split(self):
        result = split_comma_outside_parens("Python, Java, SQL")
        self.assertEqual(result, ["Python", "Java", "SQL"])

    def test_comma_inside_parens_not_split(self):
        """Commas inside parentheses must NOT be treated as separators."""
        result = split_comma_outside_parens(
            "OOP principles (encapsulation, inheritance, polymorphism)"
        )
        self.assertEqual(len(result), 1)
        self.assertIn("OOP principles", result[0])

    def test_mixed_comma_inside_and_outside_parens(self):
        result = split_comma_outside_parens(
            "Software testing (unit, integration), Python"
        )
        self.assertEqual(len(result), 2)
        self.assertIn("Python", result[1])

    def test_no_comma(self):
        result = split_comma_outside_parens("Machine Learning")
        self.assertEqual(result, ["Machine Learning"])

    def test_empty_string(self):
        result = split_comma_outside_parens("")
        self.assertEqual(result, [])

    def test_nested_parens_deep(self):
        """Deeply nested parens: inner comma must not split at depth > 1."""
        result = split_comma_outside_parens("A (b (c, d)), E")
        self.assertEqual(len(result), 2)

    def test_trailing_comma(self):
        """A trailing comma produces no ghost empty token."""
        result = split_comma_outside_parens("Python, Java,")
        self.assertNotIn("", result)

    def test_single_item(self):
        result = split_comma_outside_parens("Docker")
        self.assertEqual(result, ["Docker"])


class TestExpandParentheses(unittest.TestCase):
    """Tests for expand_parentheses()"""

    def test_version_control_example(self):
        """'Version control (Git)' → ['Version control', 'Git']"""
        result = expand_parentheses("Version control (Git)")
        self.assertEqual(result, ["Version control", "Git"])

    def test_oop_with_multiple_comma_items(self):
        result = expand_parentheses(
            "OOP principles (encapsulation, inheritance, polymorphism, abstraction)"
        )
        self.assertIn("OOP principles", result)
        self.assertIn("encapsulation", result)
        self.assertIn("inheritance", result)
        self.assertIn("polymorphism", result)
        self.assertIn("abstraction", result)

    def test_slash_inside_parens_is_split(self):
        """'Data structures (lists/dicts/tuples)' → slash splits inside parens."""
        result = expand_parentheses("Data structures (lists/dicts/tuples)")
        self.assertIn("lists", result)
        self.assertIn("dicts", result)
        self.assertIn("tuples", result)

    def test_sql_ddl_dml_dcl(self):
        """SQL (DDL/DML/DCL) — slash parts inside parens always split."""
        result = expand_parentheses("SQL (DDL/DML/DCL)")
        self.assertIn("DDL", result)
        self.assertIn("DML", result)
        self.assertIn("DCL", result)

    def test_no_parens(self):
        """No parentheses → returns the whole string as a single element."""
        result = expand_parentheses("Machine Learning")
        self.assertEqual(result, ["Machine Learning"])

    def test_empty_parens(self):
        """Empty parentheses '()' should not crash."""
        result = expand_parentheses("Python ()")
        self.assertIn("Python", result)

    def test_multiple_paren_groups(self):
        """Two separate paren groups are both expanded."""
        result = expand_parentheses("DB (MySQL) caching (Redis)")
        self.assertIn("DB", result)
        self.assertIn("MySQL", result)
        self.assertIn("Redis", result)


class TestSplitSlash(unittest.TestCase):
    """Tests for split_slash()"""

    def test_agile_scrum_split(self):
        """'Agile/Scrum' → ['Agile', 'Scrum'] (mixed case → not compound)."""
        self.assertEqual(split_slash("Agile/Scrum"), ["Agile", "Scrum"])

    def test_file_io_preserved(self):
        """'File I/O' → ['File I/O'] (I/O is a compound abbreviation)."""
        self.assertEqual(split_slash("File I/O"), ["File I/O"])

    def test_cicd_pipeline_preserved(self):
        """'CI/CD Pipeline' → ['CI/CD Pipeline']."""
        self.assertEqual(split_slash("CI/CD Pipeline"), ["CI/CD Pipeline"])

    def test_aws_azure_gcp_split(self):
        """'AWS/Azure/GCP' → three tokens (Azure has lowercase)."""
        result = split_slash("AWS/Azure/GCP")
        self.assertEqual(len(result), 3)
        self.assertIn("AWS", result)
        self.assertIn("Azure", result)
        self.assertIn("GCP", result)

    def test_no_slash(self):
        """No slash → returns the original skill unchanged."""
        self.assertEqual(split_slash("Python"), ["Python"])

    def test_sorting_searching(self):
        """'Sorting/searching algos' → split because 'searching' is lowercase."""
        result = split_slash("Sorting/searching algos")
        self.assertEqual(len(result), 2)

    def test_osi_tcpip_model_preserved(self):
        """'OSI/TCP-IP model' → not split (both parts are uppercase)."""
        result = split_slash("OSI/TCP-IP model")
        self.assertEqual(result, ["OSI/TCP-IP model"])


class TestRemoveStopwords(unittest.TestCase):
    """Tests for remove_stopwords()"""

    def test_removes_intro(self):
        result = remove_stopwords("Introduction to Python")
        self.assertNotIn("Introduction", result)
        self.assertNotIn("to", result)
        self.assertIn("Python", result)

    def test_removes_the(self):
        result = remove_stopwords("the quick brown fox")
        self.assertNotIn("the", result)
        self.assertIn("quick", result)

    def test_no_stopwords(self):
        """Skill with no stopwords must be returned as-is."""
        result = remove_stopwords("Machine Learning")
        self.assertEqual(result, "Machine Learning")

    def test_all_stopwords(self):
        """A phrase made entirely of stopwords becomes an empty string."""
        result = remove_stopwords("introduction to the basics")
        self.assertEqual(result.strip(), "")

    def test_preserves_non_stopwords(self):
        result = remove_stopwords("Fundamentals of Data Science")
        self.assertIn("Data", result)
        self.assertIn("Science", result)
        self.assertNotIn("Fundamentals", result)
        self.assertNotIn("of", result)

    def test_case_insensitive_removal(self):
        """Stopword check is case-insensitive."""
        result = remove_stopwords("The And OR")
        self.assertEqual(result.strip(), "")

    def test_empty_string(self):
        result = remove_stopwords("")
        self.assertEqual(result, "")


class TestCleanSkillsCell(unittest.TestCase):
    """Integration tests for clean_skills_cell()"""

    def test_nan_returns_nan(self):
        result = clean_skills_cell(np.nan)
        self.assertTrue(pd.isna(result))

    def test_simple_comma_list(self):
        result = clean_skills_cell("Python, SQL, Docker")
        skills = [s.strip() for s in result.split(",")]
        self.assertIn("python", skills)
        self.assertIn("sql", skills)
        self.assertIn("docker", skills)

    def test_output_is_lowercase(self):
        """All tokens in the cleaned output must be lowercase."""
        result = clean_skills_cell("Machine Learning, Cloud Computing")
        for token in result.split(","):
            self.assertEqual(token.strip(), token.strip().lower())

    def test_stopwords_removed_from_output(self):
        result = clean_skills_cell("Introduction to Python")
        self.assertNotIn("introduction", result)
        self.assertNotIn("to", result)
        self.assertIn("python", result)

    def test_parentheses_expanded(self):
        result = clean_skills_cell("Version control (Git)")
        self.assertIn("version control", result)
        self.assertIn("git", result)

    def test_commas_inside_parens_expanded(self):
        result = clean_skills_cell(
            "OOP principles (encapsulation, inheritance, polymorphism)"
        )
        self.assertIn("encapsulation", result)
        self.assertIn("inheritance", result)
        self.assertIn("polymorphism", result)

    def test_slash_split_in_output(self):
        """Agile/Scrum should appear as two separate skills."""
        result = clean_skills_cell("Agile/Scrum")
        self.assertIn("agile", result)
        self.assertIn("scrum", result)

    def test_compound_abbreviation_preserved(self):
        """CI/CD must NOT be split and should survive the pipeline intact."""
        result = clean_skills_cell("CI/CD")
        self.assertIn("ci/cd", result)

    def test_extra_whitespace_collapsed(self):
        result = clean_skills_cell("  Python  ,  Java  ")
        for skill in result.split(","):
            self.assertEqual(skill.strip(), skill.strip())  # no leading/trailing space

    def test_empty_string(self):
        result = clean_skills_cell("")
        self.assertEqual(result, "")

    def test_complex_mixed_cell(self):
        """Full pipeline test with commas, parens, slashes, and stopwords."""
        cell = "Software testing (unit, integration, system), Agile/Scrum"
        result = clean_skills_cell(cell)
        self.assertIn("unit", result)
        self.assertIn("integration", result)
        self.assertIn("system", result)
        self.assertIn("agile", result)
        self.assertIn("scrum", result)


class TestParseMarketSkills(unittest.TestCase):
    """Tests for parse_market_skills()"""

    def test_valid_python_list_string(self):
        raw = "['Python', 'SQL', 'Docker']"
        result = parse_market_skills(raw)
        self.assertEqual(result, {"python", "sql", "docker"})

    def test_double_quoted_list_string(self):
        raw = '["Machine Learning", "Deep Learning"]'
        result = parse_market_skills(raw)
        self.assertEqual(result, {"machine learning", "deep learning"})

    def test_fallback_comma_split(self):
        """Malformed string falls back to comma split."""
        raw = "python, sql, docker"
        result = parse_market_skills(raw)
        self.assertIn("python", result)
        self.assertIn("sql", result)
        self.assertIn("docker", result)

    def test_output_is_lowercase(self):
        raw = "['Python', 'SQL']"
        result = parse_market_skills(raw)
        for skill in result:
            self.assertEqual(skill, skill.lower())

    def test_empty_items_excluded(self):
        raw = "['Python', '', '  ']"
        result = parse_market_skills(raw)
        self.assertNotIn("", result)
        self.assertNotIn("  ", result)

    def test_single_item_list(self):
        raw = "['Python']"
        result = parse_market_skills(raw)
        self.assertEqual(result, {"python"})

    def test_returns_set(self):
        raw = "['Python', 'Python']"
        result = parse_market_skills(raw)
        self.assertIsInstance(result, set)
        self.assertEqual(len(result), 1)  # duplicates removed by set


class TestPhraseToTokens(unittest.TestCase):
    """Tests for phrase_to_tokens()"""

    def test_simple_phrase(self):
        result = phrase_to_tokens("Machine Learning")
        self.assertEqual(result, {"machine", "learning"})

    def test_ignores_numbers(self):
        """Digits and purely numeric tokens are excluded (regex is [a-z]+)."""
        result = phrase_to_tokens("Python3")
        self.assertNotIn("3", result)
        self.assertIn("python", result)

    def test_hyphenated_word(self):
        """Hyphens break the word into two tokens."""
        result = phrase_to_tokens("state-of-the-art")
        self.assertIn("state", result)
        self.assertIn("of", result)

    def test_lowercase_conversion(self):
        result = phrase_to_tokens("SQL")
        self.assertEqual(result, {"sql"})

    def test_empty_string(self):
        result = phrase_to_tokens("")
        self.assertEqual(result, set())

    def test_returns_set(self):
        result = phrase_to_tokens("data data science")
        self.assertIsInstance(result, set)
        self.assertIn("data", result)  # deduplication
        self.assertEqual(len(result), 2)


class TestJaccardSimilarity(unittest.TestCase):
    """Tests for jaccard_similarity()"""

    def test_identical_sets(self):
        s = {"python", "sql", "docker"}
        self.assertEqual(jaccard_similarity(s, s), 1.0)

    def test_disjoint_sets(self):
        self.assertEqual(jaccard_similarity({"a", "b"}, {"c", "d"}), 0.0)

    def test_partial_overlap(self):
        a = {"python", "sql", "docker"}
        b = {"python", "sql", "kubernetes"}
        # intersection=2, union=4 → 0.5
        self.assertAlmostEqual(jaccard_similarity(a, b), 0.5, places=3)

    def test_empty_set_a(self):
        self.assertEqual(jaccard_similarity(set(), {"python"}), 0.0)

    def test_empty_set_b(self):
        self.assertEqual(jaccard_similarity({"python"}, set()), 0.0)

    def test_both_empty(self):
        self.assertEqual(jaccard_similarity(set(), set()), 0.0)

    def test_returns_float(self):
        result = jaccard_similarity({"a"}, {"a", "b"})
        self.assertIsInstance(result, float)

    def test_result_rounded_to_4_places(self):
        a = {"a"}
        b = {"a", "b", "c"}
        result = jaccard_similarity(a, b)
        self.assertEqual(result, round(result, 4))

    def test_subset_relationship(self):
        """When A ⊂ B, score = |A| / |B|."""
        a = {"python"}
        b = {"python", "sql", "docker", "kubernetes"}
        expected = round(1 / 4, 4)
        self.assertEqual(jaccard_similarity(a, b), expected)


# ─────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)