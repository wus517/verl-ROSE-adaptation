"""Standalone verifier assembled from the answer rules, legacy math grader,
and reward wrapper.

This file intentionally contains the implementation of all three original
local modules so it can be uploaded and used on its own. It still requires
the third-party packages used by the legacy math grader.

Answer-format rules applied around the legacy verifier.

The ground-truth convention: multiple-choice answers are sorted letters
separated by ASCII commas (``A,C``).
"""

from __future__ import annotations

import importlib
import multiprocessing
import re
import sys
import threading
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_CHOICE_GT_RE = re.compile(r"[A-Z](?:,[A-Z])*")
_LATEX_WRAPPER_RE = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\s*\{([^{}]*)\}")
# After NFKC, full-width commas, semicolons and parentheses become ASCII.
_CHOICE_PRED_RE = re.compile(r"[A-Z(),、;\s]+")


def choice_letters(gt: str) -> str | None:
    """Return the letters of a multiple-choice ground truth such as ``A,C``, else None."""
    if not _CHOICE_GT_RE.fullmatch(gt):
        return None
    letters = gt.split(",")
    if letters != sorted(set(letters)):
        return None
    return "".join(letters)


def parse_choice(pred: str) -> str | None:
    """Parse predicted choice labels into sorted letters, or None if not a pure label list.

    Accepts e.g. ``AC``, ``A、C``, ``(A)(C)``, ``\\text{A, C}`` and full-width letters.
    Rejects any other text (``A或C``) and repeated letters (``II``).
    """
    value = unicodedata.normalize("NFKC", pred).replace("$", "")
    while True:
        unwrapped = _LATEX_WRAPPER_RE.sub(r"\1", value)
        if unwrapped == value:
            break
        value = unwrapped
    if not _CHOICE_PRED_RE.fullmatch(value):
        return None
    letters = re.findall(r"[A-Z]", value)
    if not letters or len(set(letters)) != len(letters):
        return None
    return "".join(sorted(letters))


def split_top_level(text: str, separators: str) -> list[str] | None:
    """Split on separators outside brackets; None if brackets are unbalanced or a part is empty.

    Bracket kinds are counted together so half-open intervals such as ``[1,2)``
    stay one part. A separator escaped as ``\\,`` (LaTeX thin space) is not split.
    """
    parts, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                return None
        elif char in separators and depth == 0 and not (i > 0 and text[i - 1] == "\\"):
            parts.append(text[start:i].strip())
            start = i + 1
    if depth != 0:
        return None
    parts.append(text[start:].strip())
    return parts if all(parts) else None


# A label is a simple identifier before the only "=": x, x_1, x_{1}, \alpha, f(x).
_LABEL_RE = re.compile(
    r"\s*((?:[A-Za-z]|\\[A-Za-z]+)(?:_(?:\{[A-Za-z0-9]+\}|[A-Za-z0-9]))?(?:\([A-Za-z]\))?)\s*=(.*)",
    re.S,
)


def split_label(field: str) -> tuple[str | None, str]:
    """Split ``x_{1}=3`` into a normalized label ``x_1`` and value ``3``; unlabeled fields give (None, field)."""
    if field.count("=") != 1:
        return None, field
    match = _LABEL_RE.fullmatch(field)
    if not match or not match.group(2).strip():
        return None, field
    return re.sub(r"[\s{}]", "", match.group(1)), match.group(2).strip()


_SPACE = r"(?:\s|~|\\[,;:! ])*"
_POWER = r"(?:\^\{?[23]\}?|[²³])?"
# Units allowed inside \text{} / \mathrm{} / \mbox{}; single letters only when wrapped.
_WRAPPED_UNITS = {"%", "cm", "mm", "km", "dm", "kg", "mg", "ml", "mL", "m", "g", "s", "h", "L", "min"}
_WRAPPED_UNIT_RE = re.compile(rf"\\(?:text|mathrm|mbox)\s*\{{{_SPACE}(?P<unit>[^{{}}]*?){_SPACE}\}}{_POWER}$")
_SYMBOL_UNIT_RE = re.compile(r"(?:\^\{\\circ\}|\^\\circ|\\circ|°|\\%|%)$")
_BARE_UNIT_RE = re.compile(rf"(?<![A-Za-z])(?:cm|mm|km|dm|kg|mg|ml|mL){_POWER}$")
_TRAILING_SPACE_RE = re.compile(rf"{_SPACE}$")


def strip_unit(field: str) -> str:
    """Remove one trailing unit (%, degree sign, metric unit) from a numeric field.

    No conversion is done: ``25%`` becomes ``25``, never ``0.25``. The unit is
    kept when nothing remains or the rest ends with a letter (``xcm``, ``2m``).
    """
    value = field.strip()
    match = _WRAPPED_UNIT_RE.search(value)
    if match and match.group("unit") not in _WRAPPED_UNITS:
        match = None
    match = match or _SYMBOL_UNIT_RE.search(value) or _BARE_UNIT_RE.search(value)
    if not match:
        return value
    rest = _TRAILING_SPACE_RE.sub("", value[: match.start()])
    if not rest or re.search(r"[A-Za-z]$", rest):
        return value
    return rest
# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except Exception in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa

"""Provides a math answer grading function with high recall.
Based on HF math_verify, verl, open reasoner zero, etc.
"""

import os
import re
import signal
from itertools import islice, zip_longest
from math import isclose
from typing import Optional

import sympy
from latex2sympy2_extended import latex2sympy
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from pylatexenc import latex2text
from sympy import N, simplify
from sympy.parsing import sympy_parser
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

"""
This code is adapted from: Dr. GRPO (https://github.com/sail-sg/understand-r1-zero/blob/main/understand_r1_zero/math_grader.py).
"""


def timeout_ours(timeout_seconds: int = 8):
    if os.name == "posix":
        import signal

        def decorator(func):
            def handler(signum, frame):
                raise TimeoutError("Operation timed out!")

            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)

                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            return wrapper

        return decorator
    else:
        raise NotImplementedError(f"Unsupported OS: {os.name}")


# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except Exception:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove unit: texts
    for _ in range(2):
        for unit_text in unit_texts:
            # use regex, the prefix should be either the start of the string or a non-alphanumeric character
            # the suffix should be either the end of the string or a non-alphanumeric character
            _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]


REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    This code comes from https://arxiv.org/pdf/2206.14858.pdf, page18.
    """
    # final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def repeatness(s: str):
    def ranks(seq):
        index = {v: i for i, v in enumerate(sorted(set(seq)))}
        return [index[v] for v in seq]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return (cnt * 2 / (n * (n + 1))) > 0.2


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def latex_eval(latex):
    sym = parse_latex(latex)
    val = sym.evalf()
    return sym, val


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


@timeout_ours(timeout_seconds=5)
def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False


def _is_latex_equal(str1, str2):
    try:
        sym1, val1 = latex_eval(str1)
        sym2, val2 = latex_eval(str2)
        if sym1 == sym2 or val1 == val2:
            return True
        else:
            raise ValueError
    except Exception:  # noqa
        try:
            norm1, norm2 = normalize_final_answer(str1), normalize_final_answer(str2)
            sym1, val1 = latex_eval(norm1)
            sym2, val2 = latex_eval(norm2)
            if sym1 == sym2 or val1 == val2:
                return True
        except Exception:  # noqa
            return norm1 == norm2
    return False


def is_latex_equal(given_answer: str, ground_truth: str) -> bool:
    try:
        with timeout(1):
            try:
                if (len(given_answer) > 128 and repeatness(given_answer)) or (
                    len(ground_truth) > 128 and repeatness(ground_truth)
                ):
                    return False
                # First conduct normalized string matching.
                ground_truth_normalized = _normalize(ground_truth)
                given_normalized = _normalize(given_answer)
                if ground_truth_normalized is None:
                    return False
                if ground_truth_normalized == given_normalized:
                    return True

                # Next call math verify.
                given_answer.replace("\n", "")
                ground_truth.replace("\n", "")
                if "$" not in given_answer:
                    given_answer = f"${given_answer}$"
                if "$" not in ground_truth:
                    ground_truth = f"${ground_truth}$"
                return verify(
                    parse(
                        ground_truth,
                        extraction_config=(
                            LatexExtractionConfig(boxed_match_priority=0),
                            ExprExtractionConfig(),
                        ),
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    ),
                    parse(
                        given_answer,
                        extraction_config=(
                            LatexExtractionConfig(boxed_match_priority=0),
                            ExprExtractionConfig(),
                        ),
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    ),
                    timeout_seconds=1,
                )
                # or symbolic_equal(ground_truth, given_answer)
            except Exception:
                return False
    except TimeoutError:
        return False


def is_value_equal(given_answer: str, ground_truth: str) -> bool:
    assert ground_truth is not None
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    str_equal = ground_truth_normalized_mathd == given_answer_normalized_mathd
    try:
        number_equal = float(ground_truth_normalized_mathd) == float(given_answer_normalized_mathd)
        return str_equal or number_equal
    except Exception:
        return str_equal


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub("\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


@timeout_ours(timeout_seconds=5)
def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match
                # (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def _legacy_grade(model_answer: str, gt_answer: str, fast: bool = True):
    if "\\boxed" in gt_answer:
        gt_answer = extract_answer(gt_answer)
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(model_answer, gt_answer)
    if not fast:
        # This mode further uses math_verify to recall originally false positives.
        # Will be a bit slower, and sensitive to bad inputs.
        correct = correct or is_latex_equal(
            model_answer,
            gt_answer,
        )
    return correct


def _legacy_compute_score(model_response, gt_answer, fast=False):
    model_answer = extract_answer(model_response)
    if model_answer is None:
        return False
        # return {
        #     "score": 0.0,
        #     "format_score": 0.0,
        #     "acc": False,
        #     "extracted_gt": gt_answer,
        #     # "extracted_pred": None,
        # }
        # return 0.0, 0.0  # Cannot even parse anything.
    is_correct = False
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = _legacy_grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= _legacy_grade(model_answer, gt, fast)
    # if is_correct:
    #     return {
    #         "score": 1.0,
    #         "format_score": 1.0,
    #         "acc": True,
    #         "extracted_gt": gt_answer,
    #         # "extracted_pred": None,
    #     }
    # else:
    #     return {
    #         "score": 0.0,
    #         "format_score": 1.0,
    #         "acc": False,
    #         "extracted_gt": gt_answer,
    #         # "extracted_pred": None,
    #     }
    return is_correct

# Public aliases for the legacy grader retained in this single-file build.
legacy_grade = _legacy_grade
legacy_compute_score = _legacy_compute_score
legacy_entropy_math_score = _legacy_compute_score

# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Canonical verl 0.1 HPT reward compatibility wrapper."""



THINK_END = "</think>"


def legacy_reward_input(solution_str: str) -> str:
    """Reproduce ``RewardManager``'s reward_impl_version=6 text prefix."""
    return "<think>\n" + solution_str


def final_response(solution_str: str) -> str:
    """有 </think> 时取其后内容，否则使用完整输出。"""
    if THINK_END in solution_str:
        return solution_str.rsplit(THINK_END, 1)[1]
    return solution_str


def ground_truth_answers(ground_truth) -> list[str]:
    """Normalize a ground truth (str, number, or list of them) to unboxed strings."""
    answers = []
    for gt in ground_truth if isinstance(ground_truth, list) else [ground_truth]:
        if isinstance(gt, (int, float)):
            gt = str(gt)
        if not isinstance(gt, str):
            continue
        if "\\boxed" in gt:
            gt = extract_answer(gt)
            if gt is None:
                continue
        answers.append(gt)
    return answers


GT_SEPARATORS = ","
PRED_SEPARATORS = ",，;；"


def field_equal(pred: str, gt: str) -> bool:
    """Compare one answer field; labels must match when both sides have one.

    Units are stripped from both sides first, so 30° matches 30 but 25% does
    not match 0.25 (legacy alone would convert the percent).
    """
    pred_label, pred = split_label(pred)
    gt_label, gt = split_label(gt)
    if pred_label is not None and gt_label is not None and pred_label != gt_label:
        return False
    pred, gt = strip_unit(pred), strip_unit(gt)
    # [L2] The legacy verifier extracts the last box, so pass the field boxed.
    return bool(legacy_entropy_math_score("\\boxed{" + pred + "}", gt))


def grade(final: str, pred: str, gt: str) -> bool:
    """Grade the extracted prediction against one ground-truth answer."""
    letters = choice_letters(gt)
    if letters is not None:
        # [L1] Legacy verdict first; parse the label format only if it fails.
        if legacy_entropy_math_score(final, gt):
            return True
        return parse_choice(pred) == letters
    gt_fields = split_top_level(gt, GT_SEPARATORS)
    if gt_fields is not None and len(gt_fields) > 1:
        # Ordered blanks: never grade the whole string, legacy may ignore order.
        pred_fields = split_top_level(pred, PRED_SEPARATORS)
        if pred_fields is None or len(pred_fields) != len(gt_fields):
            return False
        return all(field_equal(p, g) for p, g in zip(pred_fields, gt_fields))
    return field_equal(pred, gt)


def compute_score_in_process(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    """Grade in a process's main thread, where signal timeouts are supported."""
    del data_source, extra_info, kwargs
    final = final_response(solution_str)
    if final is None:
        return 0.0
    pred = extract_answer(final)
    if pred is None:
        return 0.0
    return float(any(grade(final, pred, gt) for gt in ground_truth_answers(ground_truth)))


_reward_process_pool = None
_reward_process_entry = None
_reward_process_lock = threading.Lock()


def _get_reward_process_pool():
    global _reward_process_pool, _reward_process_entry
    with _reward_process_lock:
        if _reward_process_pool is None:
            # verl loads custom rewards under synthetic module names that spawn
            # cannot import. Import this file by its real deployment filename
            # before creating the pool so children inherit the search path.
            path = Path(__file__).resolve()
            directory = str(path.parent)
            if directory not in sys.path:
                sys.path.insert(0, directory)
            module = importlib.import_module(path.stem)
            if Path(module.__file__).resolve() != path:
                raise ImportError(f"Reward module name {path.stem!r} conflicts with {module.__file__}")
            _reward_process_entry = module.compute_score_in_process
            _reward_process_pool = ProcessPoolExecutor(
                max_workers=4, mp_context=multiprocessing.get_context("spawn")
            )
        return _reward_process_pool, _reward_process_entry


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Thread-safe entry point for naive reward managers; reuse child processes."""
    # The grader ignores these fields; do not require them to be pickleable.
    del data_source, extra_info, kwargs
    pool, entry = _get_reward_process_pool()
    try:
        return pool.submit(entry, None, solution_str, ground_truth).result()
    except TimeoutError:
        # Keep the legacy signal deadlines in the child, scoring a timeout as 0.
        return 0.0
