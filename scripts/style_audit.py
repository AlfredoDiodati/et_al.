"""List every line of a tracked file that breaks README's "Code, comment and
documentation style" policy, written to out/style_audit_report.txt.

Run from anywhere: python3 scripts/style_audit.py

Two kinds of rule are checked. The exact rules (non-ASCII in a comment,
decoration in a comment, alignment spacing in code, emojis) are decided by
the characters on the line, so every hit is a violation. The phrase rules
(marketing register, bold takeaways, reactionary comments, invented labels)
match a list of words and phrases, so a hit is a candidate a person still
has to judge, and a violation phrased without any listed word is missed.
The rules that need the meaning of the text (a comment that restates its
line, a number without its setup, a name that differs from the paper's) are
not checked at all.
"""

import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "out" / "style_audit_report.txt"

C_SUFFIXES = {".c", ".h"}
HASH_COMMENT_SUFFIXES = {".R", ".sh"}

# Runs of one repeated character, as in ---------- or ##########.
DECORATION = re.compile(r"([-=#*~_+/])\1{3,}")

EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]"
)

MARKETING_PHRASES = [
    r"dramatic(ally)?", r"stagger(ing|ingly)", r"severe(ly)?", r"huge(ly)?",
    r"massive(ly)?", r"enormous(ly)?", r"remarkabl[ey]", r"impressive(ly)?",
    r"incredibl[ey]", r"striking(ly)?", r"crucial(ly)?", r"surprising(ly)?",
    r"interesting(ly)?", r"notabl[ey]", r"spectacular(ly)?", r"brutal(ly)?",
    r"blazing", r"game[- ]changer", r"worth noting", r"it turned out",
    r"turns out", r"the big one", r"here's", r"key insight", r"the punchline",
    r"the catch", r"the trick", r"the real (story|win|problem|culprit|lesson)",
    r"a real, ", r"by far", r"strikingly", r"wildly", r"vastly",
]
MARKETING = re.compile(r"\b(" + "|".join(MARKETING_PHRASES) + r")\b", re.I)
EXCLAMATION = re.compile(r"[A-Za-z0-9)\]]!(\s|$)")

REACTIONARY_PHRASES = [
    r"not argued", r"measured, not", r"rather than (transcribed|guessed|assumed|argued)",
    r"load-bearing", r"this is why", r"that is why", r"which is why", r"on purpose",
    r"used to", r"previously", r"no longer", r"originally",
    r"was (removed|replaced|changed|rewritten|dropped|reverted)",
    r"(earlier|older|old|previous|original|first) (version|code|implementation|draft)",
    r"before (the|this) (fix|change|rewrite|refactor)", r"we (tried|found)",
    r"reviewer", r"review", r"objection", r"in this session",
]
REACTIONARY = re.compile(r"\b(" + "|".join(REACTIONARY_PHRASES) + r")\b", re.I)

INVENTED_LABELS = re.compile(
    r"\b(path|item|variant|option|approach|candidate|version) [A-H]\b"
    r"|\bitem \d+\b|\bthe (sweep|prelude|shortcut)\b|\bv\d\b",
    re.I,
)

# A bold sentence opening a prose paragraph, as in "**A bug was found.** ...".
BOLD_SENTENCE_OPENER = re.compile(r"^\*\*[^*]+[.:!?]\*\*")


def tracked_files():
    listing = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.split("\n")
    return [ROOT / name for name in listing if name]


def read_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def split_c_lines(text):
    """Per line, the code with string and character literals blanked to 'x'
    and the comment text, plus whether the comment follows code on its line.
    Blanking keeps column positions so spacing inside a literal is never
    taken for alignment."""
    lines = []
    code, comment = [], []
    state = "code"
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        following = text[index + 1] if index + 1 < length else ""
        if char == "\n":
            lines.append(("".join(code), "".join(comment)))
            code, comment = [], []
            if state == "line_comment":
                state = "code"
            index += 1
            continue
        if state == "code":
            if char == "/" and following == "/":
                state = "line_comment"
                comment.append("//")
                index += 2
                continue
            if char == "/" and following == "*":
                state = "block_comment"
                comment.append("/*")
                index += 2
                continue
            if char in "\"'":
                state = "string" if char == '"' else "char"
            code.append(char)
        elif state in ("line_comment", "block_comment"):
            if state == "block_comment" and char == "*" and following == "/":
                comment.append("*/")
                state = "code"
                index += 2
                continue
            comment.append(char)
        else:
            closing = '"' if state == "string" else "'"
            if char == "\\" and following:
                code.append("xx")
                index += 2
                continue
            if char == closing:
                state = "code"
                code.append(char)
            else:
                code.append("x")
        index += 1
    lines.append(("".join(code), "".join(comment)))
    return lines


def split_hash_lines(text):
    """Same split for languages whose comments start with # (R, shell,
    Makefile), tracking only double and single quotes."""
    lines = []
    for raw in text.split("\n"):
        code, quote = [], None
        comment = ""
        for position, char in enumerate(raw):
            if quote:
                code.append(char if char == quote else "x")
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
                code.append(char)
            elif char == "#":
                comment = raw[position:]
                break
            else:
                code.append(char)
        lines.append(("".join(code), comment))
    return lines


def split_python_lines(text):
    """Same split for Python through the tokenizer, with docstrings counted
    as comments. Returns None when the file does not tokenize."""
    raw_lines = text.split("\n")
    code = [list(line) for line in raw_lines]
    comment = [""] * len(raw_lines)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        tree = ast.parse(text)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return None

    docstring_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                docstring_lines.update(range(body[0].lineno, body[0].end_lineno + 1))

    for token in tokens:
        (start_row, start_col), (end_row, end_col) = token.start, token.end
        if token.type == tokenize.COMMENT:
            comment[start_row - 1] = token.string
            for column in range(start_col, len(code[start_row - 1])):
                code[start_row - 1][column] = " "
        elif token.type == tokenize.STRING:
            for row in range(start_row, end_row + 1):
                first = start_col if row == start_row else 0
                last = end_col if row == end_row else len(code[row - 1])
                if row in docstring_lines:
                    comment[row - 1] += raw_lines[row - 1][first:last]
                for column in range(first, last):
                    code[row - 1][column] = "x"
    return [("".join(characters).rstrip(), comment[row]) for row, characters in enumerate(code)], tokens


def alignment_gaps(code_line):
    """Runs of two or more spaces, or a tab, between two non-space characters
    of code, ignoring leading indentation."""
    stripped = code_line.rstrip()
    body_start = len(stripped) - len(stripped.lstrip())
    return [match.start() for match in re.finditer(r"(?<=\S)( {2,}|\t+)(?=\S)", stripped[body_start:])]


def comment_alignment(comment_text):
    """Runs of two or more spaces inside a comment, other than the one after
    a sentence end, which line text up in columns."""
    inner = comment_text.lstrip("/#*").lstrip()
    return re.search(r"(?<=[^\s.?!:])( {2,}|\t)(?=\S)", inner) is not None


def excerpt(line):
    line = line.strip()
    return line if len(line) <= 160 else line[:157] + "..."


def audit_source_file(path, relative, text, findings):
    raw_lines = text.split("\n")
    python_tokens = None
    if path.suffix in C_SUFFIXES:
        split = split_c_lines(text)
    elif path.suffix == ".py":
        result = split_python_lines(text)
        if result is None:
            findings["unparsed"].append(f"{relative}: does not tokenize as Python, not checked")
            return
        split, python_tokens = result
    else:
        split = split_hash_lines(text)

    for number, (code_line, comment_text) in enumerate(split, start=1):
        raw = raw_lines[number - 1] if number <= len(raw_lines) else ""
        where = f"{relative}:{number}: {excerpt(raw)}"
        if EMOJI.search(raw):
            findings["emoji"].append(where)
        if comment_text:
            if any(ord(char) > 127 for char in comment_text):
                findings["comment_non_ascii"].append(where)
            if DECORATION.search(comment_text):
                findings["comment_decoration"].append(where)
            if comment_alignment(comment_text):
                findings["comment_columns"].append(where)
            if REACTIONARY.search(comment_text):
                findings["comment_reactionary"].append(where)
            if MARKETING.search(comment_text):
                findings["comment_marketing"].append(where)
            if INVENTED_LABELS.search(comment_text):
                findings["comment_labels"].append(where)
        is_makefile_recipe = path.name == "Makefile" and raw.startswith("\t")
        if not is_makefile_recipe and alignment_gaps(code_line):
            findings["code_alignment"].append(where)

        trailing_gap = None
        if comment_text and code_line.strip() and path.suffix != ".py":
            trailing_gap = len(code_line) - len(code_line.rstrip())
        if trailing_gap is not None and trailing_gap >= 2:
            findings["trailing_comment_padding"].append(where)

    if python_tokens is not None:
        previous = None
        for token in python_tokens:
            if token.type == tokenize.COMMENT and previous is not None and previous.end[0] == token.start[0] and previous.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                gap = token.start[1] - previous.end[1]
                row = token.start[0]
                where = f"{relative}:{row}: {excerpt(raw_lines[row - 1])}"
                if gap > 2:
                    findings["trailing_comment_padding"].append(where)
                elif gap == 2:
                    findings["python_two_space_inline_comment"].append(where)
            if token.type not in (tokenize.NL, tokenize.COMMENT):
                previous = token
            elif token.type == tokenize.NL:
                previous = None


def audit_markdown_file(relative, text, findings):
    in_fence = False
    previous_blank = True
    for number, raw in enumerate(text.split("\n"), start=1):
        where = f"{relative}:{number}: {excerpt(raw)}"
        if raw.lstrip().startswith("```"):
            in_fence = not in_fence
            previous_blank = False
            continue
        if EMOJI.search(raw):
            findings["emoji"].append(where)
        if in_fence:
            continue
        prose = re.sub(r"`[^`]*`", "", raw)
        if MARKETING.search(prose):
            findings["doc_marketing"].append(where)
        if EXCLAMATION.search(prose):
            findings["doc_exclamation"].append(where)
        if INVENTED_LABELS.search(prose):
            findings["doc_labels"].append(where)
        if previous_blank and BOLD_SENTENCE_OPENER.match(raw):
            findings["doc_bold_opener"].append(where)
        previous_blank = raw.strip() == ""


SECTIONS = [
    ("emoji", "Emoji or pictographic symbol, any file", "exact"),
    ("comment_non_ascii", "Non-ASCII character in a comment", "exact"),
    ("comment_decoration", "Divider or banner made of a repeated character in a comment", "exact"),
    ("code_alignment", "Two or more spaces, or a tab, between tokens of code (alignment padding)", "exact"),
    ("trailing_comment_padding", "Two or more spaces before a trailing comment (more than two in Python)", "exact"),
    ("comment_columns", "Two or more spaces inside a comment, not after a sentence end (text lined up in columns)", "exact"),
    ("python_two_space_inline_comment", "Exactly two spaces before a Python inline comment (the PEP 8 convention; the policy asks for one)", "exact"),
    ("comment_reactionary", "Comment containing a phrase typical of defending the code or narrating its history", "candidate"),
    ("comment_marketing", "Comment containing a marketing-register word", "candidate"),
    ("comment_labels", "Comment containing a label of the form 'path A', 'item 3', 'the sweep'", "candidate"),
    ("doc_marketing", "Documentation line containing a marketing-register word", "candidate"),
    ("doc_exclamation", "Documentation line containing an exclamation mark outside code", "candidate"),
    ("doc_bold_opener", "Documentation paragraph opening with a bold sentence (possible bolded takeaway)", "candidate"),
    ("doc_labels", "Documentation line containing a label of the form 'path A', 'item 3', 'the sweep'", "candidate"),
    ("unparsed", "Files that could not be checked", "exact"),
]


def main():
    findings = defaultdict(list)
    checked = defaultdict(int)
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("_temp/") or not path.is_file():
            continue
        is_source = path.suffix in C_SUFFIXES | HASH_COMMENT_SUFFIXES | {".py"} or path.name == "Makefile"
        if not is_source and path.suffix != ".md":
            continue
        text = read_text(path)
        if text is None:
            findings["unparsed"].append(f"{relative}: not UTF-8, not checked")
            continue
        if path.suffix == ".md":
            audit_markdown_file(relative, text, findings)
            checked["Markdown"] += 1
        else:
            audit_source_file(path, relative, text, findings)
            checked["source"] += 1

    out = []
    out.append("Style audit against README's \"Code, comment and documentation style\" policy.")
    out.append(f"Files checked: {checked['source']} source files (.c .h .py .R .sh Makefile), {checked['Markdown']} Markdown files, all tracked by git, _temp/ excluded.")
    out.append("exact: every line listed breaks the rule. candidate: a listed word or phrase matched; a person decides.")
    out.append("Not checked: comments that restate their line, numbers given without their setup, names that differ from the paper's.")
    out.append("")
    out.append("Counts (lines):")
    for key, title, kind in SECTIONS:
        out.append(f"{len(findings[key])} [{kind}] {title}")
    out.append("")
    for key, title, kind in SECTIONS:
        if not findings[key]:
            continue
        out.append(f"[{kind}] {title}: {len(findings[key])} lines")
        out.extend(findings[key])
        out.append("")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
