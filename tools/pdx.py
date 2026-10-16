"""Minimal reader for Paradox script files (Victoria 2 and Victoria 3).

A file parses to a Block: an ordered list whose items are either
(key, op, value) tuples or bare values (as in `{ 1 2 3 }`). Values are str or
Block. Typed values such as Vic3's `rgb { 1 2 3 }` become a Block whose `tag`
is "rgb". Quotes are stripped from strings.
"""
import re
from pathlib import Path

_TOKEN = re.compile(r'''
    (?P<ws>\s+)
  | (?P<comment>\#[^\n]*)
  | (?P<string>"(?:[^"\\]|\\.)*"?)
  | (?P<op><=|>=|!=|==|\?=|=|<|>)
  | (?P<lbrace>\{)
  | (?P<rbrace>\})
  | (?P<word>(?:[^\s=<>!{}\#"]|!(?!=))+)
''', re.VERBOSE)


class Block(list):
    tag = None

    def pairs(self):
        return [item for item in self if isinstance(item, tuple)]

    def values(self):
        return [item for item in self if not isinstance(item, tuple)]

    def get_all(self, key):
        return [item[2] for item in self if isinstance(item, tuple) and item[0] == key]

    def get(self, key, default=None):
        found = self.get_all(key)
        return found[-1] if found else default


def _tokenize(text):
    tokens = []
    for m in _TOKEN.finditer(text):
        kind = m.lastgroup
        if kind in ("ws", "comment"):
            continue
        value = m.group()
        if kind == "string":
            value = value[1:-1] if value.endswith('"') and len(value) > 1 else value[1:]
        tokens.append((kind, value))
    return tokens


def parse_text(text):
    tokens = _tokenize(text)
    n = len(tokens)
    pos = 0

    def parse_value():
        nonlocal pos
        kind, value = tokens[pos]
        pos += 1
        if kind == "lbrace":
            return parse_block()
        if kind == "word" and pos < n and tokens[pos][0] == "lbrace":
            pos += 1
            block = parse_block()
            block.tag = value
            return block
        return value

    def parse_block():
        nonlocal pos
        block = Block()
        while pos < n:
            kind, value = tokens[pos]
            if kind == "rbrace":
                pos += 1
                return block
            if pos + 1 < n and tokens[pos + 1][0] == "op" and kind in ("word", "string"):
                op = tokens[pos + 1][1]
                pos += 2
                if pos >= n:
                    break
                block.append((value, op, parse_value()))
            elif kind == "op":
                pos += 1  # stray operator, skip
            else:
                block.append(parse_value())
        return block

    # The top level is a block without braces. parse_block returns early at a
    # stray closing brace (common in hand-edited mod files); keep going after it.
    root = Block()
    while pos < n:
        root.extend(parse_block())
    return root


_NEEDS_QUOTES = re.compile(r'[\s{}=<>#"]')


def format_scalar(value):
    value = str(value)
    if value == "" or _NEEDS_QUOTES.search(value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _format_value(value, indent, tab):
    if not isinstance(value, Block):
        return format_scalar(value)
    prefix = value.tag or ""
    if all(not isinstance(item, (tuple, Block)) for item in value):
        inner = " ".join(format_scalar(item) for item in value)
        return f"{prefix}{{ {inner} }}" if inner else f"{prefix}{{ }}"
    return f"{prefix}{{\n{dumps(value, indent + 1, tab)}\n{tab * indent}}}"


def dumps(block, indent=0, tab="\t"):
    """Serialise a Block (or list of (key, op, value) tuples / bare values)."""
    lines = []
    for item in block:
        if isinstance(item, tuple):
            key, op, value = item
            lines.append(f"{tab * indent}{format_scalar(key)} {op} {_format_value(value, indent, tab)}")
        else:
            lines.append(tab * indent + _format_value(item, indent, tab))
    return "\n".join(lines)


def read_text(path):
    """Read a game file: UTF-8 (with or without BOM), falling back to cp1252 (Vic2)."""
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def parse_file(path):
    return parse_text(read_text(path))
