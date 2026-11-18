import ast
import re
from core.models import LayoutSpec

TOKEN = "#tensor_ext.layout<"

def _attribute_end(text, opening):
    depth, quoted, escaped = 1, False, False
    for pos in range(opening + 1, len(text)):
        ch = text[pos]
        if quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
        else:
            if ch == '"':
                quoted = True
            elif ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
                if depth == 0:
                    return pos
    raise ValueError("Unterminated tensor_ext.layout attribute")

def _isl_string(body):
    body = body.strip()
    if not body.startswith('"'):
        raise ValueError("Expected a quoted ISL string in tensor_ext.layout")
    escaped = False
    end = None
    for i in range(1, len(body)):
        ch = body[i]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            end = i
            break
    if end is None:
        raise ValueError("Unterminated ISL string")
    literal = body[:end + 1]
    try:
        return ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        return literal[1:-1].replace(r'\\"', '"').replace(r'\\\\', '\\')

def extract_layouts(mlir):
    found, cursor, ordinal = [], 0, 0
    while True:
        start = mlir.find(TOKEN, cursor)
        if start < 0:
            break
        opening = start + len(TOKEN) - 1
        end = _attribute_end(mlir, opening)
        payload = _isl_string(mlir[opening + 1:end])
        ordinal += 1
        line_prefix = mlir[mlir.rfind("\n", 0, start) + 1:start]
        alias = re.search(r"(#[\w.$-]+)\s*=\s*$", line_prefix)
        name = alias.group(1) if alias else f"inline_layout_{ordinal}"
        kind = "alias definition" if alias else "inline attribute"
        found.append(LayoutSpec(name, payload, kind, start, end + 1))
        cursor = end + 1

    # Keep each inline occurrence, but avoid duplicate alias definitions.
    result, aliases = [], set()
    for item in found:
        if item.source_kind == "alias definition":
            if item.name in aliases:
                continue
            aliases.add(item.name)
        result.append(item)
    return result
