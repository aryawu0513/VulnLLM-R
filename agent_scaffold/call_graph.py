"""
Call graph extraction for C projects using tree-sitter.
Builds: function_name -> body mapping, and a call graph for path sampling.
"""

import random
from pathlib import Path

try:
    # tree-sitter-languages (used by VulnLLM-R venv, tree-sitter ~0.21)
    from tree_sitter_languages import get_parser as _ts_lang_get_parser

    def _get_parser(language: str):
        return _ts_lang_get_parser(language)

except ImportError:
    # Newer tree-sitter >=0.22 with per-language packages (tree-sitter-c etc.)
    from tree_sitter import Language, Parser

    def _get_parser(language: str):
        if language in ("c", "cpp"):
            import tree_sitter_c as ts_c
            lang = Language(ts_c.language())
        elif language == "python":
            import tree_sitter_python as ts_py
            lang = Language(ts_py.language())
        elif language == "java":
            import tree_sitter_java as ts_java
            lang = Language(ts_java.language())
        else:
            raise ValueError(f"Unsupported language: {language}")
        return Parser(lang)


def _get_name_node(node):
    """Recursively find the identifier inside a declarator node."""
    if node.type == "identifier":
        return node
    for field in ("declarator",):
        child = node.child_by_field_name(field)
        if child:
            result = _get_name_node(child)
            if result:
                return result
    # pointer_declarator, function_declarator have their identifier buried in children
    if node.type in ("pointer_declarator", "function_declarator", "abstract_function_declarator"):
        for child in node.children:
            if child.type in ("identifier", "pointer_declarator", "function_declarator"):
                result = _get_name_node(child)
                if result:
                    return result
    return None


def _node_text(node, source_bytes: bytes) -> str:
    """Extract node text, preferring node.text (tree-sitter >=0.22) then byte slicing."""
    if hasattr(node, "text") and node.text is not None:
        return node.text.decode("utf-8", errors="replace")
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def extract_functions(source: str, language: str = "c") -> dict[str, str]:
    """Return {function_name: full_function_source} for all top-level functions."""
    parser = _get_parser(language)
    source_bytes = source.encode("utf-8", errors="replace")
    tree = parser.parse(source_bytes)
    functions = {}

    def visit(node):
        if node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            if declarator:
                name_node = _get_name_node(declarator)
                if name_node:
                    name = _node_text(name_node, source_bytes)
                    body = _node_text(node, source_bytes)
                    functions[name] = body
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return functions


def extract_calls(func_body: str, language: str = "c") -> set[str]:
    """Return names of all functions called within func_body."""
    parser = _get_parser(language)
    source_bytes = func_body.encode("utf-8", errors="replace")
    tree = parser.parse(source_bytes)
    calls = set()

    def visit(node):
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node and func_node.type == "identifier":
                calls.add(_node_text(func_node, source_bytes))
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return calls


def parse_project(project_dir: str, language: str = "c") -> dict[str, str]:
    """Parse all source files under project_dir; return {func_name: body}."""
    ext_map = {
        "c": [".c", ".h"],
        "cpp": [".cpp", ".cc", ".cxx", ".h", ".hpp"],
        "python": [".py"],
        "java": [".java"],
    }
    extensions = set(ext_map.get(language, [".c"]))
    all_functions: dict[str, str] = {}

    for path in sorted(Path(project_dir).rglob("*")):
        if path.suffix in extensions and path.is_file():
            try:
                source = path.read_text(errors="replace")
                funcs = extract_functions(source, language)
                all_functions.update(funcs)
            except Exception as e:
                print(f"[warn] skipping {path}: {e}")

    return all_functions


def build_call_graph(functions: dict[str, str], language: str = "c") -> dict[str, set[str]]:
    """Build call graph restricted to project-internal functions.
    Returns {caller: {callee, ...}}."""
    known = set(functions.keys())
    graph: dict[str, set[str]] = {}
    for name, body in functions.items():
        callees = extract_calls(body, language) & known
        callees.discard(name)  # no self-loops
        graph[name] = callees
    return graph


def find_entry_points(graph: dict[str, set[str]]) -> list[str]:
    """Functions not called by any other function in the project (call graph roots)."""
    called = {callee for callees in graph.values() for callee in callees}
    roots = [name for name in graph if name not in called]
    if "main" in roots:
        return ["main"]
    return roots or list(graph.keys())[:1]


def find_paths(
    graph: dict[str, set[str]],
    source: str,
    target: str,
    max_paths: int = 3,
    max_depth: int = 15,
) -> list[list[str]]:
    """DFS to find up to max_paths simple paths from source to target."""
    if source == target:
        return [[target]]

    all_paths: list[list[str]] = []
    # stack entries: (current_node, path_so_far, visited_set)
    stack = [(source, [source], {source})]

    while stack and len(all_paths) < 50:
        node, path, visited = stack.pop()
        if len(path) > max_depth:
            continue
        for neighbor in graph.get(node, set()):
            if neighbor in visited:
                continue
            new_path = path + [neighbor]
            if neighbor == target:
                all_paths.append(new_path)
            else:
                stack.append((neighbor, new_path, visited | {neighbor}))

    if len(all_paths) > max_paths:
        return random.sample(all_paths, max_paths)
    return all_paths


def get_context_functions(
    graph: dict[str, set[str]],
    functions: dict[str, str],
    entry: str,
    target: str,
    n_paths: int = 3,
) -> list[tuple[str, str]]:
    """Return [(name, body)] for all functions on sampled paths from entry to target
    (excluding the target itself), plus direct callees of the target so the model
    can see what allocation/helper functions could return NULL."""
    paths = find_paths(graph, entry, target, max_paths=n_paths)
    seen: set[str] = set()
    context: list[tuple[str, str]] = []
    for path in paths:
        for name in path:
            if name != target and name not in seen:
                seen.add(name)
                if name in functions:
                    context.append((name, functions[name]))
    # Also include direct callees of the target so the model can inspect
    # allocation helpers (e.g. make_buffer) without needing a retrieval round.
    for callee in graph.get(target, set()):
        if callee not in seen and callee in functions:
            seen.add(callee)
            context.append((callee, functions[callee]))
    return context
