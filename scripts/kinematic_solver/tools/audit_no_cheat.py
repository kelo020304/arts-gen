"""AST audit for KinematicSolver USD-limit helper boundaries."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_LIMIT_ATTR_NAMES = {"GetLowerLimitAttr", "GetUpperLimitAttr"}
_PHYSICS_ATTR_NAMES = {"physics:lowerLimit", "physics:upperLimit"}
_LIMIT_TOKENS = _LIMIT_ATTR_NAMES | _PHYSICS_ATTR_NAMES


def _is_limit_attr_call(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr in _LIMIT_ATTR_NAMES:
        return True
    if func.attr == "GetAttribute" and len(call.args) == 1:
        arg = call.args[0]
        return isinstance(arg, ast.Constant) and arg.value in _PHYSICS_ATTR_NAMES
    return False


def _string_expr_value(node: ast.AST, constants: dict[str, str] | None = None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_expr_value(node.left, constants)
        right = _string_expr_value(node.right, constants)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                formatted = _string_expr_value(value.value, constants)
                if formatted is None:
                    return None
                parts.append(formatted)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.Call):
        method = node.func
        if isinstance(method, ast.Attribute):
            receiver = _string_expr_value(method.value, constants)
            if method.attr == "format" and receiver is not None:
                args = [_string_expr_value(arg, constants) for arg in node.args]
                if all(arg is not None for arg in args):
                    kwargs = {
                        keyword.arg: _string_expr_value(keyword.value, constants)
                        for keyword in node.keywords
                        if keyword.arg is not None
                    }
                    if len(kwargs) == len(node.keywords) and all(
                        value is not None for value in kwargs.values()
                    ):
                        return receiver.format(*args, **kwargs)
            if method.attr == "replace" and receiver is not None and len(node.args) == 2:
                old = _string_expr_value(node.args[0], constants)
                new = _string_expr_value(node.args[1], constants)
                if old is not None and new is not None:
                    return receiver.replace(old, new)
            if method.attr == "join" and receiver is not None and len(node.args) == 1:
                values = _string_sequence_value(node.args[0], constants)
                if values is not None:
                    return receiver.join(values)
    if isinstance(node, ast.Name) and constants is not None:
        return constants.get(node.id)
    return None


def _string_sequence_value(
    node: ast.AST,
    constants: dict[str, str] | None = None,
) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[str] = []
    for elt in node.elts:
        value = _string_expr_value(elt, constants)
        if value is None:
            return None
        values.append(value)
    return values


def _assign_string_constants(
    target: ast.AST,
    value: ast.AST,
    constants: dict[str, str],
) -> None:
    string_value = _string_expr_value(value, constants)
    if isinstance(target, ast.Name):
        if string_value is None:
            constants.pop(target.id, None)
        else:
            constants[target.id] = string_value
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts):
            for target_elt, value_elt in zip(target.elts, value.elts):
                _assign_string_constants(target_elt, value_elt, constants)
        else:
            for target_elt in target.elts:
                _clear_assigned_names(target_elt, constants)


def _clear_assigned_names(target: ast.AST, constants: dict[str, str]) -> None:
    if isinstance(target, ast.Name):
        constants.pop(target.id, None)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _clear_assigned_names(elt, constants)


def _token_for(call: ast.Call) -> str:
    func = call.func
    assert isinstance(func, ast.Attribute)
    if func.attr in _LIMIT_ATTR_NAMES:
        return func.attr
    arg = call.args[0]
    assert isinstance(arg, ast.Constant)
    return str(arg.value)


def audit_non_helper_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    violations: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.constants_stack: list[dict[str, str]] = [{}]

        @property
        def constants(self) -> dict[str, str]:
            return self.constants_stack[-1]

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.constants_stack.append(dict(self.constants))
            self.generic_visit(node)
            self.constants_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.constants_stack.append(dict(self.constants))
            self.generic_visit(node)
            self.constants_stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                _assign_string_constants(target, node.value, self.constants)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is None:
                _clear_assigned_names(node.target, self.constants)
            else:
                _assign_string_constants(node.target, node.value, self.constants)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if _is_limit_attr_call(node):
                violations.append(
                    f"{path}:{node.lineno}: {_token_for(node)} call in non-helper file"
                )
            if isinstance(node.func, ast.Name) and node.func.id in {"getattr", "hasattr"}:
                if len(node.args) >= 2:
                    token = _string_expr_value(node.args[1], self.constants)
                    if token in _LIMIT_TOKENS:
                        violations.append(
                            f"{path}:{node.lineno}: {token} dynamic access in non-helper file"
                        )
            self.generic_visit(node)

    Visitor().visit(tree)
    return violations


def audit_helper_file(path: Path, kind: str) -> list[str]:
    assert kind in {"reader", "writer"}
    tree = ast.parse(path.read_text())
    violations: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            outer = node.func
            if (
                isinstance(outer, ast.Attribute)
                and outer.attr in {"Get", "Set"}
                and isinstance(outer.value, ast.Call)
                and _is_limit_attr_call(outer.value)
            ):
                if kind == "reader" and outer.attr == "Set":
                    violations.append(f"{path}:{node.lineno}: reader cannot Set a USD limit")
                if kind == "writer" and outer.attr == "Get":
                    violations.append(f"{path}:{node.lineno}: writer cannot Get a USD limit")
            self.generic_visit(node)

    Visitor().visit(tree)
    return violations


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if path.name in {"usd_limit_reader.py", "usd_limit_writer.py", "audit_no_cheat.py"}:
            continue
        violations.extend(audit_non_helper_file(path))
    reader = root / "usd_limit_reader.py"
    writer = root / "usd_limit_writer.py"
    if reader.is_file():
        violations.extend(audit_helper_file(reader, kind="reader"))
    if writer.is_file():
        violations.extend(audit_helper_file(writer, kind="writer"))
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        raise SystemExit(1)
    print("[OK] audit_no_cheat AST scan passed")


if __name__ == "__main__":
    main()
