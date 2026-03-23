"""Financial calculator with FinQA program support."""

from __future__ import annotations

import ast
import math
import re
from typing import Any


FINANCIAL_FORMULAS = {
    "pe_ratio": "Price / Earnings Per Share",
    "pb_ratio": "Price / Book Value Per Share",
    "roe": "Net Income / Shareholders' Equity",
    "roa": "Net Income / Total Assets",
    "current_ratio": "Current Assets / Current Liabilities",
    "debt_to_equity": "Total Debt / Total Equity",
    "gross_margin": "(Revenue - COGS) / Revenue",
    "net_margin": "Net Income / Revenue",
    "cagr": "((End Value / Start Value) ^ (1 / Years)) - 1",
    "yoy_growth": "(Current - Previous) / Previous * 100",
    "pct_change": "(New - Old) / Old * 100",
}

ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": pow,
}

ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        if char == delimiter and depth == 0:
            part = "".join(buffer).strip()
            if part:
                parts.append(part)
            buffer = []
            continue
        buffer.append(char)
    tail = "".join(buffer).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_number(value: str) -> float:
    cleaned = value.strip().replace(",", "").replace("%", "")
    if cleaned.startswith("const_"):
        return float(cleaned.split("_", 1)[1])
    return float(cleaned)


class SafeEval(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        fn_name = node.func.id
        if fn_name not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Function not allowed: {fn_name}")
        args = [self.visit(arg) for arg in node.args]
        return ALLOWED_FUNCTIONS[fn_name](*args)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in ALLOWED_CONSTANTS:
            raise ValueError(f"Name not allowed: {node.id}")
        return ALLOWED_CONSTANTS[node.id]

    def visit_Constant(self, node: ast.Constant) -> Any:
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed")
        return node.value

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _safe_eval(expression: str) -> Any:
    tree = ast.parse(expression, mode="eval")
    return SafeEval().visit(tree)


def _format_number(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if value == 0:
            return "0"
        if abs(value) >= 1_000_000:
            return f"{value:,.2f}"
        if abs(value) < 0.01:
            return f"{value:.6f}"
        return f"{value:.4f}"
    return str(value)


def execute_finqa_program(program: str) -> str:
    steps = _split_top_level(program)
    results: list[Any] = []

    for step in steps:
        match = re.fullmatch(r"(\w+)\((.*)\)", step)
        if not match:
            return f"Calculation error: Invalid FinQA step '{step}'"
        op = match.group(1).lower()
        args = []
        for token in _split_top_level(match.group(2)):
            token = token.strip()
            if token.startswith("#"):
                ref_idx = int(token[1:])
                if ref_idx >= len(results):
                    return f"Calculation error: Reference {token} not found"
                args.append(results[ref_idx])
            else:
                try:
                    args.append(_parse_number(token))
                except ValueError:
                    args.append(token)

        try:
            if op == "add":
                result = args[0] + args[1]
            elif op == "subtract":
                result = args[0] - args[1]
            elif op == "multiply":
                result = args[0] * args[1]
            elif op == "divide":
                result = args[0] / args[1]
            elif op == "greater":
                result = "yes" if args[0] > args[1] else "no"
            elif op == "exp":
                result = args[0] ** args[1]
            elif op == "table_sum":
                result = sum(float(item) for item in args)
            elif op == "table_average":
                numbers = [float(item) for item in args]
                result = sum(numbers) / len(numbers) if numbers else 0.0
            elif op == "table_max":
                result = max(float(item) for item in args)
            elif op == "table_min":
                result = min(float(item) for item in args)
            else:
                return f"Calculation error: Unsupported FinQA op '{op}'"
        except Exception as exc:
            return f"Calculation error in step '{step}': {exc}"

        results.append(result)

    if not results:
        return "Calculation error: No result computed"
    return f"Result: {_format_number(results[-1])}"


def execute_calculate(expression: str) -> str:
    expression = expression.strip()
    if not expression:
        return "Calculation error: empty expression"

    lowered = expression.lower()
    if any(op in lowered for op in ("add(", "subtract(", "multiply(", "divide(", "greater(", "exp(", "table_")):
        return execute_finqa_program(expression)

    pct_match = re.search(
        r"(?:pct_change|yoy_growth|growth|change)[:\s]*([\-0-9.,]+)\s*[,/→to]+\s*([\-0-9.,]+)",
        expression,
        re.IGNORECASE,
    )
    if pct_match:
        new_value = float(pct_match.group(1).replace(",", ""))
        old_value = float(pct_match.group(2).replace(",", ""))
        if old_value == 0:
            return "Calculation error: division by zero in percentage change"
        change = (new_value - old_value) / old_value * 100
        return f"Percentage change: ({new_value} - {old_value}) / {old_value} * 100 = {change:.2f}%"

    cagr_match = re.search(
        r"cagr[:\s]*([\-0-9.,]+)\s*[,/]\s*([\-0-9.,]+)\s*[,/]\s*(\d+)",
        expression,
        re.IGNORECASE,
    )
    if cagr_match:
        start_value = float(cagr_match.group(1).replace(",", ""))
        end_value = float(cagr_match.group(2).replace(",", ""))
        years = int(cagr_match.group(3))
        if start_value <= 0 or years <= 0:
            return "Calculation error: CAGR requires start_value > 0 and years > 0"
        cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
        return f"CAGR: ({end_value} / {start_value})^(1 / {years}) - 1 = {cagr:.2f}%"

    try:
        safe_expression = expression.replace("^", "**").replace(",", "")
        result = _safe_eval(safe_expression)
        return f"Result: {_format_number(result)}"
    except Exception as exc:
        return f"Calculation error: {exc}. Expression: {expression}"

