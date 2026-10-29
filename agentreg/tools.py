"""A small deterministic tool suite for the bundled example (an order-support agent). Each run gets a
fresh ToolBox, so side effects (refunds) are isolated and observable. Real deployments implement the
same interface over their own tools; the harness only needs `decls()` and `invoke()`."""

from __future__ import annotations

import ast
import operator
from typing import Any

ORDERS = {
    "A-1001": {
        "customer": "alice",
        "status": "shipped",
        "total": 120.0,
        "items": 2,
        "days_since_delivery": None,
    },
    "A-1002": {
        "customer": "alice",
        "status": "processing",
        "total": 80.0,
        "items": 1,
        "days_since_delivery": None,
    },
    "A-1003": {
        "customer": "bob",
        "status": "delivered",
        "total": 45.5,
        "items": 3,
        "days_since_delivery": 10,
    },
    "A-1004": {
        "customer": "bob",
        "status": "delivered",
        "total": 200.0,
        "items": 4,
        "days_since_delivery": 45,
    },
}
DOCS = {
    "returns": "Return policy: items can be returned within 30 days of delivery. Refunds go to the original payment method.",
    "refunds": "Refunds are only available for delivered orders within 30 days of delivery. Orders that are not yet delivered cannot be refunded.",
    "shipping": "Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days.",
}

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def safe_eval(expr: str) -> float:
    def ev(n: ast.AST) -> float:
        if (
            isinstance(n, ast.Constant)
            and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)
        ):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("only arithmetic is allowed")

    return ev(ast.parse(expr, mode="eval").body)


SPECS: dict[str, dict[str, Any]] = {
    "get_order": {
        "description": "Get an order by id: status, total, item count, customer and days since delivery.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    "list_orders": {
        "description": "List order ids for a customer name.",
        "parameters": {
            "type": "object",
            "properties": {"customer": {"type": "string"}},
            "required": ["customer"],
        },
    },
    "calculate": {
        "description": "Evaluate an arithmetic expression such as '120 * 0.15'.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    "search_docs": {
        "description": "Search company policy documents (returns, refunds, shipping).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    "refund_order": {
        "description": "Issue a refund for an order. Irreversible; only call when policy allows it.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["order_id", "reason"],
        },
    },
}


class ToolBox:
    def __init__(self, names: list[str]):
        self.names = names
        self.refunds: list[dict[str, str]] = []

    def decls(self) -> list[dict[str, Any]]:
        return [{"name": n, **SPECS[n]} for n in self.names]

    def invoke(self, name: str, args: dict[str, Any]) -> str:
        if name not in self.names:
            raise KeyError(f"unknown tool {name}")
        if name == "get_order":
            o = ORDERS.get(str(args.get("order_id", "")).upper())
            return (
                str({"order_id": str(args["order_id"]).upper(), **o})
                if o
                else "error: no such order"
            )
        if name == "list_orders":
            ids = [
                k
                for k, v in ORDERS.items()
                if v["customer"] == str(args.get("customer", "")).lower()
            ]
            return str(ids) if ids else "[]"
        if name == "calculate":
            try:
                return str(round(safe_eval(str(args.get("expression", ""))), 4))
            except (ValueError, SyntaxError, ZeroDivisionError) as e:
                return f"error: {e}"
        if name == "search_docs":
            q = str(args.get("query", "")).lower()
            hits = [
                t
                for k, t in DOCS.items()
                if k in q or any(w in t.lower() for w in q.split() if len(w) > 4)
            ]
            return "\n".join(hits[:2]) or "no results"
        if name == "refund_order":
            self.refunds.append(
                {"order_id": str(args.get("order_id")), "reason": str(args.get("reason"))}
            )
            return f"refund issued for {args.get('order_id')}"
        raise KeyError(name)


class ReplayToolBox:
    """Serves recorded results for recorded (tool, args) pairs; a call the recording never made is
    `novel` and goes to the live fallback toolbox if there is one (otherwise it errors)."""

    def __init__(self, recorded: list, fallback: ToolBox | None):
        self._rec = {(c.name, _key(c.args)): c.result for c in recorded}
        self.fallback = fallback
        self.novel: list[tuple[str, dict]] = []
        self.names = fallback.names if fallback else sorted({c.name for c in recorded})

    def decls(self):
        return (
            self.fallback.decls()
            if self.fallback
            else [
                {"name": n, "description": n, "parameters": {"type": "object"}} for n in self.names
            ]
        )

    def invoke(self, name: str, args: dict[str, Any]) -> str:
        hit = self._rec.get((name, _key(args)))
        if hit is not None:
            return hit
        self.novel.append((name, args))
        if self.fallback is None:
            return "error: no recorded result for this call"
        return self.fallback.invoke(name, args)


def _key(args: dict[str, Any]) -> str:
    import json

    return json.dumps(args, sort_keys=True, default=str)
