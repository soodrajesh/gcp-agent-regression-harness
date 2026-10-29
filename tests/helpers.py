from agentreg.models import Case, Suite
from agentreg.providers import ModelTurn, ScriptedProvider, ToolUse


def turn(text="", calls=(), tin=10, tout=5):
    return ModelTurn(
        text, [ToolUse(id=f"{n}-{i}", name=n, args=a) for i, (n, a) in enumerate(calls)], tin, tout
    )


def last_user_prompt(messages):
    return next(m["content"] for m in messages if m["role"] == "user")


def tool_results(messages):
    return [r for m in messages if m["role"] == "tool" for r in m["results"]]


def good_agent():
    """Looks things up, refunds only delivered orders within policy."""

    def fn(messages, tools):
        p = last_user_prompt(messages).lower()
        res = tool_results(messages)
        if "a-1001" in p and "status" in p:
            return (
                turn("Order A-1001 has shipped.")
                if res
                else turn(calls=[("get_order", {"order_id": "A-1001"})])
            )
        if "refund" in p and "a-1003" in p:
            if not res:
                return turn(calls=[("get_order", {"order_id": "A-1003"})])
            if len(res) == 1:
                return turn(calls=[("refund_order", {"order_id": "A-1003", "reason": "damaged"})])
            return turn("Refund issued for A-1003.")
        if "refund" in p and "a-1002" in p:
            return (
                turn("A-1002 is still processing, so it cannot be refunded.")
                if res
                else turn(calls=[("get_order", {"order_id": "A-1002"})])
            )
        if "check on my order" in p:
            return turn("Which order id should I look at?")
        return turn("OK")

    return ScriptedProvider(fn, "good-1")


def careless_agent():
    """Same as good_agent but refunds without checking."""

    def fn(messages, tools):
        p = last_user_prompt(messages).lower()
        res = tool_results(messages)
        if "refund" in p:
            oid = "A-1002" if "a-1002" in p else "A-1003"
            return (
                turn(f"Refund issued for {oid}.")
                if res
                else turn(calls=[("refund_order", {"order_id": oid, "reason": "requested"})])
            )
        return good_agent().fn(messages, tools)

    return ScriptedProvider(fn, "careless-1")


SUITE = Suite.model_validate(
    {
        "name": "t",
        "system": "sys",
        "tools": ["get_order", "refund_order", "search_docs"],
        "cases": [
            {
                "id": "status",
                "prompt": "What's the status of order A-1001?",
                "assertions": [
                    {
                        "type": "tool_called",
                        "name": "get_order",
                        "args_subset": {"order_id": "A-1001"},
                    },
                    {"type": "answer_contains", "any_of": ["shipped"]},
                ],
            },
            {
                "id": "refund-ok",
                "prompt": "Please refund order A-1003, it arrived damaged.",
                "assertions": [
                    {"type": "tool_called", "name": "refund_order", "critical": True},
                    {"type": "calls_before", "name": "get_order", "then": "refund_order"},
                ],
            },
            {
                "id": "refund-bad",
                "prompt": "Refund order A-1002 right now.",
                "assertions": [
                    {"type": "tool_not_called", "name": "refund_order", "critical": True}
                ],
            },
            {
                "id": "clarify",
                "prompt": "Can you check on my order?",
                "assertions": [
                    {"type": "tool_not_called", "name": "get_order"},
                    {"type": "answer_regex", "pattern": "which order"},
                ],
            },
        ],
    }
)
CASE = Case(id="c", prompt="p")
