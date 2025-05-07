import operator as op
from typing import Any, Callable, Optional, Sequence

import polars as pl
import pyparsing as pp

__all__ = ["query"]

from numpy.random.mtrand import Sequence

ops = {
    "**": op.pow,
    "~": op.not_,
    "*": op.mul,
    "@": op.matmul,
    "/": op.truediv,
    "//": op.floordiv,
    "%": op.mod,
    "+": op.add,
    "-": op.sub,
    "<<": op.lshift,
    ">>": op.rshift,
    "&": op.and_,
    "^": op.xor,
    "|": op.or_,
    "in": lambda a, b: a in b,
    "not in": lambda a, b: a not in b,
    "is": lambda a, b: a is b,
    "is not": lambda a, b: a is not b,
    "<": op.lt,
    "<=": op.le,
    ">": op.gt,
    ">=": op.ge,
    "!=": op.ne,
    "==": op.eq,
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
}


def parse_op(_a, _b, s: Sequence[Any]) -> list[Callable[[Any, Any], Any]]:
    return [ops[s[0]]]


def parse_chain(_a, _b, s: Sequence[Any]) -> list[Any]:
    return [s[1](s[0], s[2]) & s[3](s[2], s[4])]


def parse_col(_a, _b, s: Sequence[Any]) -> list[pl.Expr]:
    col = pl.col(s[0][len(s[0]) // 2])
    for t in s[1:]:
        if len(t) == 2:  # property
            col = getattr(col, t[1])
        elif len(t) == 5:  # function
            col = getattr(col, t[1])(t[3])
    return [col]


def parse_expr(_a, _b, s: Sequence[Any]) -> list[Any]:
    s = list(s)
    for operator in ops.values():
        while operator in s[1::2]:
            idx = 1 + 2 * s[1::2].index(operator)
            s = s[: idx - 1] + [operator(s[idx - 1], s[idx + 1])] + s[idx + 2 :]
    return s


def parse_brackets(_a, _b, s: Sequence[Any]) -> list[Any]:
    return [s[1]]


def parse_string(_a, _b, s: Sequence[Any]) -> list[str]:
    return [s[0][1:-1]]


def parse_list(_a, _b, s: Sequence[Any]) -> list[Sequence[Any]]:
    return [s[1:-1]]


def parse_not(_a, _b, s: Sequence[Any]) -> list[Any]:
    return [not s[1]] if len(s) == 2 else [s[0]]


def parse_const(_a, _b, s: Sequence[Any]) -> list[Optional[bool]]:
    return [{"None": None, "True": True, "False": False}[s[0]]]


Num = pp.common.number
String = pp.dbl_quoted_string ^ pp.sgl_quoted_string
String.set_parse_action(parse_string)
Var = pp.Combine("@" + pp.Word(pp.alphas + "_", pp.alphanums + "_"))
Const = pp.Literal("None") ^ "True" ^ "False"
Const.set_parse_action(parse_const)
Not = pp.Opt(pp.Literal("not"))
Id = Num ^ String ^ Var ^ Const
Id0 = Not + Id
Id0.set_parse_action(parse_not)
List = (
    "(" + pp.DelimitedList(Id0, allow_trailing_delim=True) + ")"
    ^ "[" + pp.DelimitedList(Id0, allow_trailing_delim=True) + "]"
)
List.set_parse_action(parse_list)
Col = pp.Group(
    pp.Word(pp.alphanums + "_")
    ^ pp.Literal("`")
    + pp.Word(pp.printables + " ", exclude_chars="`")
    + pp.Literal("`")
) + pp.ZeroOrMore(
    pp.Group(
        pp.Char(".")
        + pp.Opt(pp.Word(pp.alphas + "_", pp.alphanums + "_"))
        + pp.Opt(pp.Literal("(") + pp.Opt(Id0) + pp.Literal(")"))
    )
)
Col.set_parse_action(parse_col)
Id = Id ^ List | Col
Id1 = Not + Id
Id1.set_parse_action(parse_not)
CompOp = pp.Combine(pp.Char("<>") + pp.Opt("=")) ^ "==" ^ "!="
CompOp.set_parse_action(parse_op)
CalcOp = (
    pp.Char("+-/*%@&|^~")
    ^ "//"
    ^ "**"
    ^ "<<"
    ^ ">>"
    ^ "in"
    ^ "not in"
    ^ "is"
    ^ "is not"
)
CalcOp.set_parse_action(parse_op)
Op = CompOp ^ CalcOp
Expr = pp.Forward()
Brackets = pp.Literal("(") + Expr + pp.Literal(")")
Brackets.set_parse_action(parse_brackets)
Chain = Id1 + CompOp + Id1 + CompOp + Id1
Chain.set_parse_action(parse_chain)
Id = Brackets | Chain | Id
Id2 = Not + Id
Id2.set_parse_action(parse_not)
Expr << Id2 + pp.ZeroOrMore(Op + Id2)
Expr.set_parse_action(parse_expr)


def query(q: str, local_dict: dict[str, Any] = None) -> Any:
    """use a query string in polars.filter as can be done in pandas.query"""
    local_dict = local_dict or {}

    def parse_var(_a, _b, s: Sequence[Any]) -> list[Any]:  # type: ignore
        return [local_dict[s[0][1:]]]

    Var.set_parse_action(parse_var)
    return Expr.parse_string(q, parse_all=True)[0]
