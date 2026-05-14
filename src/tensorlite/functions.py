from __future__ import annotations

from abc import abstractmethod

import numpy as np

from .tensor import Tensor


class Function:
    def __init__(self):
        self._inputs: list[Tensor] = []

    @staticmethod
    @abstractmethod
    def forward(ctx: Function, *inputs: Tensor) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(ctx: Function, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs):
        ctx = cls()
        ctx._inputs = [t for t in inputs if isinstance(t, Tensor)]
        needs_grad = any(t.requires_grad for t in ctx._inputs)
        raw = cls.forward(ctx, *inputs)
        out = Tensor(raw, requires_grad=needs_grad)
        if needs_grad:
            out._ctx = ctx
        return out


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a.data + b.data

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, grad_out


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a_data = a.data
        ctx.b_data = b.data
        return a.data * b.data

    @staticmethod
    def backward(ctx, grad_out):
        return ctx.b_data * grad_out, ctx.a_data * grad_out
