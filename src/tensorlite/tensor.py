from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .functions import Function


class Tensor:
    """A Tensor class for the operand of mathematical functions.
    Contains values and gradient ready operations."""

    def __init__(self, data, requires_grad: bool = False) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self.requires_grad: bool = requires_grad
        self._ctx: Function | None = None

    def __repr__(self) -> str:
        return f"Tensor({self.data})"

    # ===================================================================================
    #                              Definition of Operations
    # ===================================================================================
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                          Add
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __add__(self, other: Tensor | float | int) -> Tensor:
        """Addition of Tensor to Tensor or Tensor to scalar."""
        from .functions import Add

        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Add.apply(self, other)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                      Right Add
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __radd__(self, other: Tensor | float | int) -> Tensor:
        return self.__add__(other)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                     Multiplication
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __mul__(self, other: Tensor | float | int) -> Tensor:
        """Multiplication of Tensor times Tensor or Tensor times scalar."""
        from .functions import Mul

        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Mul.apply(self, other)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                  Right Multiplication
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __rmul__(self, other: Tensor | float | int) -> Tensor:
        return self.__mul__(other)

    # ===================================================================================
    #                              Definition of Functionality
    # ===================================================================================

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                    Backpropagation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def backward(self):

        # ---------------------------------------------------
        #                 Topological Sort
        # ---------------------------------------------------
        topo = []
        visited = set()

        def build_topo(T):
            if T not in visited:
                visited.add(T)
                if T._ctx is not None:
                    for child in T._ctx._inputs:
                        build_topo(child)
                topo.append(T)

        # Run the topological sort for the operation nodes
        build_topo(self)

        # ---------------------------------------------------
        #                  Backpropagation
        # ---------------------------------------------------
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node._ctx is None:
                # the node is a leaf
                continue
            grads = type(node._ctx).backward(node._ctx, node.grad)
            for inp, grad in zip(node._ctx._inputs, grads):
                if inp.requires_grad:
                    if inp.grad is None:
                        inp.grad = np.zeros_like(inp.data)
                    inp.grad += grad
