from __future__ import annotations

from typing import Callable


class Tensor:
    """A Tensor class for the operand of mathematical functions.
    Contains values and gradient ready operations."""

    def __init__(self, data: float | int, _children: tuple[Tensor, ...] = ()) -> None:
        self.data = data
        self.grad = 0
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Tensor] = set(_children)

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

        # ---------------------------------------------------
        #             Definition of Forward Pass
        # ---------------------------------------------------
        # Check if other is direct scalar
        if isinstance(other, (float, int)):
            other = Tensor(other)
        # If above check fails, check if other is Tensor
        elif not isinstance(other, Tensor):
            return NotImplemented
        # Define and return the output of the operation
        # out = self + other
        out = Tensor(self.data + other.data, _children=(self, other))

        # ---------------------------------------------------
        #             Definition of Backward Pass
        # ---------------------------------------------------
        def _backward():
            # ∂L/∂self  += (∂L/∂out)(∂out/∂self)  = out.grad * 1.0
            self.grad += 1.0 * out.grad
            # ∂L/∂other += (∂L/∂out)(∂out/∂other) = out.grad * 1.0
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

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

        # ---------------------------------------------------
        #             Definition of Forward Pass
        # ---------------------------------------------------
        # Check if other is direct scalar
        if isinstance(other, (float, int)):
            other = Tensor(other)
        # If the above check fails, check if other is Tensor
        elif not isinstance(other, Tensor):
            return NotImplemented
        # Define and return the output of the operation
        # out = self * other
        out = Tensor(self.data * other.data, _children=(self, other))

        # ---------------------------------------------------
        #             Definition of Backward Pass
        # ---------------------------------------------------
        def _backward():
            # ∂L/∂self  += (∂L/∂out)(∂out/∂self)  = out.grad * other.data
            self.grad += other.data * out.grad
            # ∂L/∂other += (∂L/∂out)(∂out/∂other) = out.grad * self.data
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

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
        #                Topological Sort
        # ---------------------------------------------------
        topo = []
        visited = set()

        def build_topo(T):
            if T not in visited:
                visited.add(T)
                for child in T._prev:
                    build_topo(child)
                topo.append(T)

        # Run the topological sort for the operation nodes
        build_topo(self)

        # Start the process of backpropagation
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
