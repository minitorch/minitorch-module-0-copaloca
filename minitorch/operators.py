"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(a: float, b: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        a: first number.
        b: second number.

    Returns:
    -------
        the product of a and b.

    """
    return a * b


def id(a: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        a: number to be returned unchanged.

    Returns:
    -------
        the input number a.

    """
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
    ----
        a: first number.
        b: second number.

    Returns:
    -------
        the sum of a and b.

    """
    return a + b


def neg(a: float) -> float:
    """Negates a number.

    Args:
    ----
        a: number to be negated.

    Returns:
    -------
        the negated number.

    """
    return -a


def lt(a: float, b: float) -> bool:
    """Checks if one number is less than another.

    Args:
    ----
        a: first number.
        b: second number.

    Returns:
    -------
        True if a is less than b, False otherwise.

    """
    return a < b


def eq(a: float, b: float) -> bool:
    """Checks if two numbers are equal.

    Args:
    ----
        a: first number.
        b: second number.

    Returns:
    -------
        True if a is equal to b, False otherwise.

    """
    return a == b


def max(a: float, b: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        a: first number.
        b: second number.

    Returns:
    -------
        the larger of a and b.

    """
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Checks if two numbers are close in value.

    Args:
    ----
        a: first number.
        b: second number.

    Returns:
    -------
        True if a and b are close, False otherwise.

    """
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        a: number to be sigmoided.

    Returns:
    -------
        the sigmoid of a.

    """
    return 1.0 / (1.0 + math.exp(-a)) if a >= 0 else math.exp(a) / (1.0 + math.exp(a))


def relu(a: float) -> float:
    """Applies the ReLU activation function.

    Args:
    ----
        a: number to be ReLUed.

    Returns:
    -------
        the ReLU of a.

    """
    return a if a > 0 else 0


def log(a: float) -> float:
    """Calculates the natural logarithm.

    Args:
    ----
        a: number to be logarithmized.

    Returns:
    -------
        the natural logarithm of a.

    """
    return math.log(a)


def exp(a: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
        a: number to be exponentiated.

    Returns:
    -------
        the exponential of a.

    """
    return math.exp(a)


def inv(a: float) -> float:
    """Calculates the reciprocal.

    Args:
    ----
        a: number to be reciprocated.

    Returns:
    -------
        the reciprocal of a.

    """
    return 1 / a


def log_back(a: float, b: float) -> float:
    """Calculates the derivative of log times a second arg.

    Args:
    ----
        a: first number.
        b: second number.

    Returns:
    -------
        the derivative of the logarithm of a times b.

    """
    return b / a


def inv_back(a: float, b: float) -> float:
    """Calculates the derivative of reciprocal times a second arg.

    Args:
    ----
        a: first number.
        b: second number.

    Returns:
    -------
        the derivative of the reciprocal of a times b.

    """
    return -b / a**2


def relu_back(a: float, b: float) -> float:
    """Calculates the derivative of ReLU times a second arg.

    Args:
    ----
        a: first number.
        b: second number.

    Returns:
    -------
        the derivative of the ReLU of a times b.

    """
    return b * (1 if a > 0 else 0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
        f: function to be transformed.

    Returns:
    -------
        a new function that applies f to its input.

    """

    def apply(iter: Iterable[float]) -> Iterable[float]:
        return [f(x) for x in iter]

    return apply


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
    ----
        f: function to be applied to each pair of elements.

    Returns:
    -------
        a new function that applies f to each pair of elements from two iterables.

    """

    def apply(iter1: Iterable[float], iter2: Iterable[float]) -> Iterable[float]:
        return [f(x, y) for x, y in zip(iter1, iter2)]

    return apply


def reduce(
    f: Callable[[float, float], float], first: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function.

    Args:
    ----
        f: function to be applied to each pair of elements.
        first: first element to be used in the reduction.

    Returns:
    -------
        a new function that reduces an iterable to a single value using f.

    """

    def apply(iter: Iterable[float]) -> float:
        n = first
        for x in iter:
            n = f(n, x)
        return n

    return apply


def negList(iter: Iterable) -> Iterable:
    """Negates all elements in a list using map.

    Args:
    ----
        iter: iterable to be negated.

    Returns:
    -------
        a list of the negated elements of iter.

    """
    return map(neg)(iter)


def addLists(iter1: Iterable, iter2: Iterable) -> Iterable:
    """Adds corresponding elements from two lists using zipWith.

    Args:
    ----
        iter1: first iterable.
        iter2: second iterable.

    Returns:
    -------
        a list of the sums of corresponding elements from iter1 and iter2.

    """
    return zipWith(add)(iter1, iter2)


def sum(iter: Iterable) -> float:
    """Sums all elements in a list using reduce.

    Args:
    ----
        iter: iterable to be summed.

    Returns:
    -------
        the sum of all elements in iter.

    """
    return reduce(add, 0)(iter)


def prod(iter: Iterable) -> float:
    """Calculates the product of all elements in a list using reduce.

    Args:
    ----
        iter: iterable to be multiplied.

    Returns:
    -------
        the product of all elements in iter.

    """
    return reduce(mul, 1)(iter)
