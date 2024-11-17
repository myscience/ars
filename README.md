# Abstract Rewriting System in Pure Python 🐍

A basic package implementing a simple Abstract Rewriting System in pure Python using Structural Pattern Matching and a hint of meta-programming to make it work.

## Quick Start

Here is an example usage of the main class:

```python
from src.ars import ARS

# Define the initial abstract system as a collection of
# symbols, here we use simple characters
system = [('a', 'b'), ('c', 'd'), ('a', 'd'), ('c', 'b')]

# Define the abstract rewriting rule for the system
rule1 = '((a, b), (a, c), (d, b)) -> (c, a), (d, b)'
rule2 = '(a, b) -> (a, b), (a, b)'

# Build an Abstract Rewriting System
ars = ARS(rule1, rule2)

# Rewrite the system by applying the ARS rules
# (in definition order, consuming all possible matches)
ans = ars(system)

# Prints:
# [('d', 'a'), ('d', 'a'), ('c', 'b'), ('c', 'b'), ('c', 'd'), ('c', 'd')]
```
