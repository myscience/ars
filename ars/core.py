import os
import re
import logging
from copy import copy
from uuid import uuid4
from random import choice
from itertools import combinations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

from ars.log import setup_logging

setup_logging()

# Create a logger for the ars.core module
logger = logging.getLogger(__name__)  # __name__ will be 'ars.core'

# Load the symbols from a file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
symbols_path = os.path.join(parent_dir, 'res', 'symbols.txt')

with open(symbols_path, 'r') as f:
  SYMBOLS = f.read().splitlines()

@dataclass
class Symbol:
  val: str = None
  _id: str = field(default_factory=lambda: str(uuid4()))
  
  def __hash__(self):
    return hash(self._id)
  
  def __repr__(self) -> str:
    return self.val
  
  def __eq__(self, other):
    if not isinstance(other, Symbol): return False
    return hash(self) == hash(other)

  def __post_init__(self):
    if self.val is None:
      self.val = choice(SYMBOLS)

Term = Tuple[Symbol, ...]

System = List[Term]

Program = Callable[[System], System]

def build_system(terms : List[Tuple[str, ...]]) -> System:
  symmap = {k : Symbol() for k in set([s for p in terms for s in p])}
  system = [tuple(symmap[symb] for symb in term) for term in terms]
  
  return system

class Rule(str):
  '''
  A string that represents a rule in an ARS system.
  '''
  def __new__(cls, s : str):
    if '->' not in s:
      raise ValueError('Rule must contain "->"')
    return super().__new__(cls, s)
  
  def __iter__(self):
    return self

  def __next__(self):
    # Generate a new unique symbol
    return Symbol()
  
  def __init__(self, obj : str) -> None:
    super().__init__()

    tail, head = map(str.strip, obj.split('->'))
    self._tail = tail
    self._head = head
    
    # Find element in head not in tail
    t_el = re.findall(r'[a-z]', tail)
    h_el = re.findall(r'[a-z]', head)
    diff = {x.strip() for x in set(h_el) if x not in set(t_el)}
    self._diff : dict[str] = diff
    
    # Check whether the tail has any repeated elements
    # and if so replace them with a sub-pattern like x1, x2, ...
    # and add a guard to the rewrite function
    self._suffix = defaultdict(int)
    for char in tail:
      if char.isalnum():
        self._suffix[char] += 1
    self._suffix = {k : v for k, v in self._suffix.items() if v > 1}
  
  @property
  def size(self) -> int:
    # Find the number of patterns in the tail
    return len(re.findall(r'\([a-z]\s*,\s*[a-z]\)', self._tail))

  @property
  def tail(self) -> str:
    tail = copy(self._tail)
    
    # Ensure that the tail has no repeated elements
    # NOTE: Pattern matching does not support it
    for k, v in self._suffix.items():
      for n in range(1, v + 1):  
        arr = tail.split(k)
        s = f'{k}'.join(arr[:n])
        e = f'{k}'.join(arr[n:])
        tail = s + f'{k}{n}' + e
    
    return tail
  
  @property
  def head(self) -> str:
    head = copy(self._head)
    
    # Align the head with the tail
    for k, v in self._suffix.items():
      head = head.replace(k, f'{k}{v}')
    
    # Add the new symbols to the head
    for v in self._diff:
      head = head.replace(v, f'"{v.upper()}"')
    
    return head
  
  @property
  def guard(self) -> str:
    guard = []
    for k, v in self._suffix.items():
      if v > 1: guard.extend([
        f'{k}{a} == {k}{b}'
        for a, b in combinations(range(1, v+1), 2)
      ])
    
    return ' and '.join(guard) or True

class ARS:
  '''
  An ARS system.
  '''
  
  def __init__(self, *rules : Rule):
    self.rules = tuple(map(Rule, rules))
  
  def __repr__(self):
    return f'ARS - Defined rules are:\n{'\n'.join(self.rules)}'
  
  def __call__(self, sys : System, **kwargs) -> System:
    for rule in self.rules:
      sys = self.rewrite(sys, rule, **kwargs)
    return sys
  
  @property
  def find_source(self) -> Callable[[Rule], str]:
    return lambda rule: f'''
    def run(parts):
      parts = parts[0] if len(parts) == 1 else parts
      match parts:
        case {rule.tail} if {rule.guard}:
          return {rule.head}
        case _:
          return None
    '''.strip()
  
  @property
  def cast_source(self) -> Callable[[Rule], str]:
    return lambda rule: f'''
    def run(sys : System) -> System:
      if not sys: return sys
      fill = {{k.upper() : Symbol() for k in {rule._diff}}}
      return tuple(
        tuple(fill.get(symb, symb) for symb in term)
        for term in sys
      )
    '''.strip()
  
  def rewrite(
    self,
    syst : System,
    rule : Rule,
    mode : Literal['node', 'mode']
  ) -> System:        
    def compile(
      source : Callable[[Rule], str],
      rule : Rule,
      scope : Dict = None,
    ) -> Program:
      namespace = {}
      
      exec(source(rule), scope or {}, namespace)
      
      return namespace['run']
    
    find = compile(self.find_source, rule)
    cast = compile(self.cast_source, rule, {'Symbol': Symbol, 'System': System})
    
    idxs = range(len(syst))
    connect = {term for term in syst}
    library = {symb for term in syst for symb in term}
    propose = tuple(
      (idxs, node, parts, terms)
      for idxs, parts in zip(
        combinations(idxs, rule.size),
        combinations(syst, rule.size),
      )
      if (terms := cast(find(parts)))
      and (node := {symb for term in terms for symb in term} & library)
    )
    
    off = 0
    system = []
    for idxs, nodes, edges, terms in propose:
      match mode:
        case 'node': used, total = set(nodes), library
        case 'edge': used, total = set(edges), connect
        case _: raise ValueError(f'Invalid mode {mode}')
      
      # Check whether we can apply the rule
      if used <= total:
        total -= used
        system.extend(terms)
        
        # Remove the used elements from the pattern
        for idx in sorted(idxs, reverse=True): del syst[idx - off]
        off += len(idxs)
    
    # Fill what's left of the pattern with the new system
    system.extend(syst)
    
    return system