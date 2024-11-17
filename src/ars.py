from collections import defaultdict
from itertools import combinations
import re
from typing import Callable, Tuple


Term = Tuple[str, ...]

System = Tuple[Term, ...]

Program = Callable[[System], System]

class Rule(str):
  '''
  A string that represents a rule in an ARS system.
  '''
  def __new__(cls, s : str):
    if '->' not in s:
      raise ValueError('Rule must contain "->"')
    return super().__new__(cls, s)
  
  def __init__(self, obj) -> None:
    super().__init__()

    tail, head = map(str.strip, obj.split('->'))
    self.tail = tail
    self.head = head

class ARS:
  '''
  An ARS system.
  '''
  
  _rule_check: str = lambda *args, **kwargs: '''
  def check(parts):
    parts = parts[0] if len(parts) == 1 else parts
    match parts:
      case {new_t} if {guard}:
        return {head}
      case _:
        return None
  '''.format(**kwargs).strip()
  
  def __init__(self, *rules : Rule):
    self.rules = map(Rule, rules)
    
    self.programs = [self._compile(rule) for rule in self.rules]

  def __repr__(self):
    return f'ARS - Defined rules are:\n{'\n'.join(self.rules)}'
  
  def __call__(self, sys : System) -> System:
    for program in self.programs:
      sys = program(sys)
    return sys
  
  def _compile(self, rule : Rule) -> Program:    
    # Find element in head not in tail
    t_el = re.findall(r'[a-z]', rule.tail)
    h_el = re.findall(r'[a-z]', rule.head)
    
    diff = {x.strip() for x in set(h_el) if x not in set(t_el)}
    for el in diff: rule.head = rule.head.replace(el, f'"{el}"')
    
    # Find the number of patterns in the tail
    n_el = len(re.findall(r'\([a-z]\s*,\s*[a-z]\)', rule.tail))
    
    # Check whether the tail has any repeated elements
    # and if so replace them with a subpattern like x1, x2, ...
    # and add a guard to the rewrite function
    new_t = ''
    guard = []
    suffx = defaultdict(int)
    for char in rule.tail:
      if char.isalnum():
        suffx[char] += 1
        new_t += char + str(suffx[char])
      else:
        new_t += char
    for k, v in suffx.items():
      rule.head = rule.head.replace(k, f'{k}{v}')
      if v > 1: guard.extend([
        f'{k}{a} == {k}{b}'
        for a, b in combinations(range(1, v+1), 2)
      ])
    
    guard = ' and '.join(guard) or True
    
    scope = {
      'new_t': new_t,
      'guard': guard,
      'head': rule.head,
    }
    
    def rewrite(pattern):
      namespace = {}
      
      exec(self._rule_check(**scope), {}, namespace)
      P = len(pattern)
      S = {
        idxs : rule
        for idxs, parts in zip(
          combinations(range(P), n_el),
          combinations(pattern,  n_el),
        )
        if (rule := namespace['check'](parts))
      }
      
      valid_idxs = {i for i in range(P)}
      new_system = []
      off = 0
      for idxs, rule in S.items():
        # Check whether we can apply the rule
        if all(idx in valid_idxs for idx in idxs):
          valid_idxs -= set(idxs)
          new_system.extend(rule)
          
          # Remove the used elements from the pattern
          for idx in sorted(idxs, reverse=True): del pattern[idx - off]
          off += len(idxs)
      
      # Fill what's left of the pattern with the new system
      new_system.extend(pattern)
      
      return new_system
    
    return rewrite