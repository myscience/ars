from typing import Callable, Tuple
import numpy as np

from numpy.typing import NDArray

Domain = Tuple[Tuple[float, float], ...]
Coord = Tuple[NDArray, ...]
Field = NDArray

def poisson(
  metric : float | Callable[[Coord], Field],
  domain : Domain,
  max_metric : float = None,
  resolution : int = 100,
  size : int | None = 100,
  imap : Callable[[Coord], Coord] = None,
) -> NDArray:
  imap = imap or (lambda x: x)
  
  # * Compute the number of points to generate in the domain
  # * according to the poisson distribution
  # Fix a constant which is greater than the intensity
  # everywhere in the domain
  l_star = max_metric or np.max(
    M := metric(
      # Domain coordinates
      *np.meshgrid(*[
        np.linspace(a, b, resolution)
        for a, b in domain
      ])
    )
  ) * 1.1
  
  # Generate a random number of points
  # according to the poisson distribution
  vol = M.sum() * np.prod([(b - a) / resolution for a, b in domain])
  num = size or np.random.poisson(l_star * vol)
  
  # Generate the points coordinates uniformly in the domain
  P = np.stack([
    # Sample more points than needed so to
    # leave room for rejection sampling
    np.random.uniform(a, b, 5 * num)
    for a, b in domain
  ], axis=-1)
  
  # Compute the intensity at each point
  I : np.ndarray = metric(*P.T)
  
  # Implement rejection sampling by discarding points
  # according to the thinning strategy
  n, d = P.shape
  while n > size:
    mask = np.random.uniform(0, 1, size=n) < I / l_star
    idxs = tuple(idx[:size] for idx in np.nonzero(mask))
    P = P[idxs]
    n, d = P.shape
  
  return imap(*P.T)
