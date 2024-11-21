import logging
import numpy as np
from functools import partial
from itertools import product
from typing import Callable, List, Literal
from numpy.typing import NDArray
from scipy.integrate import solve_bvp
from scipy.optimize import OptimizeResult

from einops import einsum

from ars.utils import blerp, hlerp, slerp

from ars.log import setup_logging

setup_logging()

# Create a logger for the ephys.io module
logger = logging.getLogger(__name__)  # __name__ will be 'ephys.core'

Coord = NDArray

def schwarzschild_psi(
  r : NDArray,
  M : float = 1,
  G : float = 1,
  c : float = 1,
) -> NDArray:
  return np.sqrt(r / (r - 2 * G * M * c ** 2))

def schwarzschild_vol(
  rad : NDArray,
  tht : NDArray,
  phi : NDArray,
  M : float = 1,
  G : float = 1,
  c : float = 1,
) -> NDArray:
  return rad * rad * np.sin(tht) * np.sqrt(
    rad / (rad - 2 * G * M * c ** 2)
  )

def schwarzschild_metric(
  x : Coord,
  M : float = 1,
  G : float = 1,
  c : float = 1,
) -> NDArray:
  '''Compute the metric tensor for the Schwarzschild metric.
  '''
  r, tht, phi = x
  rs = 2 * G * M / c ** 2
  
  g_mn = np.zeros((3, 3, *r.shape))
  
  # Compute the metric tensor for the Schwarzschild metric
  g_mn[0, 0] = 1 / (1 - rs / r)
  g_mn[1, 1] = r ** 2
  g_mn[2, 2] = r ** 2 * np.sin(tht) ** 2
  
  if not np.isfinite(g_mn).all():
    raise ValueError('Invalid metric tensor')
  
  return g_mn

def schwarzschild_christoffel(
  x : Coord,
  M : float = 1,
  G : float = 1,
  c : float = 1,
) -> NDArray:
  '''Compute the Christoffel symbol for the Schwarzschild metric.
  '''
  r, tht, phi = x
  rs = 2 * G * M / c ** 2
  
  gamma = np.zeros((3, 3, 3, *r.shape))
  
  # Compute the Christoffel symbol for the Schwarzschild metric
  gamma[0, 0, 0] = -rs / (2 * r * (r - rs))
  gamma[0, 1, 1] = rs - r
  gamma[0, 2, 2] = (rs - r) * np.sin(tht) ** 2
  
  gamma[1, 0, 1] = gamma[2, 0, 2] = 1 / r
  gamma[1, 2, 2] = -np.cos(tht) * np.sin(tht)
  gamma[2, 1, 2] = 1 / (np.tan(tht) + 1e-10)
  
  if not np.isfinite(gamma).all():
    raise ValueError(f'Invalid Christoffel symbol. {r}, {tht}, {phi}')
  
  return gamma

METRIC = {
  'schwarzschild': {
    'metric'     : schwarzschild_metric,
    'christoffel': schwarzschild_christoffel,
  },
}

def geodesic(
  A : Coord,
  B : Coord,
  chris : Callable[[Coord], NDArray],
  guess : NDArray | None = None,
  f_ini : Callable[[Coord], NDArray] | None = None,
  f_fix : Callable[[NDArray], NDArray] | None = None,
  num_p : int = 50,
  **kwargs,
) -> OptimizeResult:
  '''
  Solve the geodesic equation for a given metric tensor
  between two points A and B. The geodesic equation is:
  
  x''^i + Γ^i_jk x'^j x'^k = 0
  
  where x is the coordinate parameter, Γ is the Christoffel symbol,
  and the derivatives are with respect to a curve-parameter l.
  
  We express the geodesic equation as a system of first-order ODEs
  and solve it using a numerical method (scipy.integrate.bvp).
  
  The system of ODEs is:
  { y1^i = x^i           ==>   { y1'^i = y2^i
  { y2^i = x'^i = dx/dl  ==>   { y2'^i = -Γ^i_jk y2^j y2^k
  '''
  A = np.asarray(A)
  B = np.asarray(B)
  
  f_fix = f_fix or (lambda *args: np.asarray(*args))
  
  def geo(_ : Coord, y : Coord):
    # Impose hard-constraints
    y = f_fix(y)
    
    y1, y2 = y[:3], y[3:]
    
    return np.stack([
      *y2,
      *einsum(
        -chris(y1),
        y2, y2,
        'i j k t, j t, k t -> i t',
      )
    ])
  
  def bc(ya : NDArray, yb : NDArray):
    return np.concatenate([
      ya[:3] - A,
      yb[:3] - B
    ])
  
  # This is the geodesic parametrization parameter
  # i.e. ɣ : [0, 1] -> [A, B]
  t = np.linspace(0, 1, num_p)

  # Set up a linear initial guess
  f_ini = f_ini or (lambda a, b, r: np.linspace(a, b, r).T)
  guess = guess or np.stack([
    *(line := f_ini(A, B, num_p)),
    *[np.gradient(l, t) for l in line]
  ])
  
  name = f_ini.__name__ if hasattr(f_ini, '__name__') else f'{f_ini}'
  logging.debug(f'Solver BVP Problem with {name} and {num_p}')
  return solve_bvp(
    geo, bc,
    t, guess,
    **kwargs,
  )

def distance(
  A : Coord,
  B : Coord,
  metric : Literal['schwarzschild'] = 'schwarzschild',
  init_fn : List[Callable[[Coord], NDArray]] | None = None,
  res : int = 100,
  **kwargs,
) -> float:
  '''Compute the distance between two points A and B
  along a geodesic path.
  '''
  A = np.asarray(A)
  B = np.asarray(B)
  
  metric_tens = METRIC[metric]['metric']
  christoffel = METRIC[metric]['christoffel']
  
  num_ps = kwargs.pop('num_p', [50, 25, 10, 5])
  init_fn = init_fn or [slerp, partial(slerp, dir=-1), hlerp, blerp, None]
  
  if not isinstance(init_fn, list): init_fn = [init_fn]
  if not isinstance(num_ps, list): num_ps = [num_ps]
  
  
  for f_ini, num_p in product(init_fn, num_ps):
    try:
      geo = geodesic(A, B, christoffel, f_ini=f_ini, num_p=num_p, **kwargs)
      if not geo.success: raise ValueError(f':(')
      break
    except ValueError as err:
      logging.info(f'Geodesic BVP Solver failed 🙁. Retrying...')
      continue
  
  if not geo.success:
    logging.exception(f'All Geodesic Solver failed. Stop trying.')
    raise ValueError(f'All initializations failed ☠️. Geodesic solver failed.\n{geo}')
  
  t = np.linspace(0, 1, res)
  Y = geo.sol(t)
  y, dydx = Y[:A.size], Y[A.size:]
  
  return np.trapz(
    np.sqrt(
      # g_ij dx^i dx^j
      einsum(
        metric_tens(y),
        dydx, dydx,
        'i j t, i t, j t -> t',
      )
    ),
    t,
  )