from typing import Callable, Literal
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_bvp
from scipy.optimize import OptimizeResult

from einops import einsum

Coord = NDArray

def cart2pol(*args):
  match len(args):
    case 2:
      x, y = args
      rho = np.sqrt(x**2 + y**2)
      phi = np.arctan2(y, x)
      return np.stack((rho, phi))
    case 3:
      x, y, z = args
      rho = np.sqrt(x**2 + y**2 + z**2)
      theta = np.arccos(z / rho)
      phi = np.arctan2(y, x)
      return np.stack((rho, theta, phi))
    case _:
      raise ValueError('Invalid number of arguments')

def pol2cart(*args):
  match len(args):
    case 2:
      rho, phi = args
      x = rho * np.cos(phi)
      y = rho * np.sin(phi)
      return np.stack((x, y))
    case 3:
      rho, theta, phi = args
      x = rho * np.sin(theta) * np.cos(phi)
      y = rho * np.sin(theta) * np.sin(phi)
      z = rho * np.cos(theta)
      return np.stack((x, y, z))
    case _:
      raise ValueError('Invalid number of arguments')

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
  
  assert np.isfinite(g_mn).all(), 'Invalid metric tensor'
  
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
  
  assert np.isfinite(gamma).all(), 'Invalid Christoffel symbol'
  
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
  christoffel : Callable[[Coord], NDArray],
  res : int = 100,
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
  
  def geo(_ : Coord, y : Coord):
    y1, y2 = y[:3], y[3:]
    
    return np.stack([
      *y2,
      *einsum(
        -christoffel(y1),
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
  t = np.linspace(0, 1, res)

  # Set up a linear initial guess
  guess = np.stack([
    *(line := np.linspace(A, B, res).T),
    *[np.gradient(l, t) for l in line]
  ])
  
  return solve_bvp(
    geo, bc,
    t, guess,
    **kwargs,
  )

def distance(
  A : Coord,
  B : Coord,
  metric : Literal['schwarzschild'] = 'schwarzschild',
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
  
  geo = geodesic(A, B, christoffel, res, **kwargs)
  
  if not geo.success:
    raise ValueError(f'Geodesic solver failed. {geo}')
  
  y, dydx = geo.y[:A.size], geo.y[A.size:]
  
  return np.trapz(
    np.sqrt(
      # g_ij dx^i dx^j
      einsum(
        metric_tens(y),
        dydx, dydx,
        'i j t, i t, j t -> t',
      )
    ),
    geo.x,
  )