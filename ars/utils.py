import logging
import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from ars.log import setup_logging

setup_logging()

# Create a logger for the ars.utils module
logger = logging.getLogger(__name__)  # __name__ will be 'ars.utils'


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

def fix_azimuth(
  p1 : Coord,
  p2 : Coord,
  res : int,
  dir : int = +1,
) -> NDArray:
  sign = -1 if dir * (p2 - p1) > dir * np.pi else +1
  phi = np.linspace(p1, p2 + sign * 2 * np.pi if abs(p2 - p1) > np.pi else p2, res)

  return phi

def slerp(
  A : NDArray | Tuple[float, float, float],
  B : NDArray | Tuple[float, float, float],
  res : int = 100,
  dir : int = +1,
) -> NDArray:
    """
    Linear interpolation between two points in spherical coordinates.

    Args:
        A (tuple): Starting point (r1, theta1, phi1).
        B (tuple): Ending point (r2, theta2, phi2).
        res (int): Number of points to interpolate.

    Returns:
        np.ndarray: Array of interpolated points (r, theta, phi).
    """
    A = np.asarray(A)
    B = np.asarray(B)
    
    r1, t1, p1 = A
    r2, t2, p2 = B
    
    # Linear interpolation in spherical coordinates
    r, t = np.linspace((r1, t1), (r2, t2), res).T
    
    # sign = -1 if p2 - p1 > np.pi else +1
    # p = np.linspace(p1, p2 + sign * 2 * np.pi if abs(p2 - p1) > np.pi else p2, res)

    # Handle cyclic interpolation for phi
    p = fix_azimuth(p1, p2, res, dir)

    return np.array([r, t, p])

def hlerp(
  A : NDArray | Tuple[float, float, float],
  B : NDArray | Tuple[float, float, float],
  res : int = 100,
  min_r : float = 4,
) -> NDArray:
  '''Hybrid interpolation between two points in spherical coordinates.
  The interpolation is first performed linearly in Cartesian coordinates
  and then converted back to spherical coordinates. Additionally, a
  minimum radial distance can be enforced and special care is put
  in the interpolation of the azimuthal angle phi.

  Args:
        A (tuple): Starting point (r1, theta1, phi1).
        B (tuple): Ending point (r2, theta2, phi2).
        res (int): Number of points to interpolate.

    Returns:
        np.ndarray: Array of interpolated points (r, theta, phi).
  '''
  cA, cB = pol2cart(*A), pol2cart(*B)

  l = np.linspace(cA, cB, res).T
  l = cart2pol(*l)
  
  # Enforce minimum radial distance
  r, t, p = l
  r = np.clip(r, min=min_r)
  
  return np.array([r, t, p])

def blerp(
  A : NDArray | Tuple[float, float, float],
  B : NDArray | Tuple[float, float, float],
  res : int = 100,
  min_r : float = 4,
) -> NDArray:
  p1 = slerp(A, B, res)
  p2 = hlerp(A, B, res, min_r)
  
  p1 = pol2cart(*p1)
  p2 = pol2cart(*p2)
  
  p = np.mean([p1, p2], axis=0)
  
  return cart2pol(*p)