import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpy as np
import jax_cosmo as jc
from jax_cosmo.scipy.integrate import simps
import jax_cosmo.constants as constants

import numpyro
import numpyro.distributions as dist

from jaxpm.pm import pm_forces, growth_factor
from jaxpm.kernels import fftk
from jaxpm.painting import cic_paint, compensate_cic

import astropy.units as u

def linear_field(mesh_shape, box_size, pk, field):
  """
    Generate initial conditions.
  """
  kvec = fftk(mesh_shape)
  kmesh = sum(
      (kk / box_size[i] * mesh_shape[i])**2 for i, kk in enumerate(kvec))**0.5
  pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
      box_size[0] * box_size[1] * box_size[2])

  field = jnp.fft.rfftn(field) * pkmesh**0.5
  field = jnp.fft.irfftn(field)
  return field

def lpt_lightcone(cosmo, initial_conditions, positions, a, mesh_shape):
    """
    Computes first order LPT displacement
    """
    initial_force = pm_forces(positions, delta=initial_conditions).reshape(mesh_shape+[3])
    a = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a).reshape([1,1,-1,1]) * initial_force
    return dx.reshape([-1,3])

def convergence_Born(cosmo,
                     density_planes,
                     r,
                     a,
                     dx,
                     dz,
                     coords,
                     z_source):
  """
  Compute the Born convergence
  Args:
    cosmo: `Cosmology`, cosmology object.
    density_planes: list of dictionaries (r, a, density_plane, dx, dz), lens planes to use
    coords: a 3-D array of angular coordinates in radians of N points with shape [batch, N, 2].
    z_source: 1-D `Tensor` of source redshifts with shape [Nz] .
    name: `string`, name of the operation.
  Returns:
    `Tensor` of shape [batch_size, N, Nz], of convergence values.
  """
  # Compute constant prefactor:
  constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
  # Compute comoving distance of source galaxies
  r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

  convergence = 0
  n_planes = len(r)

  def scan_fn(carry, i):
    density_planes, a, r = carry

    p = density_planes[:,:,i]
    density_normalization = dz * r[i] / a[i]
    p = (p - p.mean()) * constant_factor * density_normalization

    # Interpolate at the density plane coordinates
    im = map_coordinates(p,
                         coords * r[i] / dx - 0.5,
                         order=1, mode="wrap")

    return carry, im * jnp.clip(1. - (r[i] / r_s), 0, 1000).reshape([-1, 1, 1])

  _, convergence = jax.lax.scan(scan_fn, (density_planes, a, r), jnp.arange(n_planes))

  return convergence.sum(axis=0)

def make_full_field_model(field_size, field_npix, 
                          box_shape, box_size):

  def forward_model(cosmo, nz_shear, initial_conditions):
    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    # Create initial conditions
    lin_field = linear_field(box_shape, box_size, pk_fn, initial_conditions)

    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in box_shape]),axis=-1).reshape([-1,3])

    # Compute the scale factor that corresponds to each slice of the volume
    r = (jnp.arange(box_shape[-1]) + 0.5)*box_size[-1]/box_shape[-1]
    a = jc.background.a_of_chi(cosmo, r)

    cosmo = jc.Cosmology(Omega_c=cosmo.Omega_c, sigma8=cosmo.sigma8, Omega_b=cosmo.Omega_b,
                        h=cosmo.h, n_s=cosmo.n_s, w0=cosmo.w0, Omega_k=0., wa=0.)
    
    # Initial displacement
    dx = lpt_lightcone(cosmo, lin_field, particles, a, box_shape)

    # Paint the particles on a new mesh
    lightcone = cic_paint(jnp.zeros(box_shape),  particles+dx)
    # Apply de-cic filter to recover more signal on small scales
    lightcone = compensate_cic(lightcone)

    dx = box_size[0] / box_shape[0]
    dz = box_size[-1] / box_shape[-1]

    # Defining the coordinate grid for lensing map
    xgrid, ygrid = np.meshgrid(np.linspace(0, field_size, box_shape[0], endpoint=False), # range of X coordinates
                                np.linspace(0, field_size, box_shape[1], endpoint=False)) # range of Y coordinates
    coords = jnp.array((np.stack([xgrid, ygrid], axis=0)*u.deg).to(u.rad))

    # Generate convergence maps by integrating over nz and source planes
    convergence_maps = [simps(lambda z: nz(z).reshape([-1,1,1]) *
                                convergence_Born(cosmo, lightcone, r,a, dx, dz, coords, z), 0.01, 3.0, N=32)
                        for nz in nz_shear]

    # Reshape the maps to desired resoluton
    convergence_maps = [kmap.reshape([field_npix, box_shape[0] // field_npix,  field_npix, box_shape[1] // field_npix ]).mean(axis=1).mean(axis=-1)
                        for kmap in convergence_maps]

    return convergence_maps, lightcone

  return forward_model


# Build the probabilistic model
def full_field_probmodel(config):
  forward_model = make_full_field_model(config.field_size, config.field_npix,
                                        config.box_shape, config.box_size)
  
  # Sampling the cosmological parameters
  cosmo = config.fiducial_cosmology(**{k: numpyro.sample(k, v) for k, v in config.priors.items()})

  # Sampling the initial conditions
  initial_conditions = numpyro.sample('initial_conditions', dist.Normal(jnp.zeros(config.box_shape),
                                                                        jnp.ones(config.box_shape)))

  # Apply the forward model
  convergence_maps, _ = forward_model(cosmo, config.nz_shear, initial_conditions)

  # Define the likelihood of observations
  observed_maps = [numpyro.sample('kappa_%d'%i,
                                  dist.Normal(k, 
                                              config.sigma_e/jnp.sqrt(config.nz_shear[i].gals_per_arcmin2*
                                                                      (config.field_size*60/config.field_npix)**2)))
                   for i,k in enumerate(convergence_maps)]

  return observed_maps


def pixel_window_function(l, pixel_size_arcmin):
  """
  Calculate the pixel window function W_l for a given angular wave number l and pixel size.

  Parameters:
  - l: Angular wave number (can be a numpy array or a single value).
  - pixel_size_arcmin: Pixel size in arcminutes.

  Returns:
  - W_l: Pixel window function for the given l and pixel size.
  """
  # Convert pixel size from arcminutes to radians
  pixel_size_rad = pixel_size_arcmin * (np.pi / (180.0 * 60.0))

  # Calculate the Fourier transform of the square pixel (sinc function)
  # Note: l should be the magnitude of the angular wave number vector, |l| = sqrt(lx^2 + ly^2) for a general l
  # For simplicity, we assume l is already provided as |l|
  W_l = (jnp.sinc(l * pixel_size_rad / (2 * np.pi)))**2

  return W_l


def make_2pt_model(pixel_scale, ell, sigma_e=0.3):
  """
  Create a function that computes the theoretical 2-point correlation function for a given cosmology and redshift distribution.

  Parameters:
  - pixel_scale: Pixel scale in arcminutes.
  - ell: Angular wave number (numpy array).

  Returns:
  - forward_model: Function that computes the theoretical 2-point correlation function for a given cosmology and redshift distribution.
  """
  
  def forward_model(cosmo, nz_shear):      
    tracer = jc.probes.WeakLensing(nz_shear, sigma_e=sigma_e)
    cell_theory = jc.angular_cl.angular_cl(cosmo, ell, [tracer], nonlinear_fn=jc.power.linear)
    cell_theory = cell_theory * pixel_window_function(ell, pixel_scale)
    cell_noise = jc.angular_cl.noise_cl(ell,[tracer])
    return cell_theory, cell_noise

  return forward_model
