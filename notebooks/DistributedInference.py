import os
import jax

jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Build Porqueres et al. (2023) simulation setting
# https://arxiv.org/abs/2304.04785
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from scipy.stats import norm
from jaxpm.distributed import normal_field
from jax.experimental.multihost_utils import process_allgather , sync_global_devices
from functools import partial

all_gather = partial(process_allgather, tiled=True)

vlog = lambda msg : print(f"[Rank {rank}] {msg}")

print(jax.devices())

################################################

# define sharding and distribution

################################################

pdims = (8, 1)
devices = mesh_utils.create_device_mesh(pdims)
mesh = Mesh(devices.T, axis_names=('x', 'y'))
my_sharding = NamedSharding(mesh, P('x', 'y'))

z = np.linspace(0, 2.5, 1000)

nz_shear = [jc.redshift.kde_nz(z,norm.pdf(z, loc=z_center, scale=0.12) ,
                               bw=0.01, zmax=2.5, gals_per_arcmin2=g )
                for z_center, g in zip([0.5, 1., 1.5, 2.], [7,8.5, 7.5, 7])]
nbins = len(nz_shear)

################################################

# Define the fiducial cosmology

################################################

Omega_b = 0.049
Omega_c = 0.315 - Omega_b
sigma_8 = 0.8
h = 0.677
n_s = 0.9624
w0 = -1
cosmo = jc.Cosmology(Omega_c=Omega_c, sigma8=sigma_8, Omega_b=Omega_b,
                      h=h, n_s=n_s, w0=w0, Omega_k=0., wa=0.)

# Specify the size and resolution of the patch to simulate
field_size = 16.   # transverse size in degrees
field_npix = 64    # number of pixels per side
sigma_e = 0.3
vlog(f"Pixel size in arcmin: { field_size * 60 / field_npix}")

################################################

# Generate the data

################################################

from jax_lensing.dist_model import make_full_field_model

box_size  = [1000., 1000., 4500.]     # In Mpc/h
box_shape = [256,  256,  128]           # Number of voxels/particles per side
halo_size = 64
sharding = my_sharding
# Generate the forward model given these survey settings
lensing_model = jax.jit(make_full_field_model( 
                                            field_size=field_size,
                                            field_npix=field_npix,
                                            box_size=box_size,
                                            box_shape=box_shape,
                                            halo_size=halo_size,
                                            sharding=sharding))

def model(z,om,s8):
  Omega_c = om
  Omega_b = 0.049
  sigma8 = s8
  h = 0.677
  n_s = 0.9624
  w0 = -1
  cosmo = jc.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, sigma8=sigma8,
                       h=h, n_s=n_s, w0=w0, wa=0., Omega_k=0.)
                       
  convergence_maps, lc = lensing_model(cosmo, nz_shear, z)
  return convergence_maps, lc

z = normal_field(mesh_shape=box_shape,seed=jax.random.PRNGKey(0),sharding=sharding)
om = cosmo.Omega_c
s8 = cosmo.sigma8

vlog(f"z global shape {z.shape} local shape {z.addressable_data(0).shape}")

################################################

# Save Lightcone and Convergence Maps

################################################

kappa_obs , lc = model(z,om,s8)

lc_gathered = all_gather(lc)

if rank == 0:
    np.savez("lc.npz", lc=lc_gathered)

sync_global_devices('saving lc')


gathered_kappa_obs = [all_gather(kappa_obs[i]) for i in range(nbins)]

if rank == 0:
    kappa_dict = {f"kappa_{i}": gathered_kappa_obs[i] for i in range(nbins)}
    np.savez("kappa_obs.npz", **kappa_dict)
sync_global_devices('saving kappa_obs')


################################################

# Inference

################################################

from collections import namedtuple

ParamObj = namedtuple("ParamObj", ["z", "om", "s8"])

@jax.jit
def log_posterior(params):
    z, om, s8 = params

    convergence_maps , _ = model(z,om,s8)
    log_pz = -jnp.sum(z**2/2)
    log_likelihood = [(c - kappa_o)**2/(2 * sigma_e/jnp.sqrt(nz_shear[i].gals_per_arcmin2*(field_size*60/field_npix)**2)) \
        for i, (c, kappa_o) in enumerate(zip(convergence_maps, kappa_obs))]
    log_likelihood = -jnp.sum(sum(log_likelihood))
    return log_pz + log_likelihood


new_z = normal_field(mesh_shape=box_shape,seed=jax.random.PRNGKey(1),sharding=sharding)
om = cosmo.Omega_c + jax.random.normal(jax.random.PRNGKey(2), shape=(), dtype=jnp.float32)
s8 = cosmo.sigma8 + jax.random.normal(jax.random.PRNGKey(3), shape=(), dtype=jnp.float32)

om = cosmo.Omega_c
s8 = cosmo.sigma8
new_z = z

params = ParamObj(new_z, om, s8)


hlo = jax.jit(log_posterior).lower(params).compile().as_text()
hlo_grad = jax.jit(jax.grad(log_posterior)).lower(params).compile().as_text()

print("#"*80)
print("#"*80)
print("JIT - JIT" * 20)
print("#"*80)
print("#"*80)
print(hlo)
print("#"*80)
print("#"*80)
print("Grad - Grad " * 20)
print("#"*80)
print("#"*80)
print(hlo_grad)


grad = jax.grad(log_posterior)(params)


def minimize():
    # Set the ground truth for reference
    true_om = cosmo.Omega_c
    true_s8 = cosmo.sigma8

    # Initialize variables for gradient descent
    learning_rate = 0.01  # Tune this for appropriate convergence
    num_iterations = 2  # Number of gradient descent steps

    # Keep track of parameter evolution
    om_values = []
    s8_values = []

    for i in range(num_iterations):
        # Compute the gradients of the log posterior with respect to the parameters
        grad = jax.grad(log_posterior)(params)
        print(f"at iteration {i} grads om = {grad.om}, grads s8 = {grad.s8}")

        # Update the parameters using gradient descent
        z_new = params.z - learning_rate * grad.z
        om_new = params.om - learning_rate * grad.om
        s8_new = params.s8 - learning_rate * grad.s8
        print(f"at iteration {i}, om = {om_new}, s8 = {s8_new} grads om = {grad.om}, grads s8 = {grad.s8}")

        # Save the current values of om and s8
        om_values.append(om_new)
        s8_values.append(s8_new)

        # Update the parameters for the next iteration
        params = ParamObj(z_new, om_new, s8_new)

    # Plotting the evolution of om and s8
    plt.figure(figsize=[12, 6])

    # Plot for om
    plt.subplot(1, 2, 1)
    plt.plot(om_values, label="Estimated om")
    plt.axhline(y=true_om, color='r', linestyle='--', label="True om")
    plt.xlabel("Iteration")
    plt.ylabel("om")
    plt.legend()

    # Plot for s8
    plt.subplot(1, 2, 2)
    plt.plot(s8_values, label="Estimated s8")
    plt.axhline(y=true_s8, color='r', linestyle='--', label="True s8")
    plt.xlabel("Iteration")
    plt.ylabel("s8")
    plt.legend()

    plt.tight_layout()
    plt.show()

