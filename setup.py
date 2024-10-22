from setuptools import setup

setup(
    name='jax_lensing',
    version='1.0',
    description='A small package for comparing full-field lensing constraints using JAX',
    author='Eiffl, Justinezgh',
    packages=['jax_lensing'],
    install_requires=[
        'jax-cosmo',
        'numpyro',
        'lenstools',
        'chainconsumer',
        'cmasher',
        'sbi_lens @ git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git'
    ]
)


