import numpy as np
from PDESimulation import *

# PDE Builder

def population_pde(birth, death, migration, econ):

    def population_pde_(u, dt):
        uc = u[1:-1, 1:-1]
        
        du = (birth(uc) - death(uc)) * uc - divergence(migration(u) * gradient(econ(u)))
        u[1:-1, 1:-1] = np.clip(uc + dt * du, 0, 1)
    
    return population_pde_

# Birth Functions

def constant_birth(rate):
    def constant_birth_(u):
        return rate
    return constant_birth_
    
# Death Functions

def constant_death(rate):
    def constant_death_(u):
        return rate
    return constant_death_
    
# Econ Functions

def population_econ(u, k=1):
    return k * u

def logistic_econ(u, k=15):
    return 1 / (1 + np.exp(-k * (u - 0.5)))
    
def guassian_econ(u, k=1, sigma=3):
    return k * np.exp(-np.square(u - 0.75) / sigma)
    
def poly_econ(u):
    return np.power(u, 7) - 7.3 * np.power(u, 6) + 17.11 * np.power(u, 5) - 13.315 * np.power(u, 4) + 1.18 * np.power(u,3) + 3.065 * np.square(u) - 1.38 * u + 0.18

# Resource Functions

def wrap_econ(econ, resource):
    def wrapped_econ(u):
        return econ(u) + resource
    return wrapped_econ

def normal_resource(size, scale=0.2):
    step = 2 / size
    x, y = np.mgrid[-1:1:step, -1:1:step]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    dist = multivariate_normal([0, 0], scale)
    return dist.pdf(pos)

# Migration Functions

def constant_migration(migration):
    def migration_(u):
        return migration
    return migration_
    
def inverse_migration(scale=1):
    def migration_(u):
        m = scale / (np.abs(u[1:-1, 1:-1]) + 0.1)
        return np.stack([m, m], axis=2)
    return migration_
    
    
if __name__ == "__main__":
    pde = population_pde(constant_birth(0.03), constant_death(0.025), constant_migration(0.01), wrap_econ(poly_econ, normal_resource(50)))
    simulate(pde, initial_perlin(50), neumann_bc, 3, 100000)
