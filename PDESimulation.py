import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from Utilities import generate_perlin_noise_2d

# Differential Operators

def gradient(u):
    delta = 2 / u[0].size

    u_right = u[1:-1, 2:]
    u_left = u[1:-1, 0:-2]
    u_top = u[0:-2, 1:-1]
    u_bottom = u[2:, 1:-1]
    
    dx = (u_right - u_left) / delta
    dy = (u_top - u_bottom) / delta
    
    return np.stack([dx, dy], axis=2)
    
def divergence(du):
    delta = 2 / du.shape[0]
    
    du_right = du[:, 1:, 0]
    du_left = du[:, 0:-1, 0]
    du_top = du[0:-1, :, 1]
    du_bottom = du[1:, :, 1]
    
    dx = (du_right - du_left)
    dy = (du_top - du_bottom)
    
    dx_ = np.zeros((du.shape[0], du.shape[1]))
    dy_ = np.zeros((du.shape[0], du.shape[1]))
    
    dx_[:, 0:-1] = dx
    dy_[0:-1, :] = dy
    
    return (dx_ + dy_) / delta
    
def laplacian(u):
    delta = 2 / u[0].size

    u_right = u[1:-1, 2:]
    u_left = u[1:-1, 0:-2]
    u_top = u[0:-2, 1:-1]
    u_bottom = u[2:, 1:-1]
    u_center = u[1:-1, 1:-1]
    
    return (u_top + u_left + u_bottom + u_right - 4 * u_center) / delta**2

# Simulation Calls

im = None

def display_distribution(u, ax):
    global im
    im = ax.imshow(100 * u, cmap=plt.cm.jet, interpolation='bilinear', extent=[-1, 1, -1, 1], vmin=0, vmax=100)
    ax.set_axis_off()
    
def simulate(pde, ic, bc, T, n=100000, plot_n=4):
    global im
    fig, axes = plt.subplots(plot_n, plot_n, figsize=(8, 8))
    step_plot = n // (plot_n * plot_n)
    
    dt = (T / n)
    u = ic.astype(np.float64)
    for i in range(n):
        pde(u, dt)
        bc(u)
        
        if i % step_plot == 0 and i < plot_n * plot_n * step_plot:
            ax = axes.flat[i // step_plot]
            display_distribution(u, ax=ax)
            ax.set_title(f"t={i * dt:.2f}")
            
    
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)

    cbar.set_ticks(np.arange(0, 101, 50))
    cbar.set_ticklabels(['Low', '', 'High'])
    
    plt.suptitle("Simulated Population Density Over Time")
    plt.show()
    return u
    
# Initial Conditions

def initial_zero(size):
    return np.zeros((size, size))
    
def initial_uniform(size):
    return np.full((size, size), 0.1)
    
def initial_random(size):
    return np.random.rand(size, size)
    
def initial_perlin(size):
    return np.abs(generate_perlin_noise_2d((size, size), (2, 2)))
    
def initial_normal(size, scale=1):
    step = 2 / size
    x, y = np.mgrid[-1:1:step, -1:1:step]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    dist = multivariate_normal([0, 0], scale)
    return dist.pdf(pos)

# Boundary Conditions

def neumann_bc(u):
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    
def dirichlet_bc(u):
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    
# PDEs

def heat(u, dt):
    k = 0.001
    u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * k * laplacian(u)
    
if __name__ == "__main__":
    simulate(heat, initial_random(50), neumann_bc, 3)
    
    
    

    