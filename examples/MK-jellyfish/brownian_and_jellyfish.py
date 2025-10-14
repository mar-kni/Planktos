#first make sure you are in the mvbnd branch 
#goal 1: import and read in viz_IB2d vtk data 
#goal 2: simulating default brownian particles with initial positions around (not on top of) the jellyfish

#pip install pyvista 
#or alternatively 
#conda install -c conda-forge pyvista

import planktos
import numpy as np

envir= planktos.Environment()
#read in the ib2d vtk data
#dt= dt = 1.25e-5  
#print_dump = 1600 
envir.read_IB2d_fluid_data('examples/MK-jellyfish/viz_IB2d', 1.25e-5  , 1600 )

#read in the vertex data to get an immersed mesh
#need to manually adjust to avoid connecting wrong points in the bell of jellydfish
envir.read_IB2d_mesh_data('examples/MK-jellyfish/viz_IB2d', dt=1.25e-5  , print_dump=1600, brk_idx_list=[241])



#plot the environment
envir.plot_envir()
N=100
#set initial conditions, create Nx2 array of random initial positions
#initialize the Nx2 array of 
ICs= np.zeros((N,2))
#ICs[:(N//2),0]= np.random.uniform(0.01,0.5,N//2) #x positions
#ICs[(N//2):,0]= np.random.uniform(2.5,2.99,N//2) #x positions
ICs[:,0]= np.random.uniform(0.5,0.9,N) #x positions
ICs[:,1]= np.random.uniform(0.01,2.99,N) #y positions

swrm = planktos.Swarm(envir=envir, swarm_size=N, init= ICs, seed=1)

# Let's plot our new agents to see what we have so far:
swrm.plot()
# Also try: 
# swrm.plot(fluid='quiver')
# to see fluid arrows in the background! 


# amount of jitter (variance). Remember that 
#   variance is standard deviation squared
# By default, the Swarm is set up with two properties that are the same across 
#   all agents: 'mu' and 'cov'. These are the mean drift (not counting the fluid) 
#   and covariance matrix for the brownian motion. 'mu' defaults to a vector of 
#   zeros and 'cov' to an identity matrix. Since we are working in units of meters,
#   that's quite a lot of jitter! Let's slow those down by editing that property:
swrm.shared_props['cov'] = swrm.shared_props['cov'] * 0.01
swrm.shared_props['mu'] = [0.2,0.1]

#change to brownian motion 
# Now let's move the swarm with a time step of 0.025 sec, 50 times.
print('Moving swarm...')
for ii in range(50):
    swrm.move(0.1)

swrm.plot_all(movie_filename='examples/MK-jellyfish/jellyfish_ibmesh.mp4', fps=4, fluid='vort',
              plot_heading=False)