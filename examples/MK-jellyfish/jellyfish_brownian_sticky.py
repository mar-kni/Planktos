#! /usr/bin/env python3


import numpy as np
import planktos
import numpy.ma as ma

# Let's begin by loading the same fluid and mesh as used in ex_ib2d_ibmesh.py
envir = planktos.Environment(ibmesh_color= 'magenta')
envir.read_IB2d_fluid_data('examples/MK-jellyfish/viz_IB2d', 1.25e-5  , 1600 )

#read in the vertex data to get an immersed mesh
#need to manually adjust to avoid connecting wrong points in the bell of jellydfish
envir.read_IB2d_mesh_data('examples/MK-jellyfish/viz_IB2d', dt=1.25e-5  , print_dump=1600, brk_idx_list=[241])



# In this example, we are going to create agents that stick to immersed
#   boundaries whenever they come in contact, and then stay there!

# To do this, we need to define some new agent behavior. It will rely on a 
#   user-defined agent property, which we will call 'stick'.
class permstick(planktos.Swarm):
    def apply_agent_model(self, dt):
        # first, we will get the value of the 'stick' property. It is expected
        #   to be different across agents, and to have boolean value. If it is
        #   False, we aren't sticking. If it is True, we will stay put!
        stick = self.get_prop('stick')

        # The most computationally efficient way to do this would be to only
        #   get movement vectors for the agents that are not stuck. But for
        #   brevity, we will get the movement for all of them and then adjust.
        
        all_move = planktos.motion.Euler_brownian_motion(self, dt)

        # Multiplying by a boolean array is like multiplying by 1s and 0s. So
        #   we just need to expand the dimension of the stick array (it's shape
        #   (N,) while all_move is shape (N,2)... this causes a broadcasting 
        #   error unless we expand stick to shape (N,1)), multiply and add!
        #   Putting a ~ in front of stick will negate the boolean values in
        #   the array.
        return np.expand_dims(~stick,1)*all_move +\
               np.expand_dims(stick,1)*self.positions
    
    # After an agent runs into an immersed structure, we want it to stop moving 
    #   for all future times. There is an attribute of the Swarm object called 
    #   ib_collision which is an array of bool, one for each agent. If the agent 
    #   collided with an immersed structure in the most recent move, it is set 
    #   to True for that agent. Otherwise, it is False. We'll use that to 
    #   dynamically update our 'stick' property after the move is over.
    # To do this, we will override after_move, a method that gets called 
    #   after all the agents have moved.
    def after_move(self, dt):
        swrm.props.loc[swrm.ib_collision, 'stick'] = True
        # Let's also color the agents that get stuck!
        #self.props.loc[self.ib_collision, 'color'] = 'red'
        print(f"Mask before update: {self.positions.mask}")
        #self.positions.mask= True #removes all particles 
        #want to remove only the particles that are stuck
        #self.positions[self.ib_collision].mask= True #is this correct?-Note: THis doesnt't work
        #see https://numpy.org/doc/stable/reference/maskedarray.generic.html#modifying-the-mask)
        #instead do the following:
        #import numpy.ma as ma
        self.positions[self.ib_collision] = ma.masked
        print(f"Mask after update: {self.positions.mask}")
        #record data for the particles that were eaten


# Now we create the Swarm similar to ex_ib2d_ibmesh.py.
# We will set store_prop_history=True because we want to keep track of agent 
#   property changes through time. We will also set the default ib condition to 
#   'sticky'. We could alternatively pass it in each time to the move method 
#   below.
        
N=50
#set initial conditions, create Nx2 array of random initial positions
#initialize the Nx2 array of 
ICs= np.zeros((N,2))
#ICs[:(N//2),0]= np.random.uniform(0.01,0.5,N//2) #x positions
#ICs[(N//2):,0]= np.random.uniform(2.5,2.99,N//2) #x positions
ICs[:,0]= np.random.uniform(0.5,0.9,N) #x positions
ICs[:,1]= np.random.uniform(0.01,2.99,N) #y positions

swrm = permstick(swarm_size=N, envir=envir, 
                 init=ICs, store_prop_history=True, 
                 ib_condition='sticky')
swrm.shared_props['cov'] *= 0.0001

# We also need to initialize our new agent properties. We'll set stick to False 
#   starting out, so that none of them will be stuck at the beginning
swrm.props['stick'] = np.full(N, False) # creates a length 100 array of False
# An equivalent way to do this: swrm.add_prop('stick', np.full(100, False), shared=False)

# Similarly, we need to initialize individual agent colors, since they won't all 
#   be the same now across all time steps. We'll use the default color for this 
#   initialization.
swrm.props['color'] = np.full(N, swrm.shared_props['color'])

swrm.shared_props['cov'] = swrm.shared_props['cov'] * 0.01
swrm.shared_props['mu'] = [0.2,0.1]

# Now we move the swarm. We'll use the 'sticky' option for immersed boundary
#   collisions instead of the default sliding option. This means that 
#   whenever an agent runs into an immersed structure, it will stop its movement 
#   for that time step at the point of intersection. It would be free to move in 
#   the next time step however, which is why our after_move updates a property
#   for us that is then used in apply_agent_model.

for ii in range(50):
    swrm.move(0.1, ib_collisions='sticky')
    # if np.any(swrm.ib_collision): # uncomment to display whenever something is getting stuck!
    #     swrm.plot()
    

swrm.plot_all(movie_filename='examples/MK-jellyfish/brownian_sticky_pink.mp4', fps=4, fluid='vort',
              plot_heading=False)

# Compare the result to that of ex_ib2d_ibmesh.py.

# Note that you can make use of the for-loop to update the Swarm object in all 
#   kinds of ways, or just to collect data about the Swarm dynamically. For 
#   instance, if you want to record every time that an agent encounters an 
#   immersed boundary, you could check swarm.ib_collision in the for-loop and 
#   then record the time and boolean data by appending to a list.