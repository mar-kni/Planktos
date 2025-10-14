#first make sure you are in the mvbnd branch 
#goal 1: refresh on bernoff locusts work (what Koee was working on)
     #get Tyler up to speed on the locusts work
#goal 2: extend locuts model to include Brinkman flow


########Note right now this is brownian motion, I need to get locusts model from Koee
import planktos
import numpy as np

# Create an environment. 
# The default environment is a 10 meter by 10 meter 2D rectangle in which agents 
#   can exit out of any side (after which they cease to be simulated). The fluid 
#   velocity is zero everywhere until you set it to something else. See the 
#   docstring of the Environment class in planktos/environment.py to see all the 
#   options!
envir = planktos.Environment()

# Let's add some fluid velocity. First, we'll specify some (currently unrealistic)
#   properties of our fluid:
envir.rho = 1000 # fluid density
envir.mu = 1000 # dynamic viscosity

# Now let's specify some Brinkman flow. See the docstring for set_brinkman_flow 
#   in planktos/_environment.py for information on each of these parameters!
#alpha=0 is free flow
envir.set_brinkman_flow(alpha=0.0001, h_p=1.5, U=2, dpdx=1, res=101)

# Since this is a 2D fluid field, we can quickly visualize it to make sure 
#   everything looks right. Again, see the docstring for options!
envir.plot_flow()

# Now, let's add an agent Swarm! Since this is a simple example, we will go with
#   the default agent behavior: fluid advection with brownian motion, or "jitter".
#   By default, the Swarm class creates a Swarm with 100 agents initialized to
#   random positions throughout the environment. We just need to tell it what 
#   environment it should go in by passing in our Environment object.
#   Let's also specify a seed for the random number generator, so that our 
#   results will be reproducable.
swrm = planktos.Swarm(envir=envir, seed=1)

# Let's plot our new agents to see what we have so far:
swrm.plot()
# Also try: 
# swrm.plot(fluid='quiver')
# to see fluid arrows in the background! 
# There are many options that can be 
#   passed to the plot function as well as plot_all below. These include 
#   changing how big the agents appear (circ_rad). See the documentation for 
#   a complete explanation.

# By default, the Swarm is set up with two properties that are the same across 
#   all agents: 'mu' and 'cov'. These are the mean drift (not counting the fluid) 
#   and covariance matrix for the brownian motion. 'mu' defaults to a vector of 
#   zeros and 'cov' to an identity matrix. Since we are working in units of meters,
#   that's quite a lot of jitter! Let's slow those down by editing that property:
swrm.shared_props['cov'] = swrm.shared_props['cov'] * 0.01

# It is also possible to set properties which are different between agents. For
#   that, you edit the columns of swrm.props, which is a pandas DataFrame.

# Now it's time to simulate our swarm's movement! We do this by calling the 
#   "move" method of the Swarm object in a for loop. "move" takes one argument,
#   which is the temporal step size. For example, if we want to move the swarm
#   for 24 seconds with a step size of 0.1 seconds each time, we use:
for ii in range(240):
    swrm.move(0.1)

# We can see where we ended up with swrm.plot(), or we can plot a particular time 
#   by calling Swarm.plot(time), e.g. Swarm.plot(19.1). Or we can just plot 
#   everything as a movie:
swrm.plot_all(plot_heading=False, fluid='quiver') # turn off agent headings since this is Brownian
