#import packages 
import numpy as np
import planktos
from numpy import random as rnd
import pandas as pd
from scipy.integrate import solve_ivp
import math 

#packages for color cycling 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cycler import cycler

import pdb

#set domain to be square with periodic boundary conditions
envir = planktos.environment(Lx=1, Ly=1, x_bndry='periodic', y_bndry='periodic')

#define fluid flow field 
#start with no flow and then change to a simple flow scenario later
num_xpts= 11
num_ypts= 11
envir.flow_points= [np.linspace(0,1,num_xpts), np.linspace(0,1,num_ypts)] 
X_flow= np.full((num_xpts, num_ypts), 0)
Y_flow= np.full((num_xpts, num_ypts), 0)
envir.flow = [X_flow, Y_flow] #flow is a list of ndarrays

#define the swarm size
#we'll start with something small to test the code
SWARM_SIZE = 10 #N=2


class three_zone_torque(planktos.swarm):
    def __init__(self, *args, **kwargs):
        super(three_zone_torque, self).__init__(*args, **kwargs)
        ##### Parameter choices #####
        #number of time steps 
        #self.shared_props['T'] = 300 #300 
        #swarm size is the same for all particles
        self.shared_props['N']= SWARM_SIZE

        # particles repel away from particles within a radius r_r
        self.shared_props['r_r'] = 0.05 #.05
        
        # particles align/orient to other particles within a radius r_o
        #r_o> r_r
        self.shared_props['r_o'] = 0.1  # .2

        # particles are attracted to other particles within a radius r_a
        #r_a> r_o
        self.shared_props['r_a'] = 0.9 #0.8

        #parms which can values change for each particle
        # set particle speed (abs of velocity) in absence of fluid here
        #self.props['v'] = rnd.uniform(0.1, .2, self.N)  #changed a from 0.1 to 0.01 and b from 1 to 0.2
        #self.props['v']= np.array([0.16, 0.13, 0.12, 0.3]) #for N=4
        self.props['v']= np.array([0.16, 0.16, 0.16, 0.16,0.16,0.16,0.16,0.16,0.16,0.16]) #for N=4
        print(f'self.props[v]: {self.props['v']}')

        # angle noise will be between [-nu/2, nu/2]
        self.props['nu'] = rnd.uniform(0.1, 1,self.N) 

        ##############for N=5 
        #set angles for each particle 
        #what is the range of angles?
        self.props['angle'] = np.array([0, (1/2)*np.pi, (3/2)*np.pi, 1*np.pi, (3/2)*np.pi,(1/2)*np.pi,1*np.pi,1.5*np.pi,0,(1/4)*np.pi]) #for N=3
        print(f'self.angle: {self.props['angle']}')

        ##########params for ODEs
        self.shared_props['K'] = 2
        self.shared_props['J'] = 0   

        #######set radius of attraction, repulsion, alignment 
        self.props['r_r'] = .2
        self.props['r_o'] = .5
        self.props['r_a'] = 1.0
    


        #set initial velocities 
        #use speed above with a direction in theta 
        #needs to be Nx2 
        self.velocities= (self.get_prop('v')* np.array([np.cos(self.get_prop('angle')), np.sin(self.get_prop('angle'))])).T
        
        print(f' positions: {self.positions}')

        #set colors for each particle-using color cycler 
        #https://matplotlib.org/stable/users/explain/colors/colormaps.html

        cmap= plt.cm.viridis #plasma #choose color map 

        num_colors= self.N #number of colors needed= # of particles

        # Create a ListedColormap with the desired number of colors
        discrete_cmap = ListedColormap(cmap(np.linspace(0, 1, num_colors)))
        
        #set colors for each particle
        self.props['color'] = [discrete_cmap(i / (num_colors - 1)) for i in range(num_colors)] #set color for each particle

        # Create a color cycle from the discrete colormap
        color_cycle = cycler('color', self.get_prop('color'))

        # Set the color cycle as the default property cycle
        plt.rc('axes', prop_cycle=color_cycle)

        plt.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg" #added to force ffmpeg path 
    
    def __calc_dist(self, origin, positions_array):
        ''' A private method that calculates the distance between an origin 
        position and all the postions in postions_array, respecting periodic
        boundary conditions when applicable.
            
            Parameters
            ----------
            origin : ndarray
                length d array where d is the dimension of the environment.
            positions_array : np array
                N by d array where N is the number of agents and d is the 
                    dimension of the environment.
            domain : ndarray
                length d array that specifies the dimensions of the environment.
        '''

        diffs = np.zeros(positions_array.shape)

        # get boundary condition type
        x_bndry = self.envir.bndry[0][0]
        y_bndry = self.envir.bndry[1][0]
    

        # get length of each spatial dimension
        domain = self.envir.L

        x_delta = positions_array[:,0] - origin[0]
        y_delta = positions_array[:,1] - origin[1]
        #print(f'origin: {origin}, positions_array: {positions_array}')
        #print(f'x_delta: {x_delta}, y_delta: {y_delta}')
        
        if x_bndry == 'periodic':
            diffs[:,0] = (x_delta + domain[0]/2) % domain[0] - domain[0]/2
        else:
            diffs[:,0] = x_delta

        if y_bndry == 'periodic':
            diffs[:,1] = (y_delta + domain[1]/2) % domain[1] - domain[1]/2
        else:
            diffs[:,1] = y_delta

        dist = np.sqrt(diffs[:,0]**2 + diffs[:,1]**2)
            
        return [dist, diffs]
    
    def ODEs(self, t, states):
        ''' A method that returns the ODEs for the swarm.
        
            Parameters
            ----------
            t : float
                The current time.
            states : ndarray
                A 1 x 3N array of the current state of the swarm, where each 
                particle's state is represented by its angle and positions (x and y).
        '''
        #debugging
        #import pdb; pdb.set_trace()

        #unpack the states
        angles = states[:self.N]
        x_pos = states[self.N:2*self.N]
        y_pos = states[2*self.N:]
        positions = np.stack((x_pos, y_pos), axis=1)
        
        #initialize the derivatives
        d_angle = np.zeros(self.N)
        d_x_pos = np.zeros(self.N)
        d_y_pos = np.zeros(self.N)
        #initialize positions as a 2D array
        d_pos = np.stack((d_x_pos, d_y_pos), axis=1)

        #calculate derivatives for each particle
        #vector of angle deriviative 
        #initialize summations as a vector of length number of agents in the attraction zone 
        N_attract= self.N
        N_repel=self.N
        N_align=self.N
        
        dist_all = np.zeros((self.N, self.N)) #distance from each particle to all other particles
        
        ####initialize things for attraction, repulsion, alignment
        sum_attract= np.zeros(N_attract) #initialize the sum of the attraction angle terms
        sum_repel= np.zeros(N_repel) #initialize the sum of the attraction angle terms
        fraction_align= np.zeros(2)
        sum_align= np.zeros(N_align) #initialize the sum of the attraction angle terms
        
        ############
        
        for n in range(self.N):
            
            result= self.__calc_dist(positions[n, :], positions) #distance from particle n to all other particles
            dist_n = result[0] #distances from particle n to all other particles
            diffs_n = result[1] #differences from particle n to all other particles
            
            dist_all[n,:] = dist_n #store the distances in a matrix for later use
            #find the indices of particles within each zone
            indices_attract=[]
            indices_repel=[]
            indices_align=[]
            indices_attract = np.where((dist_n > self.get_prop('r_o')) & (dist_n <= self.get_prop('r_a')))[0]
            indices_repel = np.where(dist_n <= self.get_prop('r_r'))[0]
            indices_align = np.where((dist_n > self.get_prop('r_r')) & (dist_n <= self.get_prop('r_o')))[0]
            
            x_comp_n_attract= []
            y_comp_n_attract= []

            x_comp_n_repel= []
            y_comp_n_repel= []

            fractions_n_align=[]
            
            for j in range(self.N):
                if j != n:  # don't include self
                   
                   #attraction
                    if (len(indices_attract) >= 1) and (j in indices_attract):

                        # Unit vector from n to j
                        fraction_attract = (diffs_n[j]) / (dist_n[j] + 0.0000001)

                        # Store x and y components directly
                        x_comp_n_attract.append(fraction_attract[0])
                        y_comp_n_attract.append(fraction_attract[1])
                    
                    #repulsion
                    if (len(indices_repel) >= 1) and (j in indices_repel):
                        # Unit vector from n to j
                        fraction_repel = (diffs_n[j]) / (dist_n[j] + 0.0000001)**2

                        # Store x and y components directly
                        x_comp_n_repel.append(fraction_repel[0])
                        y_comp_n_repel.append(fraction_repel[1])
                    
                    #alignment
                    if (len(indices_align) >= 1) and (j in indices_align):
                        # Unit vector from n to j
                        fraction_align = np.sin(self.get_prop('angle')[j]- self.get_prop('angle')[n])
                        fractions_n_align.append(fraction_align)

                    #add a case for if all zones are empty (ie all particles are outside the attraction zone)   
            
            # Average direction angle
            if (len(indices_attract) >= 1):
                print(indices_attract)
                print(len(y_comp_n_attract))
                print(len(x_comp_n_attract))
                sum_attract[n] = np.arctan2(np.sum(y_comp_n_attract)/len(indices_attract), np.sum(x_comp_n_attract)/len(indices_attract))
            else:
                sum_attract[n] = 0


            # Average direction angle
            if (len(indices_repel) >= 1):
                print(indices_repel)
                print(len(y_comp_n_repel))
                sum_repel[n] = np.arctan2(np.sum(y_comp_n_repel)/len(indices_repel), np.sum(x_comp_n_repel)/len(indices_repel))
            else:
                sum_repel[n] = 0
            
            # Average direction angle
            if (len(indices_align) >= 1):
                sum_align[n] = np.sum(fractions_n_align) 
            else:
                sum_align[n] = 0
        
        ###############################
        d_angle= self.get_prop('K')* (np.sin(sum_attract-self.get_prop('angle'))  - (3/4)*np.sin(sum_repel-self.get_prop('angle'))  + (1/(N_align)) * sum_align)
    
        #vector of positions derivative 
        #direction = np.array([np.cos(angles), np.sin(angles)]).T
        #direction_unit_vector = direction / np.linalg.norm(direction, axis=1, keepdims=True)
        speeds = self.get_prop('v')                  # shape (N,)
        #directions = direction_unit_vector           # shape (N,2)
        #d_pos = speeds.reshape(-1, 1) * directions   # shape (N,2)
        d_pos= speeds.reshape(-1,1) * np.array([np.cos(angles), np.sin(angles)]).T
        #print(f'direction: {direction}, direction_unit_vector: {direction_unit_vector}, d_pos: {d_pos}')
        d_x_pos = d_pos[:,0]
        d_y_pos = d_pos[:,1]
        
        #speed = np.linalg.norm(d_pos, axis=1)
        #print("Speed deviation from v:", np.abs(speed - self.get_prop('v')))
        #print(f'positions: {positions}, swrm.position:{self.positions}  angles: {angles}, swrm.angles: {self.get_prop('angle')}, d_angle: {d_angle}, avg angle based on positions: {np.sin(np.arctan2(y_comp_n,x_comp_n)-self.get_prop('angle')[n])}')      
        #return the derivatives as a single array
        return np.concatenate((d_angle, d_x_pos, d_y_pos))

    def solve_ode_get_states(self, dt, params= None):
        '''
        Solve the ODE for the swarm using the RK45 solver. And get new angle and new x and y position. 
        
        '''
        #print(f'Solving ODEs with dt: {dt})')
        #debugging
        #import pdb; pdb.set_trace()
        #get the initial conditions
        ICs = np.concatenate((self.get_prop('angle'), self.positions[:,0], self.positions[:,1])) #theta_0, x_0, y_0
        #print(ICs.shape)  # should be (N*3,)

        #solve the ODEs
        ode_solns= solve_ivp(self.ODEs, t_span= [0, dt], y0= ICs, method='RK45', rtol=0.0001, atol=1e-06)
        
        #get the new angles, x pos and y pos
        new_angles = ode_solns.y[:self.N, -1] #self.__wrap_angle(ode_solns.y[:self.N, -1])
        new_angles = np.mod(new_angles, 2 * np.pi)  # Ensure angles are in [0, 2π)
        new_x = ode_solns.y[self.N:2*self.N, -1]
        new_y = ode_solns.y[2*self.N:, -1]
        
        #print(f'new_x: {new_x}')
        #print(f'new_angles: {new_angles}')
        #print(f'new_y: {new_y}')

        updated_states= np.concatenate((new_angles, new_x, new_y))

        #print(f' updated_states: {updated_states}')
        
        return updated_states
    
    
    def get_positions(self, dt, params=None):
        ''' A method that updates the positions of the swarm based on the ODEs.
        
            Parameters
            ----------
            dt : float
                The time step for the update.
            params : dict, optional
                Additional parameters for the update (not used in this case).
        
            Returns
            -------
            ndarray
                A 2D array of updated positions of the swarm.
        '''
        
        updated_states = self.solve_ode_get_states(dt, params)

        #update angles and positions
        self.props['angle'] = updated_states[:self.N]
        new_positions= np.zeros((self.N, 2))
        new_positions[:,0] = updated_states[self.N:2*self.N]
        new_positions[:,1]= updated_states[2*self.N:]
        
        return new_positions
        

#set initial conditions 
x_center = 0.2 # +/- 0.025
y_center = 0.2 # +/- 0.075
IC_pos = np.zeros((SWARM_SIZE,2))

##for N=3, domain is 1x1
IC_pos[0,0] = 0.3 #set x initial position for particle 0
IC_pos[0,1] = 0.3 #set y initial position for particle 0
IC_pos[1,0] = 0.5 #0.15  #set x initial position for particle 1
IC_pos[1,1] = 0.3 #set y initial position for particle 1
IC_pos[2,0] = .3
IC_pos[2,1] = 0.8
IC_pos[3,0] = 0.6
IC_pos[3,1] = 0.6
IC_pos[4,0] = 0.3 
IC_pos[4,1] = 0.1 
IC_pos[5,0] = 0.5 
IC_pos[5,1] = 0.5 
IC_pos[6,0] = .3
IC_pos[6,1] = 0.7
IC_pos[7,0] = 0.5
IC_pos[7,1] = 0.6
IC_pos[8,0] = .1
IC_pos[8,1] = 0.4
IC_pos[9,0] = 0.1
IC_pos[9,1] = 0.1



       
#import pdb; pdb.set_trace()
#Create a 3-zone (repulsion and alignment and attraction) swarm. 
np.random.seed(123)
swrm = three_zone_torque(swarm_size=SWARM_SIZE, envir=envir, init=IC_pos, store_prop_history=True) #store_prop_history records props history

#run simulation 
#initalize df with two columns "interation" and "speed"
df = pd.DataFrame(columns=['iteration', 'speed'])

for ii in range(3000): ##was 200
    swrm.move(0.01) #0.01 #similar to dt in Euler step 
    #print("NEW positions inside time loop after move is called:", swrm.positions)
    #print(f'iteration {ii}, speed: {swrm.get_prop("v")}')
    #append to df
    speed = np.linalg.norm(swrm.velocities, axis=1) #calculate speed from velocity vector
    #print(f'speed: {speed}')
    #print(f'iteration {ii}')
    df.loc[len(df)] = {'iteration': ii, 'speed': speed}

#save the df to a csv file- make sure speed is constant
df.to_csv('examples/MK-torque/r-o-a-one_zone/speed_history_one_zone-corrected10.csv')

#plot distances between partciles across time  
#distances = pd.DataFrame(swrm.__calc_dist(swrm.positions[0,:], swrm.positions)) #can't use __calc_dist directly, so use a wrapper function
#print(f'distances: {distances}')

#plt.plot(distances)

#convert to a dataframe
swrm.plot_all(movie_filename=f'examples/MK-torque/r-o-a-threezone/torque-threezone.mp4', fps=100, fluid='quiver', plot_heading=True) # realtime #4 #min v0, max v0, r_r, r_o, r_a, dt, constant xflow, constant yflow, iter, frames