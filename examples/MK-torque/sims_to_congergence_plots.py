# we will read in csv files from Planktos simulations
# we will then compute the minimum distance to the closest agent at each time
# we will then plot the average minimum distance to closest agent over time 

#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from matplotlib.animation import FuncAnimation, FFMpegWriter
#from matplotlib.colors import ListedColormap, BoundaryNorm #for heat maps
import os #for directory reading

#####FORMAT OF CSV FILES#####
# import csv which is an (N x 4) dataframe 
# N is the swarm size and 4 columns are agent, time, x, y, 
# there is a header
# staring at row 0, the first inter * dt rows are for agent 0
# then the next inter * dt rows are for agent 1 and so on 

# MAKE SURE ALL OF THE CSV FILES ARE IN THE SAME DIRECTORY 

# read in the path to the directory where all the simulation csv files are stored
dir_path="examples/MK-torque/r-o-a-threezone/csv_files"

# you also need to give some parameters used in the simulations
Lx = 10 # length of the domain in the x direction
Ly = 10 # length of the domain in the y direction
dim= 2 # dimension of the simulation 
x_bndry= 'periodic' # x boundary condition
y_bndry= 'periodic' # y boundary condition


def dist_origin_to_other_agents(df,Lx, Ly, x_bndry, y_bndry, dim):
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
        # get length of each spatial dimension
        domain = [Lx, Ly]

        n_times= len(df[df['agent']==0]['time']) #number of time steps
        print(n_times)
        n_agents= len(df['agent'].unique()) #list of all agents) #number of agents
        print(n_agents)
        diffs = np.zeros((n_times, n_agents, dim)) #iter * dt x N array to store diffs
        dist = np.zeros((n_times, n_agents, n_agents)) #iter * dt x N array to store distances


        #get an array of all positions over time for all agents
        positions= np.zeros((n_times, n_agents, dim))
        for i in range(n_agents):
            # Convert the string into a real DataFrame
            xi_df = df[df['agent']==i]['x']
            yi_df = df[df['agent']==i]['y']
            print(xi_df.head())
            print(len(xi_df))
            positions[:, i, 0] = xi_df
            positions[:, i, 1] = yi_df
        for n in range(n_agents): #loop over all agents
            
            origin_x = positions[:, n, 0] # x positions for agent n for all time (iter * dt x 1)
            origin_y = positions[:, n, 1] # y positions of agent n for all time (iter * dt x 1)
        
            x_delta = positions[:, : ,0] - origin_x[:, np.newaxis] # vectorized difference in x positions (iter * dt x N)
            y_delta = positions[:, : ,1] - origin_y[:, np.newaxis] # vectorized difference in y positions (iter * dt x N)
        
            if x_bndry == 'periodic':
                diffs[:, :, 0] = (x_delta + domain[0]/2) % domain[0] - domain[0]/2

            else:
                diffs[:, :, 0] = x_delta

            if y_bndry == 'periodic':
                diffs[:, :, 1] = (y_delta + domain[1]/2) % domain[1] - domain[1]/2

            else:
               diffs[:, :, 1] = y_delta
       
            dist[:, n, :] = np.sqrt(diffs[:, :, 0]**2 + diffs[:, :, 1]**2) #store the distances in a matrix for later use            
        return dist #return a matrix of distances (iter * dt x N)

avg_min_dist_all_sims = [] #to store average minimum distances for all simulations
std_dist_min_all_sims = [] #to store std of minimum distances for all simulations
#loop through all the csv files in the directory
for file in os.listdir(dir_path):
    file_path = os.path.join(dir_path, file)
    if file.endswith(".csv"):
        #read in the csv file
        df = pd.read_csv(file_path)
        distances=dist_origin_to_other_agents(df=df,Lx=Lx, Ly=Ly, x_bndry=x_bndry, y_bndry=y_bndry, dim=dim)
        #now for each time (each row) and each agent (each column), find the minimum distance to another agent
        min_distances = np.zeros((distances.shape[0], distances.shape[1])) #n_times ,n_agents
        for t in range(distances.shape[0]): #loop over time
            for a in range(distances.shape[1]): #loop over agents
                #set the distance to itself to be infinity so we don't pick it as the minimum
                distances[t, a, a] = np.inf
                min_distances[t, a] = np.min(distances[t, a, :]) #find the minimum distance to another agent
                # Extract time vector (in seconds, not time steps)
        time = np.sort(df['time'].unique())
        avg_min_dist= np.mean(min_distances, axis=1)
        std_dist_min = np.std(min_distances, axis=1)
        avg_min_dist_all_sims.append(avg_min_dist)
        std_dist_min_all_sims.append(std_dist_min)

# save avg_min_dist_all_sims and std_dist_min_all_sims to csv files
avg_min_dist_df = pd.DataFrame(avg_min_dist_all_sims).T
avg_min_dist_df.to_csv('avg_min_dist_all_sims.csv', index= False)
std_dist_min_df = pd.DataFrame(std_dist_min_all_sims).T
std_dist_min_df.to_csv('std_dist_min_all_sims.csv', index=False)

# create a plot for each simulation
for i in range(len(avg_min_dist_all_sims)):
    plt.figure()
    plt.plot(time, avg_min_dist_all_sims[i], label='Average Minimum Distance to Closest Agent')
    plt.fill_between(time, 
                     avg_min_dist_all_sims[i] - std_dist_min_all_sims[i], 
                     avg_min_dist_all_sims[i] + std_dist_min_all_sims[i], 
                     color='b', alpha=0.2, label='±1 Std Dev')
    plt.xlabel('Time')
    plt.ylabel('Distance')
    plt.title(f'Simulation {i}')
    plt.legend()
    plt.grid()
    plt.savefig(f'simulation_{i}_min_distance_plot.png')

#create a plot which plots the average of all simulations on the same plot 
plt.figure()
for i in range(len(avg_min_dist_all_sims)):
    plt.plot(time, avg_min_dist_all_sims[i], label=f'Simulation {i}')
plt.xlabel('Time')
plt.ylabel('Distance')
plt.title('Average Minimum Distance to Closest Agent Across Simulations')
plt.legend()
plt.grid()
plt.savefig('all_simulations_min_distance_plot.png')
