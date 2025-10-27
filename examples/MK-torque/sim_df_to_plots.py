#import csv which is an (N x 4) dataframe 
#N is the swarm size and 4 columns are agent, time, x, y, 
#there is a header
#staring at row 0, the first inter * dt rows are for agent 0
#then the next inter * dt rows are for agent 1 and so on 



#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

#read in csv file
filename= 'examples/MK-torque/r-o-a-threezone/torque-threezone-dfpositions.csv'
df_agent_positions = pd.read_csv(filename)
shape= df_agent_positions.shape
print(shape)
print(df_agent_positions.head())

#example of how to access the data
#dataframe of x positions for agent 0
x_positions_agent0 = df_agent_positions[df_agent_positions['agent']==0]['x']
print(x_positions_agent0.head())

# goal: mean distance to the cloest agent over time
# for each agent, calculate the distance to the closest agent over time 
#recall we have a function to calculate distance between two agents wrt periodic boundary conditions

def dist_origin_to_other_agents(df,Lx=1, Ly=1, x_bndry='periodic', y_bndry='periodic', dim=2):
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

        n_times= len(df_agent_positions[df_agent_positions['agent']==0]['time']) #number of time steps
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
            y_delta = positions[:, : ,0] - origin_y[:, np.newaxis] # vectorized difference in y positions (iter * dt x N)
        
            if x_bndry == 'periodic':
                diffs[:, :, 0] = (x_delta + domain[0]/2) % domain[0] - domain[0]/2

            else:
                diffs[:, :, 0] = x_delta

            if y_bndry == 'periodic':
                diffs[:, :, 0] = (y_delta + domain[1]/2) % domain[1] - domain[1]/2

            else:
               diffs[:, :, 0] = y_delta
       
            dist[:, n, :] = np.sqrt(diffs[:, :, 0]**2 + diffs[:, :, 1]**2) #store the distances in a matrix for later use            
        return dist #return a matrix of distances (iter * dt x N)

distances= dist_origin_to_other_agents(df=df_agent_positions,Lx=1, Ly=1, x_bndry='periodic', y_bndry='periodic')
print(distances.shape) #(n_times, n_agents, n_agents)

#now for each time (each row) and each agent (each column), find the minimum distance to another agent
min_distances = np.zeros((distances.shape[0], distances.shape[1])) #n_times ,n_agents
for t in range(distances.shape[0]): #loop over time
    for a in range(distances.shape[1]): #loop over agents
        #set the distance to itself to be infinity so we don't pick it as the minimum
        distances[t, a, a] = np.inf
        min_distances[t, a] = np.min(distances[t, a, :]) #find the minimum distance to another agent
print(min_distances.shape) #(n_times, n_agents)
print(min_distances)

#now for each time, find the mean of the minimum distances across all agents
mean_min_distances = np.mean(min_distances, axis=1) #n_times
print(mean_min_distances.shape) #(n_times,)
#plot the mean minimum distance over time
plt.figure()
#scatter plot
plt.scatter(df_agent_positions['time'].unique(), mean_min_distances, s=2)
plt.xlabel('Time (sseconds)')
plt.ylabel('Mean minimum distance to closest agent')
plt.title('Mean minimum distance to closest agent over time')
#save the plot
plt.savefig('examples/MK-torque/r-o-a-threezone/mean_min_distance_to_closest_agent.png', dpi=300)
plt.show()





