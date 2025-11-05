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
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap, BoundaryNorm #for heat maps
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

# Extract time vector (in seconds, not time steps)
time = np.sort(df_agent_positions['time'].unique())

# --- 1️⃣ Mean minimum distance over time ---
#have each line be a different color
plt.figure(figsize=(7, 4))
plt.plot(time, min_distances, color='tab:blue' , lw=1.8)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Mean minimum distance', fontsize=12)
plt.title('Mean Minimum Distance to Closest Agent Over Time', fontsize=13)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('examples/MK-torque/r-o-a-threezone/mean_min_distance_to_closest_agent.png', dpi=300)
plt.show()

# choose a colormap and generate colors for each agent
n_agents= min_distances.shape[1]
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, n_agents))

# create figure and axis explicitly
fig, ax = plt.subplots(figsize=(7, 4))

# plot each agent's minimum distance with its gradient color
for a in range(n_agents):
    ax.plot(time, min_distances[:, a], color=colors[a], lw=1.8, alpha=0.8)

ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Minimum distance to closest agent', fontsize=12)
ax.set_title('Minimum Distance to Closest Agent Over Time', fontsize=13)
ax.grid(alpha=0.3)

# create and attach colorbar to the same axis
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_agents-1))
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Agent ID', rotation=270, labelpad=15)

fig.tight_layout()
plt.savefig('examples/MK-torque/r-o-a-threezone/min_distance_each_agent_gradient.png', dpi=300)
plt.show()


# --- 2️⃣ Mean ± std minimum distance ---
avg_min_dist= np.mean(min_distances, axis=1)
std_dist_min = np.std(min_distances, axis=1)
plt.figure(figsize=(7, 4))
plt.plot(time, avg_min_dist, color='tab:blue', lw=1.8, label='Mean minimum distance')
plt.fill_between(time, 
                    avg_min_dist - std_dist_min, 
                 avg_min_dist + std_dist_min,
                 color='tab:blue', alpha=0.2, label='±1 std')
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Mean minimum distance', fontsize=12)
plt.title('Mean ± Std of Minimum Distances to Closest Agent', fontsize=13)
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('examples/MK-torque/r-o-a-threezone/mean_min_distance_to_closest_agent_std.png', dpi=300)
plt.show()

# --- 3️⃣ Histogram of final-time minimum distances ---
final_min_distances = min_distances[-1, :]
plt.figure(figsize=(6, 4))
plt.hist(final_min_distances, bins=20, color='tab:orange', edgecolor='black', alpha=0.8)
plt.xlabel('Minimum distance (final time step)', fontsize=12)
plt.ylabel('Number of agents', fontsize=12)
plt.title('Distribution of Closest-Agent Distances at Final Time Step', fontsize=13)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('examples/MK-torque/r-o-a-threezone/histogram_min_distance_final_timestep.png', dpi=300)
plt.show()

# --- 4️⃣ Mean & std of all pairwise distances over time ---
mask = ~np.eye(distances.shape[1], dtype=bool)
mean_dist = np.mean(distances[:, mask], axis=1)
std_dist = np.std(distances[:, mask], axis=1)

plt.figure(figsize=(7, 4.5))
plt.plot(time, mean_dist, color='tab:green', lw=2, label='Mean pairwise distance')
plt.fill_between(time,
                 mean_dist - std_dist,
                 mean_dist + std_dist,
                 color='tab:green', alpha=0.2, label='±1 std')
plt.plot(time, avg_min_dist, color='tab:blue', lw=1.8, label='Minimum pairwise distance')
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.title('Mean and Standard Deviation of Pairwise Distances Over Time', fontsize=13)
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('examples/MK-torque/r-o-a-threezone/mean_pairwise_distance_std.png', dpi=300)
plt.show()

# --- 5️⃣ Average distance from an agent to all others over time  ---
#ie for each agent, at each time, we want the average distance to all other agents

#now for each time (each row) and each agent (each column), find the minimum distance to another agent
# avg_distances: shape (n_times, n_agents)
avg_distances = np.zeros((distances.shape[0], distances.shape[1]))

for t in range(distances.shape[0]):
    for a in range(distances.shape[1]):
        # exclude self-distance by masking the diagonal
        mask = np.ones(n_agents, dtype=bool)
        mask[a] = False
        avg_distances[t, a] = np.mean(distances[t, a, mask])
# gradient colormap
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, n_agents))

# create figure and axis
fig, ax = plt.subplots(figsize=(7, 4))

# plot each agent's average distance with gradient color
for a in range(n_agents):
    ax.plot(time, avg_distances[:, a], color=colors[a], lw=1.5, alpha=0.8)

ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Average distance to all other agents', fontsize=12)
ax.set_title('Average Distance to All Other Agents Over Time', fontsize=13)
ax.grid(alpha=0.3)

# optional: create a legend for the first few agents only (if too many agents)
if n_agents <= 15:
    ax.legend([f'Agent {i}' for i in range(n_agents)], frameon=False, fontsize=9)


fig.tight_layout()
plt.savefig('examples/MK-torque/r-o-a-threezone/average_distance_to_all_agents_gradient.png', dpi=300)
plt.show()


#min distance distribution over time
plt.figure(figsize=(8,5))
plt.hist(min_distances, bins=30, density=True, histtype='stepfilled', alpha=0.7)
plt.xlabel('Minimum Distance to Closest Agent', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of Minimum Distances Over Time', fontsize=13)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('examples/MK-torque/r-o-a-threezone/heatmap_min_distance_over_time.png', dpi=300)
plt.show()

#2️⃣ Pairwise distance matrix snapshot
#Pick a time (e.g., final timestep), and plot a heatmap of all pairwise distances.
#Can reveal clusters or isolated agents visually.
pairwise_snapshot = distances[-1, :, :].copy()
# Create a colormap (e.g., viridis) with more discrete bins
vmax = np.nanmax(pairwise_snapshot[np.isfinite(pairwise_snapshot)])
n_colors = 20  # more bins = more subtle differences
cmap = plt.cm.viridis
bounds = np.linspace(0, vmax, n_colors + 1)  # define bin edges
norm = BoundaryNorm(bounds, cmap.N)
np.fill_diagonal(pairwise_snapshot, 0)  # distance to self = 0
plt.figure(figsize=(6,5))
plt.imshow(pairwise_snapshot, cmap=cmap, origin='lower', norm=norm)
plt.colorbar(label='Distance')
plt.xlabel('Agent ID')
plt.ylabel('Agent ID')
plt.title('Pairwise distances at final time step')
plt.tight_layout()
plt.savefig('examples/MK-torque/r-o-a-threezone/pairwise_distance_heatmap_final_timestep.png', dpi=300)
plt.show()

#################################
#make this a movie over time
# copy of distances to avoid modifying original
pairwise_time_series = distances.copy()

n_times = pairwise_time_series.shape[0]
fig, ax = plt.subplots(figsize=(6,5))
# optional: set global vmin/vmax for consistent colors
# Colormap and normalization
vmax = np.nanmax(pairwise_time_series[np.isfinite(pairwise_time_series)])
n_colors = 20
cmap = plt.cm.viridis
bounds = np.linspace(0, vmax, n_colors + 1)
norm = BoundaryNorm(bounds, cmap.N)

# Initial plot (needed to create the colorbar)
snapshot0 = pairwise_time_series[0]
im = ax.imshow(snapshot0, cmap=cmap, origin='lower', norm=norm)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Distance")

# Set axis labels and title once
ax.set_xlabel("Agent ID")
ax.set_ylabel("Agent ID")
title = ax.set_title(f"Pairwise distances at time step 0")

# Update function for animation
def update(frame):
    snapshot = pairwise_time_series[frame].copy()
    np.fill_diagonal(snapshot, 0)  # distance to self = 0
    im.set_data(snapshot)           # update the image data
    title.set_text(f"Pairwise distances at time step {frame}")
    print(frame)
    return [im, title]

# Create animation
anim = FuncAnimation(fig, update, frames=n_times, blit=False)

# Save as mp4 using ffmpeg
writer = FFMpegWriter(fps=100, metadata=dict(artist='MK'), bitrate=1800)
anim.save('examples/MK-torque/r-o-a-threezone/pairwise_distances_movie.mp4', writer=writer)

plt.close(fig)

#########
#Mean min distance distribution at final time step across multiple simulations