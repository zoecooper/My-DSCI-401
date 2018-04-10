# Conway's Game of Life: runsim and plots
# Zoe Cooper -- CPSC 420

#import statements
import numpy as np
import matplotlib.pyplot as plt


#First step is to define the functions you wish to use
# Return the number of populated neighbors (0-8) of cell x,y on this grid.

def num_neighbors(grid,x,y,WIDTH, HEIGHT):
	neighbors = 0
	if x < WIDTH-1 and grid[x+1,y] == 1:                        # Right
		neighbors = neighbors + 1
	if x > 0 and grid[x-1,y] == 1:                              # Left
		neighbors = neighbors + 1
	if y < HEIGHT-1 and grid[x,y+1] == 1:                       # Up
		neighbors = neighbors + 1
	if y > 0 and grid[x,y-1] == 1:                              # Down
		neighbors = neighbors + 1
	if x < WIDTH-1 and y < HEIGHT-1 and grid[x+1,y+1] == 1:     # Up-right
		neighbors = neighbors + 1
	if x > 0 and y > 0 and grid[x-1,y-1] == 1:                  # Lower-left
		neighbors = neighbors + 1
	if x < WIDTH-1 and y > 0 and grid[x+1,y-1] == 1:            # Lower-right
		neighbors = neighbors + 1
	if x > 0 and y < HEIGHT-1 and grid[x-1,y+1] == 1:           # Upper-left
		neighbors = neighbors + 1
	return neighbors


# Return True only if cell x,y should be populated on the generation *after*
# the one represented by the grid passed.
def should_be_pop_next_gen(grid,x,y, SURVIVE, BIRTH, WIDTH, HEIGHT):
	if grid[x,y] == 1:
		if num_neighbors(grid,x,y, WIDTH, HEIGHT) in SURVIVE:
			return True
		else:
			return False
	else:
		if num_neighbors(grid,x,y, WIDTH, HEIGHT) in BIRTH:
			return True
		else:
			return False


# Given a 2d array of 1's and 0's, return a list with the x-coordinates (in a
# list) and the y-coordinates (in another list) of the cells that are 1's.
def points_for_grid(grid, WIDTH, HEIGHT):
	xcoords = []
	ycoords = []
	for i in range(0,WIDTH):
		for j in range(0,HEIGHT):
			if grid[i,j] == 1:
				xcoords.append(j)
				ycoords.append(HEIGHT-i-1)
	return [xcoords,ycoords]
	

#Figuring out the total population	
def tot_pop(grid, WIDTH, HEIGHT):
	total_pop = 0
	for i in range(0,WIDTH):
		for j in range(0,HEIGHT):
			if grid[i,j] == 1:
				total_pop += 1
	return total_pop
	



#runsim() runs the simulations with parameters that are able to be manipulated
def runsim(PROB_POP = .3, SURVIVE = [2,3], BIRTH = [3]):
	WIDTH = 20
	HEIGHT = 20
	NUM_GEN = 100
		
	
	cube = np.empty((WIDTH, HEIGHT, NUM_GEN))

# (To hardcode a particular starting configuration:)
#config = np.array(
#    [[0,1,0,0,0,0],
#     [0,0,1,0,0,0],
#     [1,1,1,0,0,0],
#     [0,0,0,0,0,0],
#     [0,0,0,0,0,0],
#     [0,0,0,0,0,0]]
#)

# Create a random starting configuration with about PROB_POP of the cells
# being initially populated.
	config = np.random.choice([1,0],p=[PROB_POP, 1-PROB_POP],size=WIDTH*HEIGHT)
	config.shape = (WIDTH,HEIGHT)
	cube[:,:,0] = config
	
	# Run the simulation.
	for gen in range(1,NUM_GEN):
		for x in range(WIDTH):
			for y in range(HEIGHT):
				if should_be_pop_next_gen(cube[:,:,gen-1],x,y, SURVIVE, BIRTH, WIDTH, HEIGHT):
					cube[x,y,gen] = 1
				else:
					cube[x,y,gen] = 0
					
					
					
	return tot_pop(cube[:,:, NUM_GEN-1], WIDTH, HEIGHT)	

#Computes the mean (average) of the final population counts for 50 runsims.					
probs = np.arange(0,1,.1)
avg_pop = 0
pop_by_gen = []
for prob in probs:
	avg_pop = 0
	for i in range(50):
		num_pop = runsim(PROB_POP = prob, SURVIVE = [2,4], BIRTH = [2,4]) #you can change based on bacteria type
		avg_pop += num_pop
	pop_by_gen.append(avg_pop/50)

pop_by_gen = np.array(pop_by_gen)		

#for gen in range(1, NUM_GEN):
#	avg_pops = runsim(WIDTH = WIDTH,
	#				HEIGHT = HEIGHT,
	#				NUM_GEN = NUM_GEN, 
	#				SURVIVE = SURVIVE,
	#				BIRTH = BIRTH,
		#			PROB_POP = PROB_POP,
		#			plot = False)
#	avg_pop.append(avg_pops)

# Plot total population over time.
#plt.clf()
#avg_pop = np.empty(NUM_GEN)
#for gen in range(0,NUM_GEN):
#	avg_pop[gen] = cube[:,:,gen].sum()
#plt.ylim(0,avg_pop.max()+1)
#plt.xlabel("Generation #")
#plt.ylabel("Total population")
#plt.plot(avg_pop)
#plt.show()

# Simulation parameters.
#WIDTH = 20
#HEIGHT = 20
#NUM_GEN = 50
#PROB_POP = .3    # The fraction of cells that will start off populated.	

# The number of neighbors required for a currently populated cell to stay
# populated.
#SURVIVE = [2,3]

# The number of neighbors required for a currently empty cell to become
# populated.
#BIRTH = [3]



# This 3d array will use the third coordinate as a generation number, and thus
# represent the entire lifetime of the simulated model.
#cube = np.empty((WIDTH, HEIGHT, NUM_GEN))

# (To hardcode a particular starting configuration:)
#config = np.array(
#    [[0,1,0,0,0,0],
#     [0,0,1,0,0,0],
#     [1,1,1,0,0,0],
#     [0,0,0,0,0,0],
#     [0,0,0,0,0,0],
#     [0,0,0,0,0,0]]
#)

# Create a random starting configuration with about PROB_POP of the cells
# being initially populated.
#config = np.random.choice([1,0],p=[PROB_POP, 1-PROB_POP],size=WIDTH*HEIGHT)
#config.shape = (WIDTH,HEIGHT)
#cube[:,:,0] = config


#PLOTS #can change title for different bacteria
plt.plot(probs, pop_by_gen, color = "blue", label="Amount Populated")
plt.title("Bacterium evenum")
plt.xlabel("Probability of being Populated")
plt.ylabel("Avg Final Population")
plt.legend()
plt.show()