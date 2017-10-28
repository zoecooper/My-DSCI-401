
#I. FLATTEN
def Flatten(any_list):
    if isinstance(any_list, list):
        if len(any_list) == 0:
            return []
        first, rest = any_list[0], any_list[1:]
        return Flatten(first) + Flatten(rest)
    else:
        return [any_list]
        
#II. POWER SET
def powerset(any_list):
    if not any_list:
        return [[]]
    first = [ [any_list[0]] + rest for rest in powerset(any_list[1:]) ]
    wofirst = powerset(any_list[1:])
    return first + wofirst
    
#III. ALL PERMUTATIONS (I know this is not maybe what you were looking for, but was all I could figure out bc concat issues.)
from itertools import permutations
alist = list(permutations(range(1, 4)))
print alist

#IV. NUMBER SPIRAL (Did not know how to set up directions or print matrices but did my best!)
#Figure out how to do move up/down/left/right
NORTH, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0) 
move_right = {NORTH: E, E: S, S: W, W: NORTH} 

def spiral(width, height, end_corner):
	end_corner = 1, 2,  3 or 4
		if not 1, 2, 3 or 4
			print('Can't do!')
    if width < 1 or height < 1:
        raise ValueError
    x, y = width // 2, height // 2 
    dx, dy = NORTH 
    matrix = [[None] * width for _ in range(height)]
    count = 0
    while True:
        count += 1
        matrix[y][x] = count 
        changed_dx, changed_dy = move_right[dx,dy]
        changed_x, changed_y = x + changed_dx, y + changed_dy
        if (0 <= changed_x < width and 0 <= changed_y < height and
            matrix[changed_y][changed_x] is None): 
            x, y = changed_x, changed_y
            dx, dy = changed_dx, changed_dy
        else: 
            x, y = x + dx, y + dy
            if not (0 <= x < width and 0 <= y < height):
                return matrix 
		
#print the matrix made
def print_matrix(matrix):
    width = len(str(max(el for row in matrix for el in row if el is not None)))
    form = "{:0%dd}" % width
  