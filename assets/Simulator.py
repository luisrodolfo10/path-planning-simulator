import pygame
import math
from itertools import chain
import random as randy
from math import sqrt
import csv
import time


#COLORES
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

TURTLESIM_BGC = (69,86,255)

###Tortuguitas img
box = "Tortuguitas/box-turtle.png"
diamond = "Tortuguitas/diamondback.png"
electric = "Tortuguitas/electric.png"
fuerte = "Tortuguitas/fuerte.png"
groovy = "Tortuguitas/groovy.png"
hydro = "Tortuguitas/hydro.png"
indigo = "Tortuguitas/indigo.png"
jade = "Tortuguitas/jade.png"
kinetic = "Tortuguitas/kinetic.png"
lunar = "Tortuguitas/lunar.png"
melodic = "Tortuguitas/melodic.png"
sea = "Tortuguitas/sea-turtle.png"

#------------------------Clases-----------------------------------------------------------------------
class turtle():
	def __init__(self, win, width, rows, pos, img, grid, obstacles):
		self.win = win
		self.rows = rows
		self.pos = pos    #[x,y]
		self.img = img
		self.width = width
		self.grid = grid
		self.angle = 90
		self.obstacles = obstacles

	def draw(self):
		x = self.pos[0]
		y = self.pos[1]
		gap = self.width // self.rows  		#Gap between the rows

		x += 1								#For no wall positions
		y = self.rows - y - 1				#For Y axis go from down to up, and no wall positions
		#Canvas coordinates
		CX = x * gap	
		CY = y * gap
		self.win.blit(self.img, (CX-20, CY-20))   #Center the turtle in the position

	def up(self):
		if [self.pos[0], self.pos[1] + 1] in self.grid and [self.pos[0], self.pos[1] + 1] not in self.obstacles:    #If the next position it's not in the grid, then it's an invalid position
			self.pos[1] += 1
		else:
			print("Invalid position")

		#Rotation of image
		if self.angle == 0:
			self.img = pygame.transform.rotate(self.img, 90)
		elif self.angle == 180:
			self.img = pygame.transform.rotate(self.img, -90)
		elif self.angle == 270:
			self.img = pygame.transform.rotate(self.img, 180)
		self.angle = 90


	def down(self):
		if [self.pos[0], self.pos[1] - 1] in self.grid and [self.pos[0], self.pos[1] - 1] not in self.obstacles:
			self.pos[1] -= 1
		else:
			print("Invalid position")

		#Rotation of image
		if self.angle == 0:
			self.img = pygame.transform.rotate(self.img, -90)
		elif self.angle == 180:
			self.img = pygame.transform.rotate(self.img, 90)
		elif self.angle == 90:
			self.img = pygame.transform.rotate(self.img, 180)
		self.angle = 270

	def right(self):
		if [self.pos[0] + 1, self.pos[1]] in self.grid and [self.pos[0] + 1, self.pos[1]] not in self.obstacles:
			self.pos[0] += 1
		else:
			print("Invalid position")

		#Rotation of image
		if self.angle == 90:
			self.img = pygame.transform.rotate(self.img, -90)
		elif self.angle == 180:
			self.img = pygame.transform.rotate(self.img, -180)
		elif self.angle == 270:
			self.img = pygame.transform.rotate(self.img, 90)
		self.angle = 0

	def left(self):
		if [self.pos[0] - 1, self.pos[1]] in self.grid and [self.pos[0] - 1, self.pos[1]] not in self.obstacles:		
			self.pos[0] -= 1
		else:
			print("Invalid position")

		#Rotation of image
		if self.angle == 90:
			self.img = pygame.transform.rotate(self.img, 90)
		elif self.angle == 0:
			self.img = pygame.transform.rotate(self.img, 180)
		elif self.angle == 270:
			self.img = pygame.transform.rotate(self.img, -90)
		self.angle = 180
	### Moore ###
	def ne(self):
		if [self.pos[0] +1, self.pos[1] + 1] in self.grid and [self.pos[0] +1, self.pos[1] + 1] not in self.obstacles:    #If the next position it's not in the grid, then it's an invalid position
			self.pos[0] += 1
			self.pos[1] += 1
		else:
			print("Invalid position")
	def se(self):
		if [self.pos[0]+1, self.pos[1] - 1] in self.grid and [self.pos[0]+1, self.pos[1] -1] not in self.obstacles:    #If the next position it's not in the grid, then it's an invalid position
			self.pos[0] += 1
			self.pos[1] -= 1
		else:
			print("Invalid position")
	def sw(self):
		if [self.pos[0]-1, self.pos[1] - 1] in self.grid and [self.pos[0] -1, self.pos[1] - 1] not in self.obstacles:    #If the next position it's not in the grid, then it's an invalid position
			self.pos[0] -= 1
			self.pos[1] -= 1
		else:
			print("Invalid position")
	def nw(self):
		if [self.pos[0]-1, self.pos[1] + 1] in self.grid and [self.pos[0] - 1, self.pos[1] + 1] not in self.obstacles:    #If the next position it's not in the grid, then it's an invalid position
			self.pos[0] -= 1
			self.pos[1] += 1
		else:
			print("Invalid position")

#--------------------------------Funciones--------------------------------------------------------

def draw_grid(win, rows, width):			#Dibuja la cuadricula
	gap = width // rows
	for i in range(rows):
		pygame.draw.line(win, BLACK, (0, i * gap), (width, i*gap))
		pygame.event.pump()
		
	for j in range(rows):
		pygame.draw.line(win, BLACK, (j * gap, 0), (j*gap, width))
		pygame.event.pump()

def drawObstacles_Goal(win, rows, width, obstacles, goal):
	gap = width // rows
	for obstacle in obstacles:
		x = obstacle[0] 
		y = obstacle[1]
		x += 1								#For no wall positions
		y = rows - y - 1				#For Y axis go from down to up, and no wall positions
		#Canvas coordinates
		CX = x * gap	
		CY = y * gap
		pos = (CX-gap/2,CY-gap/2, gap, gap)
		pos2 = (CX-gap/2+5,CY-gap/2+5, gap-10, gap-10)
		pygame.draw.rect(win, BLACK, pos)
		pygame.draw.rect(win, GREY, pos2)
		pygame.event.pump()
	x_g = goal[0]
	y_g = goal[1]
	x_g += 1
	y_g = rows - y_g -1
	CX_g = x_g * gap
	CY_g = y_g * gap
	pos = [CX_g, CY_g]
	pygame.draw.circle(win, BLACK, pos, 20)
	pygame.draw.circle(win, RED, pos, 15)

def drawXv(win, Xv, flag):
	i = 0
	for k in Xv:
		if i > 0:
			kx = k[0]
			ky = k[1]
			kx += 1
			ky = rows - ky - 1
			CX = kx * gap
			CY = ky * gap
			pos = [CX, CY]
			pygame.draw.circle(win, GREEN, pos, 15)
			if flag:
				pygame.display.update()
				pygame.event.pump()
				pygame.time.wait(0)
		i = 1

def draw(win, rows, width, turtle, obstacles, goal, Xv, flag, route):
	win.fill(TURTLESIM_BGC)
	draw_grid(win, rows, width)
	drawObstacles_Goal(win, rows, width, obstacles, goal)
	turtle.draw()
	drawXv(win, Xv, flag)
	draw_route(win, rows, width, route)
	pygame.display.update()
	pygame.event.pump()




#Funciones para dijkststra y A*

def manhattan(p1, p2):
	x1, y1 = p1[0], p1[1]
	x2, y2 = p2[0], p2[1]
	return abs(x1-x2) + abs(y1-y2)

def euclidian(p1, p2):
    return math.sqrt( (p1[0] - p2[0])**2 + (p1[1]-p2[1])**2 )

def draw_route(win, rows, width, route):
	gap = width//rows
	for i in range(len(route)-1):
		C_x1 = (route[i][0] + 1)*gap
		C_y1 = (rows - route[i][1] -1)*gap 
		C_x2 = (route[i+1][0] + 1)*gap
		C_y2 = (rows - route[i+1][1] - 1)*gap
		pygame.draw.line(win, RED, [C_x1, C_y1], [C_x2, C_y2], 8)
		pygame.event.pump()
		

def make_grid(rows):
	grid = []
	rows -= 1
	for i in range(rows):
		for j in range(rows):
			grid.append((i,j))
	return grid


def make_graph(grid, U, obstacles):
    flag = 1 if len(U) == 4 else 0
    graph = {}
    for cell in grid:
        if cell not in obstacles:
            graph[cell] = {}
            for j in range(len(U)):
                ux = U[j]
                nx = (cell[0]+ux[0], cell[1]+ux[1])
                if nx not in obstacles and nx in grid:
                    graph[cell][nx] = 1 if (j % 2 == 0 or flag) else math.sqrt(2)

    return graph

def dijkstra(graph,start,goal):
	dist = {}
	from_nodes = {}
	visitedNodes = []
	unvisitedNodes = graph
	inf = 9999999
	path = []
	for node in unvisitedNodes:
		dist[node] = inf
	dist[start] = 0

	while unvisitedNodes:
		smNode = min(unvisitedNodes, key=lambda node: dist[node])   #Smallest node 

		for nbNode, weight in graph[smNode].items():   #nb = Neighbor
			if weight + dist[smNode] < dist[nbNode]:
				dist[nbNode] = weight + dist[smNode]
				from_nodes[nbNode] = smNode

		unvisitedNodes.pop(smNode)
		visitedNodes.append(smNode)
		if smNode == goal:
			break
    
	C_Node = goal
	if dist[goal] == inf:
		print("The path doesn't exist")
		return None
		
	while C_Node != start:
		path.insert(0, C_Node)
		C_Node = from_nodes[C_Node]

	path.insert(0,start)

	# print('The path distance is: ', dist[goal])
	# print('The path: ', path)
	# print("Visited nodes", visitedNodes)

	return dist[goal], path, visitedNodes

def a_star(graph,start,goal):
    dist = {}
    dist_goal = {}
    dist_total = {}
    from_nodes = {}
    visitedNodes = []
    unvisitedNodes = graph
    inf = 9999999
    path = []
    for node in unvisitedNodes:
        dist[node] = inf
    dist[start] = 0

    for node in unvisitedNodes:
        dist_goal[node] = euclidian(node, goal)

    #update total distances
    for node in unvisitedNodes:
        dist_total[node] = dist[node] + dist_goal[node]

    #print(dist_goal)
 
    while unvisitedNodes:

        smNode = min(unvisitedNodes, key=lambda node: dist_total[node])   #Smallest node 
        #print(smNode)
   
        for nbNode, weight in graph[smNode].items():   #nb = Neighbor
            if weight + dist[smNode] < dist[nbNode]:
                dist[nbNode] = weight + dist[smNode]
                dist_total[nbNode] = dist[nbNode] + dist_goal[nbNode]
                from_nodes[nbNode] = smNode

        unvisitedNodes.pop(smNode)
        visitedNodes.append(smNode)

        if smNode == goal:
            break
    
    C_Node = goal
    if dist_total[goal] == inf:
        print("The path doesn't exist")
        return None
        
    while C_Node != start:
        path.insert(0, C_Node)
        C_Node = from_nodes[C_Node]

    path.insert(0,start)

    # print('The path distance is: ', dist[goal])
    # print('The path: ', path)
    # print("Visited nodes", visitedNodes)
    # print("N of visitedNodes: ", len(visitedNodes))

    return dist[goal], path, visitedNodes


###Codigo
if __name__ == "__main__":


	resols = [1,2,4,8]

    # res = int(input("""
    # 0) 1.000
    # 1) 0.500
    # 2) 0.250
    # 3) 0.125
    # Seleccione la resolución: 
    # """))

    # vec = int(input("""
    # 0) Von Neumann
    # 1) Moore
    # Seleccione la vecindad: 
    # """))

    # alg = int(input("""
    # 0) Dijkstra
    # 1) A*
    # Seleccione el tipo de algoritmo: 
    # """))

	res = 1
	vec = 1
	alg = 0


	#Variables
	res = resols[res]
	pos = input("Ingrese la posición inicial ex. 2,2: ").split(",")
	pos = [int(x)*res for x in pos]
	goal = input("Ingrese la posicion final ex. 7,7: ").split(",")
	goal = [int(x)*res for x in goal]

	rows = (10 * res) + 2 #the shape of the grid will be  rows-2*rows-2
	run = True
	grid = make_grid(rows)

	### Obstacles CSV
	maps = ("Maps/map01.csv", "Maps/map02.csv", "Maps/map03.csv", "Maps/map04.csv")

	obstacles = []
	with open(maps[1], newline='') as f:
		reader = csv.reader(f)
		data = list(reader)

	data = [list(map(lambda x: int(float(x)), x)) for x in data]  #Transforming all the strings to ints


	####  REVISAR OBSTACULOS
	
	# one_rc = {1:1, 2:2, 4:4, 8:5}

	# for line in data:
	# 	horizontal = one_rc[res] if line[2]==1 else (line[2]*res - (res-1))
	# 	vertical = one_rc[res] if line[3]==1 else (line[3]*res - (res-1))
	# 	for k in range(vertical):
	# 		for j in range(horizontal):
	# 			obs = (line[0]*res+j, line[1]*res+k)
	# 			#print(obs)
	# 			obstacles.append(obs)

	for line in data:
		for k in range((line[3]+1)*res - (res-1)):
			for j in range((line[2]+1)*res - (res-1)):
				obs = (line[0]*res+j, line[1]*res+k)
				obstacles.append(obs)

	#Dijkstra y A*
	U_moore = [[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1, 0], [-1, 1]]
	U_von = [[0,1], [1,0], [0,-1], [-1, 0]]

	U = U_von if vec == 0 else U_moore

	graph = make_graph(grid, U, obstacles)
	#obstacles = [[x[0]/res, x[1]/res] for x in obstacles]
	#print(obstacles)
	#obstacles = []

	###pygame
	pygame.init()

	WIDTH = 800
	WIN = pygame.display.set_mode((WIDTH, WIDTH))

	pygame.display.set_caption("Path planning")
	#Icon code
	icon = pygame.image.load("Tortuguitas/jade.png")
	icon = pygame.transform.scale(icon, (32, 32))
	surface= pygame.Surface(icon.get_size())
	key = (0,255,0)
	surface.fill(key)
	surface.set_colorkey(key)
	surface.blit(icon, (0,0))
	pygame.display.set_icon(icon)

	turtle_list = [box, diamond, electric, fuerte, hydro, indigo, jade, kinetic, lunar, melodic, sea]
	r_turtle = turtle_list[randy.randint(0,10)]

	img = pygame.image.load(r_turtle)
	img = pygame.transform.scale(img, (40, 45))

	turtle = turtle(WIN, WIDTH, rows, pos, img, grid, obstacles)
	gap = WIDTH // rows

	Xv = [] 
	flag = True
	route_d = []
	while run:
		draw(WIN, rows, WIDTH, turtle, obstacles, goal, Xv, flag, route_d)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False

			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE:
					
					W_t, route_d, Xv = dijkstra(graph, tuple(pos), tuple(goal)) if alg == 0 else a_star(graph, tuple(pos), tuple(goal))
					drawXv(WIN, Xv, flag)
					print("Estados visitados: ", [[x[0]/res, x[1]/res] for x in Xv])
					print("Numero de estados factibles: ", len(Xv))
					route = [[x[0]/res, x[1]/res] for x in route_d]
					print("La ruta es: ", route)
					print("El costo de la ruta es: ", W_t/res)
					flag = False
