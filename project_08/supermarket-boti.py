import numpy as np
import pandas as pd
import cv2
import time
#from a_star import find_path

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

TILE_SIZE = 32
OFS = 50

MARKET = """
##################
##..............##
##..##..##..##..##
#B..#S..#D..#F..##
##..##..##..##..##
#B..#S..#D..#F..##
##..##..##..##..##
##...............#
##..C#..C#..C#...#
##..##..##..##...#
##...............#
##############GG##
""".strip()

MARKET2 = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])


MARKET3 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

class SupermarketMap:
    """Visualizes the supermarket background"""

    def __init__(self, layout, tiles):
        """
        layout : a string with each character representing a tile
        tile   : a numpy array containing the tile image
        """
        self.tiles = tiles
        self.contents = [list(row) for row in layout.split("\n")]
        self.xsize = len(self.contents[0])
        self.ysize = len(self.contents)
        self.image = np.zeros(
            (self.ysize * TILE_SIZE, self.xsize * TILE_SIZE, 3), dtype=np.uint8
        )
        self.prepare_map()

    def get_tile(self, char):
        """returns the array for a given tile character"""
        if char == "#":
            return self.tiles[0:32, 0:32]
        elif char == "G":
            return self.tiles[7 * 32 : 8 * 32, 3 * 32 : 4 * 32]
        elif char == "C":
            return self.tiles[2 * 32 : 3 * 32, 8 * 32 : 9 * 32]
        elif char == "B":
            return self.tiles[3 * 32 : 4 * 32, 13 * 32 : 14 * 32]
        elif char == "S":
            return self.tiles[6 * 32 : 7 * 32, 9 * 32 : 10 * 32]
        elif char == "D":
            return self.tiles[6 * 32 : 7 * 32, 3 * 32 : 4 * 32]
        elif char == "F":
            return self.tiles[0 : 32, 4 * 32 : 5 * 32]
        else:
            return self.tiles[32:64, 64:96]

    def prepare_map(self):
        """prepares the entire image as a big numpy array"""
        for y, row in enumerate(self.contents):
            for x, tile in enumerate(row):
                bm = self.get_tile(tile)
                self.image[
                    y * TILE_SIZE : (y + 1) * TILE_SIZE,
                    x * TILE_SIZE : (x + 1) * TILE_SIZE,
                ] = bm

    def draw(self, frame, offset=OFS):
        """
        draws the image into a frame
        offset pixels from the top left corner
        """
        frame[
            OFS : OFS + self.image.shape[0], OFS : OFS + self.image.shape[1]
        ] = self.image

    def write_image(self, filename):
        """writes the image into a file"""
        cv2.imwrite(filename, self.image)



class Customer:

    def __init__(self, id, location, terrain_map, image, x, y):
        self.id = id
        self.location = location

        self.terrain_map = terrain_map
        self.image = image
        self.x = x
        self.y = y
        
        self.goal = [0,0]
        self.oldlocation = "entrance"

    def draw(self, frame):
        xpos = OFS + self.x * TILE_SIZE
        ypos = OFS + self.y * TILE_SIZE
        frame[ypos: ypos + 32, xpos: xpos + 32] = self.image

        # overlay the Customer image / sprite onto the frame
        
    def moveonmap (self, direction):
        if direction == 'u':
            if self.terrain_map.contents[self.y - 1][self.x] == '.':
                self.y -= 1
        if direction == 'd':
            if self.terrain_map.contents[self.y + 1][self.x] == '.':
                self.y += 1
        if direction == 'r':
            if self.terrain_map.contents[self.y][self.x + 1] == '.':
                self.x += 1
        if direction == 'l':
            if self.terrain_map.contents[self.y][self.x - 1] == '.':
                self.x -= 1

    def __repr__(self):
        return f'Customer ({self.id}, {self.location})'

    def is_active(self):
        '''is the customer still shopping?'''
        if self.location == 'checkout':
            return False
        else:
            return True

    def change_location(self):
        '''where does the customer go next'''
        new_location = np.random.choice(locations, p=matrix.loc[self.location])
        self.location = new_location



class Supermarket:
    """manages multiple Customer instances that are currently in the market.
    """

    def __init__(self, marketmap):        
        # a list of Customer objects
        self.customers = []
        self.minutes = 419
        self.last_id = 0
        self.marketmap = marketmap
        
    def __repr__(self):
        return f'T: {self.minutes}, Customers: {self.customers}'

    def get_time(self):
        """current time in HH:MM format,
        """
        hours = str(self.minutes//60)
        if len(hours) == 1:
            hours = '0' + hours
        minutes = str(self.minutes % 60)
        if len(minutes) == 1:
            minutes = '0' + minutes
        return f'{hours}:{minutes}'

    def print_customers(self):
        """print all customers with the current time and id in CSV format.
        """
        currentstatus = []
        for element in self.customers:
            currentstatus.append([self.get_time(),element.id,element.location])
        return currentstatus            

    def next_minute(self, frame):
        """propagates all customers to the next state.
        """
        self.minutes += 1
        self.marketmap.draw(frame)        
        for customer in self.customers:
            if customer.oldlocation != 'stalled':       # if customer is not stalled, define new location and set goal on route map
                customer.oldlocation = customer.location
                customer.change_location()            
            newlocation = customer.location
            if customer.oldlocation != newlocation:
                shortestdistance = 100
                for locelement in waitinglocation[newlocation]:                 # check destinations
                    if (MARKET3[locelement[1]][locelement[0]] == 1):             # is it free?
                        distancefromlocation = (abs(locelement[0]-customer.x) + abs(locelement[1]-customer.y))
                        if distancefromlocation < shortestdistance:             # is it the closest?
                            customer.goal = locelement
                            shortestdistance = distancefromlocation

                cv2.rectangle(frame, (0,500),(1000,600), (00,0,0), 50)
                if ghostinvasion == 1:
                    cv2.putText(frame, f'Customer {customer.id} moves from {customer.oldlocation} to: {newlocation}.', (0, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)                     
                elif np.random.rand() < .9:
                    cv2.putText(frame, f'Customer {customer.id} moves from {customer.oldlocation} to: {newlocation}.', (0, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2) 
                else:
                    cv2.putText(frame, 'Secret hint: long press "g" for ghost invasion... if you dare!', (0, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2) 
                print(f'Customer {customer.id} moves from {customer.oldlocation} to: {newlocation}.')
            else:   

                cv2.rectangle(frame, (0,500),(1000,600), (00,0,0), 50)
                cv2.putText(frame, f'Customer {customer.id} is taking some more time at: {customer.oldlocation}.', (0, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)                 
                print(f'Customer {customer.id} is taking some more time at: {customer.oldlocation}.')
            customer.draw(frame)
        
    def add_new_customers(self):
        """randomly creates new customers.
        """
        if np.random.rand() < .33:
            self.last_id += 1
            customerid = str(self.last_id)
            self.customers.append(Customer(customerid,'entrance', marketmap, tiles[4 * 32 : 5 * 32, 2 * 32 : 3 * 32],15,10))


    def remove_existing_customers(self):
        """removes every customer that is not active any more.
        """
        j = 0
        while j < len(self.customers):
            if self.customers[j].location == 'checkout' and self.customers[j].oldlocation != 'stalled':
                MARKET3[self.customers[j].y][self.customers[j].x] = 1                
                self.customers.remove(self.customers[j])                # remove customer if reached one of the tills
            else:
                j += 1





class Ghost:

    def __init__(self, id, location, terrain_map, image, x, y):
        self.id = id
        self.location = location

        self.terrain_map = terrain_map
        self.image = image
        self.x = x
        self.y = y

    def draw(self, frame):
        xpos = OFS + self.x * TILE_SIZE
        ypos = OFS + self.y * TILE_SIZE
        frame[ypos: ypos + 32, xpos: xpos + 32] = self.image

        # overlay the Customer image / sprite onto the frame
        
    def moveonmap (self, direction):
        destination = 0
        newx = self.x
        newy = self.y
        if direction == 'u':
            newy = self.y - 1
        elif direction == 'd':
            newy = self.y + 1
        elif direction == 'r':
            newx = self.x + 1
        if direction == 'l':
            newx = self.x - 1
        if self.terrain_map.contents[newy][newx] == '.':
            self.x = newx
            self.y = newy
            destination = 1
        elif self.terrain_map.contents[newy][newx] in ['B', 'S', 'D', 'F']:
            self.x = newx
            self.y = newy
            destination = 2
        return destination
            
    def __repr__(self):
        return f'Ghost ({self.id}, {self.location})'


    def change_location(self):
        '''where does the ghost go next'''
        new_location = np.random.choice(locations, p=matrix.loc[self.location])
        self.location = new_location




if __name__ == "__main__":

    locations = ['checkout','dairy','drinks','fruit','spices']
    waitinglocation = {}
    waitinglocation['dairy'] = [[10,3],[10,4],[10,5]]
    waitinglocation['drinks'] = [[2,3],[2,4],[2,5]]
    waitinglocation['fruit'] = [[14,3],[14,4],[14,5]] 
    waitinglocation['spices'] = [[6,3],[6,4],[6,5]]
    waitinglocation['checkout'] = [[3,8],[3,9],[7,8],[7,9],[11,8],[11,9]]
    matrix = pd.read_csv('transition_matrix.csv', sep=';',index_col=['location'])    

    background = np.zeros((700, 1000, 3), np.uint8)
    tiles = cv2.imread("tiles.png")

    marketmap = SupermarketMap(MARKET, tiles)

    hmframe = np.zeros((700, 1000, 3), np.uint8)
    cv2.imshow("frame", hmframe)

    doodl = Supermarket(marketmap)

    closetime = 0
    while int(closetime) < 8 or int(closetime) > 24:
        closetime = input('At what time does the supermarket close today? (8 to 24, hours only)')
    
    df = pd.DataFrame(columns=['Time','CustomerID','Location'])


    ghostinvasion = 0
    ghost = []

    while doodl.minutes < int(closetime) * 60:
        frame = background.copy()
        
        
        key = chr(cv2.waitKey(1) & 0xFF)
        if key == "g":
            ghostinvasion = 1
            
        doodl.next_minute(frame)
        print(f'Current time is {doodl.get_time()}.')
        doodl.add_new_customers()
        data = doodl.print_customers()
        print('Moving customers...')
        moving = 1
        while moving == 1:
            doodl.marketmap.draw(frame)
            moving = 0

            for element in doodl.customers:
#                print(element.id)
                if element.goal != [0,0]:
                    moving = 1
                    MARKET3[element.y][element.x] = 1                           # clear previous location on the pathfinding map
                    start1 = (element.x, element.y)
                    goal1 = (element.goal[0], element.goal[1])
                    grid = Grid(matrix=MARKET3)
                    start = grid.node(element.x, element.y)
                    end = grid.node(element.goal[0], element.goal[1])
                    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
                    path, runs = finder.find_path(start, end, grid)                         # pathfinding algorithm
                    
                    if (path == []):         # no route found
                        if element.oldlocation == 'stalled':    # if already stalled in the previous round, give up the search
                            element.goal = [0,0]
                        element.oldlocation = 'stalled'
                        print (f"Oops! Customer {element.id} cannot find the route from {element.oldlocation} {start1} to {element.location} {goal1}!")
                    else:
                        newx = path[1][0]
                        newy = path[1][1]
                        if MARKET3[newy][newx] == 0:        # what if the next step's tile is occupied
                            if element.oldlocation == 'stalled':    # if already stalled in the previous round, give up the search
                                element.goal = [0,0]
                            element.oldlocation = 'stalled'
                        else:
                            element.x, element.y = newx, newy
                    if element.goal == [element.x, element.y]:      # reached the goal
                        element.goal = [0,0]
                        element.oldlocation = "" # empty it, just in case it was stalled
                    MARKET3[element.y][element.x] = 0                           # mark new location on the pathfinding map
                element.draw(frame)
                if ghostinvasion == 1:
                    for ghostie in ghost:
                        if np.random.rand() < .25:              # if ghost invasion is on, blink the ghosties every now and then
                            ghostie.draw(frame)
            cv2.imshow("frame", frame)
            k = cv2.waitKey(75) 
            if k == "g":
                ghostinvasion = 1

        for element in data:
            df.loc[len(df)] = element
        doodl.remove_existing_customers()

        if ghostinvasion == 1:
            dest = 0
            if np.random.rand() < .2:
                ghost.append(Ghost(str(len(ghost) + 1),'entrance',marketmap,tiles[7 * 32 : 8 * 32, 2 * 32 : 3 * 32],15,10))
            for element in ghost:
                element.draw(frame)
                rndnum = np.random.rand()                               # ghosties fly around in random directions...
                if rndnum < .25:
                    direction = 'u'
                elif rndnum < .5:
                    direction = 'd'
                elif rndnum < .75:
                    direction = 'l'
                else:
                    direction = 'r'
                dest = element.moveonmap(direction)
                if dest == 2:
                    ghost.remove(element)                           # ...until they start munching on supermarket produce (and disappear, devoured by their own hunger)

    print('The supermarket is closing. All remaining customers rush to the checkout!')
    df.to_csv('simulation.csv')

    cv2.destroyAllWindows()

    marketmap.write_image("supermarket.png")



