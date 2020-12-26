import pygame
import random
import sys
import numpy as np

pygame.init()
font = pygame.font.Font('freesansbold.ttf', 32)

Screen_Height = 500
Screen_Width = 500
win = pygame.display.set_mode((Screen_Height, Screen_Width))
pygame.display.set_caption("AMO-lution")
clock = pygame.time.Clock()
gameOver = False
collide = 0
#================================Car Class===========================================#\start
class car():
    def __init__(self):
        self.car_img = pygame.transform.scale(pygame.image.load('car.png') , (40 , 40))
        self.car_img = pygame.transform.flip(self.car_img , 0 , 1)
        self.velo = 10
        self.x = 250
        self.y = 250
        self.height = 20
        self.width = 20
        self.pose = 0
        self.score = 0

        self.text = font.render(str(self.score), True, (0,255,0), (255,255,255))
        self.textRect = self.text.get_rect()
        self.textRect.center = (Screen_Width - 120, 50)

    def move(self , left , right):

        if left == 1:
            self.pose = 0
        if right == 1:
            self.pose = 1

        if self.pose == 0:
            self.x =  100

        if self.pose == 1:
            self.x = Screen_Width - 100

    def draw(self):
        global win
        rect = pygame.Rect(self.x , self.y , self.width , self.height)

        self.text = font.render("Gen: " + str(self.score), True, (0,255,0), (255,255,255))
        win.blit(self.car_img , (self.x , self.y))
        win.blit(self.text, self.textRect)
#================================Car Class===========================================#\end

#================================Collision Class=====================================#\start
class log():
    def __init__(self):
        self.pose = random.randint(0,1)
        self.y = 0
        self.x = 0
        self.log_img = pygame.transform.scale(pygame.image.load('log.png') ,(200,50))
        self.speed = 5
        self.count = 0


    def draw(self,win1):
        self.win = win1
        if self.pose == 0:
            self.x = 50
        elif self.pose == 1:
            self.x = Screen_Width - 50 + -200
        win.blit(self.log_img , (self.x , self.y))

        self.y += 1 * self.speed

    def update(self,win2, car2):
        self.draw(win2)
        if self.y > car2.y -50 and self.y < car2.y + 50:
            if self.pose == car2.pose:
                print("collide")
                global collide
                collide = 1
                #sys.exit()
            else:
                if self.count == 0:
                    self.count = 1

        if self.y > Screen_Height:
            self.y = -50
            self.pose = random.randint(0,1)
            self.count = 0
            self.speed += random.randint(-100,200) / 100
            print(self.speed)
#================================Collision Class=====================================#\end
    pygame.draw.rect(win , (0,0,0) , pygame.Rect((Screen_Width / 2) , 0 , 10 , Screen_Height))


synaptic_weights = 2 * np.random.random((2,1)) - 1
print('Random Starting Synaptic Weights:')
print(synaptic_weights)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1 - x)

def train(input_layer,synaptic_weights , training_outputs):
    outputs = sigmoid(np.dot(input_layer,synaptic_weights))
    error = training_outputs - outputs
    adjustments = error * sigmoid_der(outputs)
    synaptic_weights += np.dot(input_layer.T , adjustments)

def think(inputs):
    inputs = inputs.astype(float)
    output = sigmoid(np.dot(inputs, synaptic_weights))
    return output

log1 = log()
car1 = car()
car1.score = 1

while not gameOver:
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            gameOver = True
            pygame.quit()
            sys.exit()

        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
        #     right , left = keys[pygame.K_RIGHT],keys[pygame.K_LEFT]
        #     car1.move(left , right)

    win.fill((255,255,255))
    car1.draw()

    log1.update(win, car1)
    pygame.display.update()
    if log1.pose == 0:
        training_input = [1,0]
    else:
        training_input = [0,1]

    output_data = think(np.array(training_input))
    #print(str(output_data))

    if output_data < 0.5:
        move_input_left = 1
        move_input_right = 0

    else:
        move_input_left = 0
        move_input_right = 1

    car1.move(move_input_left , move_input_right)

    if collide == 1:
        rnd_adjustment = 2 * np.random.random((2,1)) - 1
        synaptic_weights += rnd_adjustment
        log1.y = 0
        collide = 0
        car1.score += 1
