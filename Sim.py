import pygame
from physicsobject import PO
from inputclass import Input
from target import Target
import numpy as np
pygame.init()
clock = pygame.time.Clock()

screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
xf = 0
yf = 0
rf = 0
oh = 50
ow = 50
force_right = 0
force_left = 0
mass = 1
r = .1
inertia = mass * r * r
force_front = 0
force_rot = 0
running = True
input = Input(1,1)
object = PO("rect", 100, 100,1,1,100,1)
target = Target(100,100,[100,100])
pygame.display.set_caption("TheProgram")
target.instantiate(100,100)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                force_right = .2
            if event.key == pygame.K_d:
                force_right = .1
            if event.key == pygame.K_c:
                force_right = -.1
            if event.key == pygame.K_q:
                force_left = .2
            if event.key == pygame.K_a:
                force_left = .1
            if event.key == pygame.K_z:
                force_left = -.1
        else:
            force_right = 0
            force_left = 0
    screen.fill((200, 200, 200))
    if (force_left >= 0 and force_right >= 0) or (force_left < 0 and force_right < 0):
        force_front = force_left + force_right - abs(force_left - force_right)
    elif (force_left < 0 and force_right == 0) or (force_right < 0 and force_left == 0):
        force_front = 0
    else:
        force_front = force_left + force_right #+ abs(force_left - force_right)

    force_rot = force_left - force_right
    object.setVelocity()
    object.applyForce(input.calculateforward(object.rot, force_front), force_rot)
    object.move()
    object.boundries(1920, 1080)
    target.draw(screen)
    object.draw(screen)
    clock.tick(60)
    pygame.display.update()


pygame.quit()
