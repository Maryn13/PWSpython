import numpy as np
import pygame


class Target:
    width = 10
    height = 10
    targeton = True

    def __init__(self, w, h, loc):
        self.width = w
        self.height = h
        self.loc = loc

    def instantiate(self, x=None, y=None):
        if x == None:
            x = np.random.randint(0, 1920)
        if y == None:
            y = np.random.randint(0, 1080)
        self.x = x
        self.y = y
        self.targeton = True

    def destroy(self):
        self.targeton = False

    def draw(self, screen):
        if self.targeton:
            pygame.draw.polygon(screen, [0, 120, 0], [(self.x + 0.5 * self.width, self.y + 0.5 * self.height),
                                (self.x + 0.5 * self.width, self.y - 0.5 * self.height),
                                (self.x - 0.5 * self.width, self.y - 0.5 * self.height),
                                (self.x - 0.5 * self.width, self.y + 0.5 * self.height)])
    def check(self, xobj, yobj):
        if self.x + 0.5 * self.width > xobj > self.x - 0.5 * self.width and self.y + 0.5 * self.height > yobj > self.y - 0.5 * self.height:
            return True
