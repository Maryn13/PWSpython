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

    def instantiate(self, x, y):
        self.x = self.loc[0]
        self.y = self.loc[1]
        self.targeton = True

    def destroy(self):
        self.targeton = False

    def draw(self, screen):
        if self.targeton:
            pygame.draw.polygon(screen, [0, 120, 0], [(self.x + 0.5 * self.width, self.y + 0.5 * self.height),
                                (self.x + 0.5 * self.width, self.y - 0.5 * self.height),
                                (self.x - 0.5 * self.width, self.y - 0.5 * self.height),
                                (self.x - 0.5 * self.width, self.y + 0.5 * self.height)])
