import numpy as np
import pygame
import math as math

class PO:

   x = 0
   #xpos
   y = 0 #y pos
   rot = 0 #radialen
   angvel = 0

   angacc = 0

   loc = np.array([500, 500])

   vel = np.array([0, 0])

   acc = np.array([0, 0])

   def __init__(self, name, width, length, weight, mps, cm, m):

       self.name = name
       self.width = width
       self.length = length
       self.weight = weight
       self.mps = mps
       self.cm = cm
       self.m = m
       self.i = 2 / 3 * self.m * width
       self.loc = np.array([np.random.randint(0, 1920), np.random.randint(0, 1080)])
       self.dt = 15/30






   def applyForce(self, force, rotForce):

       fmag = math.sqrt(force[0]*force[0]+force[1]*force[1])
       alpha = 0
       if(force[0]!=0):
           alpha = math.atan(force[1]/force[0])
       else:
           if(force[1]>0):
               alpha = 0.5*3.141592653589793
           else:
               alpha = 1.5*3.141592653589793
       if(force[0]< 0):
           facc = np.array([math.cos(alpha+3.141592653589793) * fmag / self.m, math.sin(alpha+3.141592653589793) * fmag / self.m])
       else:
           facc = np.array([math.cos(alpha)*fmag/self.m, math.sin(alpha)*fmag/self.m])

       a = self.vel[0]
       b = self.vel[1]


       mag = math.sqrt(a * a + b * b)
       fwmag = mag *mag* 0.5 * 1.293 * 0.4 * 0.025

       phi = 0
       if(self.vel[0]!= 0):
           phi = math.atan((self.vel[1]/self.vel[0]))

       if(a<0):
           fwacc = np.array([math.cos(phi+3.141592653589793)*fwmag/self.m, math.sin(phi+3.141592653589793)*fwmag/self.m])
       else:
           fwacc = np.array([math.cos(phi) * fwmag / self.m, math.sin(phi) * fwmag / self.m])
       facc = np.add(facc, -fwacc)
       self.acc = facc
       self.angacc = rotForce / self.i




   def setVelocity(self):
       self.vel = np.add(self.vel, self.acc*self.dt)
       self.angvel += self.angacc*self.dt





   def move(self):
       self.loc = np.add(self.loc, self.vel*self.dt)
       self.rot += self.angvel*self.dt

   def boundries(self, width, height):
       if self.loc[0] > width:
           self.loc[0] = width
       if self.loc[0]< 0:
           self.loc[0] = 0
       if self.loc[1] > height:
           self.loc[1] = height
       if self.loc[1]< 0:
           self.loc[1] = 0

   def draw(self, screen):
       p1 = np.array([0.5*self.width, 0.5*self.length])
       p2 = np.array([-0.5*self.width, 0.5*self.length])
       p3 = np.array([-0.5*self.width, -0.5*self.length])
       p4 = np.array([0.5*self.width, -0.5*self.length])
       modrot = np.array([[math.cos(self.rot), -math.sin(self.rot)],
                         [math.sin(self.rot), math.cos(self.rot)]])

       r = math.sqrt((0.5*self.width*0.5*self.width)+(0.5*self.width*0.5*self.width))

       pygame.draw.polygon(screen, [250, 0, 0], [tuple(np.add(self.loc, modrot.dot(p1))), tuple(np.add(self.loc, modrot.dot(p2))), tuple(np.add(self.loc, modrot.dot(p3))), tuple(np.add(self.loc, modrot.dot(p4)))])
