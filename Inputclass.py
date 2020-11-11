import numpy as np
import math as math

class Input:

   def __init__(self, minf, maxf):
       self.minf = minf
       self.maxf = maxf

   def outputF (self, p1, p2, f1, f2, r1, r2):
       mag1 = math.sqrt(p1[0]*p1[0]+p1[1]*p1[1])
       mag2 = math.sqrt(p2[0]*p2[0]+p2[1]*p2[1])

   def calculateforward(self, rot, f):
       outv = np.array([math.cos(rot)*f, math.sin(rot)*f])
       return outv
