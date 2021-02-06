from BatOptimization import BatOptimization
import math

def fitnessFunc(x,method):
    if method == "ackley":
       return -math.exp(-math.sqrt(0.5*sum([i**2 for i in x]))) - math.exp(0.5*sum([math.cos(i) for i in x])) + 1 + math.exp(1)
    elif method == "beale":
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
    elif method == "goldstein":
        return ((1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * (x[0] ** 2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * (x[1] ** 2))) * (30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * (x[0] ** 2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1] ** 2))))
    elif method == "levi":
        return (math.sin(3*x[0]*math.pi))**2 + ((x[0] - 1)**2)*(1 + math.sin(3*x[1]*math.pi)**2) + ((x[1] - 1)**2)*(1 + math.sin(2*x[1]*math.pi)**2)

#ackley    =========> (0,0) = 0
#beale     =========> (3,0.5) = 0
#goldstein =========> (0,-1) = 3
#levi      =========> (1,1) = 0

#Parameters => dimension, populationSize, generation, startingPulse, alfa, gamma, fmin, fmax, batLowerBound, batUpperBound,method
Bat = BatOptimization(2,20,50,0.1,0.5,0.5,0,1,-5,5,fitnessFunc,"levi")
Bat.fly()