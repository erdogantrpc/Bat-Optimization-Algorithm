import numpy as np
import math
import matplotlib.pyplot as plt

class BatOptimization():
    def __init__(self, dimension, populationSize, generation, startingPulse, alfa, gamma, fmin, fmax, batLowerBound, batUpperBound, fitnessFunc, method):
        #Assigning parameters to variables to use
        self.method = method
        self.dimension = dimension
        self.populationSize = populationSize
        self.generation = generation
        self.alfa = alfa
        self.gamma = gamma
        self.fmin = fmin
        self.fmax = fmax
        self.batLowerBound = batLowerBound
        self.batUpperBound = batUpperBound
        self.fitnessFunc = fitnessFunc
        self.startingPulse = startingPulse

        #Set arrays to use to show results
        self.xArray = []
        self.yArray = []
        self.minFitnessArray = []

        #Set the initial values ​​of loudness value and pulse rate
        self.loudness = [0.95 for i in range(self.populationSize)]
        self.heartRate = [self.startingPulse for i in range(self.populationSize)]
        
        #Set first upper and lower limits for each bat
        self.upperBound = [[0.0 for i in range(self.dimension)] for j in range(self.populationSize)]
        self.lowerBound =  [[0.0 for i in range(self.dimension)] for j in range(self.populationSize)]
        
        #Set initial initial frequency value for all bats
        self.frequency = [0.0] * populationSize
        
        #Set initial velocity for all bats
        self.velocity = [[0.0 for i in range(self.dimension)] for j in range(self.populationSize)]
        
        #Set the initial position for all bats
        self.position = [[0.0 for i in range(self.dimension)] for j in range(self.populationSize)]
        
        #Set initial fitness values ​​for all bats
        self.fitnessValue = [0.0] * populationSize
        self.fitnessValue_minimum = 0.0
        
        #Set the first best location
        self.bestPosition = [0.0] * dimension
    
    def settings(self):
        #Set all upper and lower limits according to given parameters
        for i in range(self.populationSize):
            for j in range(self.dimension):
                self.lowerBound[i][j] = self.batLowerBound
                self.upperBound[i][j] = self.batUpperBound
        #Generate a new solution according to the lower and upper limits and set the frequency of all bats to 0
        for i in range(self.populationSize):
            self.frequency[i] = 0
            for j in range(self.dimension):
                #Update locations within the lower and upper limit
                self.position[i][j] = self.lowerBound[i][j] + (self.upperBound[i][j] - self.lowerBound[i][j]) * np.random.uniform(0,1)
            #Calculate the initial fitness value
            self.fitnessValue[i] = self.fitnessFunc(self.position[i],self.method)
        #Find the bat in the best position
        self.bestPositionBat()

    def bestPositionBat(self):
        i = 0
        j = 0
        #find the best fit value and keep it in variable j
        for i in range(self.populationSize):
            if self.fitnessValue[i] < self.fitnessValue[j] :
                j = i    
        #Save all locations with the best fitness value (j) in Top Location
        for i in range(self.dimension):
            self.bestPosition[i] = self.position[j][i]
        #Keep the best fitness value
        self.fitnessValue_minimum = self.fitnessValue[j]
    
    def setBounds(self, value):
        #If value is greater than upper limit, set new upper limit
        if(value > self.batUpperBound):
            value = self.batUpperBound
        #If value is less than lower bound, set new lower bound
        if(value < self.batLowerBound):
            value = self.batLowerBound
        return value
    
    def fly(self):
        #Generate initial matrix for solution
        newPosition = [[0.0 for i in range(self.dimension)] for j in range(self.populationSize)]
        self.settings()

        for t in range(self.generation):
            for i in range(self.populationSize):
                #Calculate the frequencies of bats
                self.frequency[i] = self.fmin + (self.fmax-self.fmin)*np.random.uniform(0,1)
                for j in range(self.dimension):
                    #Set the speed of bats
                    self.velocity[i][j] = self.velocity[i][j] + (self.position[i][j] - self.bestPosition[j])*self.frequency[i]
                    #Set the position of bats
                    newPosition[i][j] = self.position[i][j] + self.velocity[i][j]
                    newPosition[i][j] = self.setBounds(newPosition[i][j])
                #If the random value [0.1] is greater than the bat's pulse rate, choose a solution among the best solutions.
                if(np.random.uniform(0,1) > self.heartRate[i]):
                    for j in range(self.dimension):
                        newPosition[i][j] = self.bestPosition[j] + np.random.uniform(-1,1)*np.mean(self.loudness)
                        newPosition[i][j] = self.setBounds(newPosition[i][j])
                #Calculate the fitness value of the new solution
                newSolutionFitness = self.fitnessFunc(newPosition[i],self.method)
                if(np.random.uniform(0,1) < self.loudness[i] and newSolutionFitness < self.fitnessValue[i]):
                    self.fitnessValue[i] = newSolutionFitness
                    for j in range(self.dimension):
                        self.position[i][j] = newPosition[i][j]
                if(self.fitnessValue[i] < self.fitnessValue_minimum):
                    #Update best solution
                    self.fitnessValue_minimum = self.fitnessFunc(newPosition[i],self.method)
                    for j in range(self.dimension):
                        self.bestPosition[j] = self.position[i][j] 
                    #Update loudness and pulse rate for each bat  
                    self.loudness[i] = self.loudness[i]*self.alfa
                    self.heartRate[i] = self.startingPulse*(1 - math.exp(-1*self.gamma*i))
            #Assign best position values ​​for chart to array
            x = self.bestPosition[0]
            y = self.bestPosition[1]
            self.xArray.append(x)
            self.yArray.append(y)
            self.minFitnessArray.append(self.fitnessValue_minimum)
            #Write the best fit value found for each iteration to the console
            print(f"{t}. iteration best fitness value: ",self.fitnessValue_minimum)
        #When all transactions are done, write the best location and minimum convenience value to the console
        print("Best Solution: ",self.bestPosition)
        print("Best Fitness Value: ",self.fitnessValue_minimum)
        #plot values ​​on graph
        fig = fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(self.minFitnessArray,'r.-')
        ax1.legend(['MinFitness'])
        ax2 = fig.add_subplot(212)
        ax2.plot(self.xArray,'b.-')
        ax2.plot(self.yArray,'g--')
        plt.legend(['x1','x2'])
        plt.show()