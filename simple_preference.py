
import numpy as np
from cmaes import CMA
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import math
import matplotlib.pyplot as plt


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

def bernoulli(prob):
	tfd = tfp.distributions
	b = tfd.Bernoulli(probs = prob)
	return b.sample(1).numpy()

def psychometric(x1,x2,alpha,beta,gamma,lambd):
	#x1 is reference (previous preference)
	percent_change=(1 - x2/x1)*100

	prob = gamma + (1 - lambd-gamma)/(1+np.exp( -beta * (percent_change-alpha)))
	JND = (1./(2 * beta))*math.log(((0.75 - gamma) * (1 - lambd - 0.25))/((1 - lambd - 0.75) * (0.25 - gamma)))
	ReferenceStim = alpha
	k = (1./(2)) * math.log((   (0.75 - gamma) * (1 - lambd - 0.25))/((1 - lambd - 0.75) * (0.25 - gamma)))
	weber_fraction = JND/ReferenceStim
	W = k/(alpha * beta)

	return prob, JND, ReferenceStim, weber_fraction, W

alpha_ = 10
beta_ = 1
gamma_ = 0
lambd_ = 0.02

if __name__ == "__main__":
    optimizer = CMA(mean=np.zeros(2), sigma=1.3)
    # solutions_prev = [1,2,3,4,5,6]
    best_fit_prev = 10000000

    for generation in range(50):
        solutions = []
        fitnesses = []
        
        for _ in range(optimizer.population_size):

        	x = optimizer.ask()
        	value = quadratic(x[0], x[1])
        	fitnesses.append(value)

        	solutions.append((x, value))
        	print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        	x_prev = x

        fitnesses.sort()
        best_fit = fitnesses[0]

        prob_, _, _, _, _ = psychometric(best_fit_prev, best_fit, alpha_, beta_, gamma_, lambd_)
        binary = bernoulli(prob_).item()

        if binary == 1 or generation ==0:
        	solutions_update = solutions 		
        else:
        	solutions_update = solutions_prev
        	print("not moved")


        solutions_prev = solutions
        best_fit_prev = best_fit

        optimizer.tell(solutions_update)


    x2 = np.linspace(20,0,100)
    x1 = 10 
    percent_change=(1 - x2/x1)*100
    y = []

    for i in x2:
    	prob_plot, _, _, _, _ =psychometric(x1, i, alpha_, beta_, gamma_, lambd_)
    	y.append(prob_plot)
    plt.ylim(0, 1)
    plt.xlim(-100,100)
    plt.plot(percent_change,y ) 
    plt.show()
    #plot fitness vs epoch
    #plot no psychometric vs original