# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:28:09 2021

@author: dinhm
"""


import random as rd
import numpy as np
import matplotlib.pyplot as plt
def fairDiceA(throw: int) -> [int]:
    '''
   Roll the faire Dice A several time 
    
    Args:
        throw     -- Number of throws
    Returns:
        Result -- List of dice result.
    '''

    n = np.random.randint(low = 1,high = 7, size = throw)
    return list(n)
    
def unfairDiceB(throw: int) -> [int]:
    '''
   Roll the faire Dice B several time 
    
    Args:
        throw     -- Number of throws
    Returns:
        Result -- List of dice result.
    '''
    
    result = []
    for i in range(0, throw):
        r = np.random.random()
        n = np.random.randint(1,4)
        if r >= 1/4 :
            result.append(2*n)
        else:
            result.append(2*n - 1)
    return result
print(fairDiceA(10))
print(unfairDiceB(10))




def bayesTheorem(
        throws: int, 
        probabilityA: [float],
        probabilityB: [float],
        prior: (float, float)
    ) -> (float, float):
    '''
    Calculate the likelihood of a throw from the sequence
    comes from dice A or dice B.
    
    Args:
        throws                -- the current value of the dice in the sequence
        probabilityA          -- the probabilities for the sides of the dice A
        proabilityB           -- the probabilities for the sides of the dice B
        prior                 -- the previous probability for dice A and dice B as tuples: (pA, pB)
    Returns:
        posterior             -- the probability for dice A and dice B as tuples after observing the current throw: (pA, pB)
    '''
  
   
    # probabilityA = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    # probabilityB = [1/12, 3/12, 1/12, 3/12, 1/12, 3/12]
    if( (throws-1) >= len(probabilityA)):
        return "Error"
    pA = ( probabilityA[throws-1] * prior[0] ) / \
            ( probabilityA[throws-1] * prior[0] + probabilityB[throws-1] * prior[1]  )
    pB = 1 - pA
    return (pA, pB)
    
bayesTheorem(6, [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [1/12, 3/12, 1/12, 3/12, 1/12, 3/12], (0.5, 0.5)) 




# At the beginning we assume that both dice are equally likely.
prior = (0.5, 0.5)
# Sequence of dice results
sequence = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
           4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
           5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

# Probabilities of the dice results
probabilityA = [] 
probabilityB = []


for i in range(len(sequence)):
    nextPrior = bayesTheorem(sequence[i], \
                             [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], \
                             [1/12, 3/12, 1/12, 3/12, 1/12, 3/12],\
                             prior)
    prior = nextPrior
    probabilityA.append(prior[0])
    probabilityB.append(prior[1])

plt.plot(probabilityA, linewidth=3, label="Dice A")
plt.plot(probabilityB, linewidth=3, label="Dice B")
plt.legend(["Dice A", "Dice B"])
plt.grid()
plt.xlabel('Sequence')
plt.ylabel('Probability')
plt.title("Visualization of two dices by Bayes")

plt.show()


