""" Recirculation Procedure: it discovers codes that allow the activity vectors in a "visible" group
 to be represented by activity vectors in a "hidden" group.

Author: Fateme Nikdelfaz <fateme.nkdl@gmail.com>

Created: 16th January 2023
"""


## imports
from RecirculationProcedure import Recirculation
import numpy as np


## Variables 
number_of_layers = np.array(int(input("Please Enter the number of layers: ")))
units_per_layer = {}
regression = {}
for i in range(number_of_layers):
    units_per_layer[str(i)] = int(input(f"Please Enter the neurons' number of layer {i} : "))
    if i == number_of_layers -1:
        regression[str(i)] = 0
    else:
        regression[str(i)] = 0.75
EPOCHS = 100
inputs = [
        [1, 0, 0, 0],
        [0, 1, 0, 0], 
        [0, 0, 1, 0], 
        [0, 0, 0, 1]
        ]

## Activation Functions 
def logistic(x):
    # it takes any real value as input and outputs values in the range of [0 1]
    return 1 / (1 + np.exp(-x))

def relu(x):
    # it outputs the input directly if it is positive, otherwise, it will output zero.
    return np.maximum(0, x)  

act_func = {}
for i in range(number_of_layers):
    if i == 0 :
        act_func[str(i)] = logistic
    else:
        act_func[str(i)] = relu

## Instance of RecirculationProcedure's Class
rp = Recirculation(inputs, units_per_layer, number_of_layers, act_func, regression)
weights = rp.weight_generalization()
error = np.empty(EPOCHS)
for i in range(EPOCHS):
    error[i] = rp.recirculation_procedure(weights)

print(f"Error in 100th epoch is: {error[-1]}")
