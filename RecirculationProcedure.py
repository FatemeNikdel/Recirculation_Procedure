# imports
import numpy as np

class Recirculation():

    """Recirculation Procedure: A learning method for networks.

    Attributes
    __________

    inputs: list
        The list of desired inputs. 

    number_of_layers: array
        The number of visible and hidden layers in network.

    units_per_layer: dict
        Each values of dictionery show the number of units (Neurons) in each group (layer).
    
    act_func: dict
        The list of activation function which were used in each layer.
    
    regression: dict
        The regression coefficient in the range of [0 1] which shows the "memory" of each layer.

    
    Returns
    _______

    error: numeric
        The squared reconstruction error
    """


    # Attributes as a constructor
    def __init__(self, inputs, units_per_layer, number_of_layers, act_func, regression):
        self.inputs = inputs
        self.units_per_layer = units_per_layer
        self.number_of_layers = number_of_layers
        self.act_func = act_func
        self.regression = regression
      
    # Method 1
    def weight_generalization(self):
        ## Step 1: initialise random weights in range [-0.5 0.5]
        list_of_weight = []
        weights = {}
        for i in range(self.number_of_layers):
            if i < self.number_of_layers -1:
                # visible to hidden layer weights
                list_of_weight.append(np.random.uniform(-0.5 , 0.5, size=(self.units_per_layer[str(i)] +1, \
                                self.units_per_layer[str(i+1)])))    # the last column considers as a bias
                weights[str(i)] = list_of_weight[i]
            else:
                # last hidden to visible  layer weights
                list_of_weight.append(np.random.uniform(-0.5 , 0.5, size=(list(self.units_per_layer.values())[-1] +1, \
                                self.units_per_layer[str(0)]) ))   # the last column considers as a bias
            weights[str(i)] = list_of_weight[-1]

        return weights
    # Method 2
    def recirculation_procedure(self, weights):
        
        col = np.array([[1],
                        [1],
                        [1],
                        [1]])  # it considers as a bias 
        y_i = {'0': np.append(self.inputs, col, axis=1)}  # inputs with bias

        ## Step 2: First Pass
        for i in range(self.number_of_layers - 1):
            # visible to hidden layers procedure
            y_i[str(i+1)] = np.append(self.act_func[str(i)](np.dot(y_i[str(i)], weights[str(i)])),  col, axis=1)

        ## Step 3: Secound Pass
        y_j = {}
        for i in range(self.number_of_layers):
            if i == 0 :
                # last hidden to visible layer procedure
                y_j[str(i)] = self.regression[str(i)] * y_i[str(i)][:,:-1] + (1-self.regression[str(i)]) * \
                    list(self.act_func.values())[-1](np.dot(list(y_i.values())[-1], list(weights.values())[-1]))
                y_j[str(i)] = np.append(y_j[str(i)],col, axis=1)        
            else: 
                # visible to hidden layers procedure
                y_j[str(i)] = self.regression[str(i)] * y_i[str(i)][:,:-1] + (1-self.regression[str(i)]) * \
                    self.act_func[str(i-1)](np.dot(y_j[str(i-1)], weights[str(i-1)]))
                y_j[str(i)] = np.append(y_j[str(i)],col, axis=1)


        ## Step 4: Update weights 
        for i in range(len(weights)):
            if i < len(weights) -1:
                # update vissible to hidden & hidden to hidden weights
                weights[str(i)] += np.dot(y_j[str(i)].T , \
                    (y_i[str(i+1)][:,:-1] - y_j[str(i+1)][:,:-1] ) )
            else:
                # update hidden to vissible weights
                list(weights.values())[-1] += np.dot(list(y_i.values())[-1].T, \
                     (y_i[str(0)][:,:-1] - y_j[str(0)][:,:-1]))

        ## Step 5: calculate error
        error = 0.5 * ((self.inputs -  y_j[str(0)][:,:-1]) ** 2).sum()

        return error
