def summation(ip, w, bias):
    result = bias
    for i in range(len(inputs)):
        result = result + ip[i] * w[i]
    return result
    
def step(x): 
    if x >= 0: 
        return 1 
    else: 
        return 0

def bipolar_step(x): 
    if x >= 0: 
        return 1 
    else: 
        return -1

def sigmoid(x): 
    return 1 / (1 + math.exp(-x))

def tanh(x): 
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def relu(x):
    if x > 0: 
        return x 
    else: 
        return 0

def leaky_relu(x): 
    if x > 0:
        return x 
    else: 
        return 0.01 * x

def error(act, pred): 
    return act - pred
