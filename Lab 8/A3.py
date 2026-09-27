import math
# activation functions
def bipolar(input_value):
    if input_value >= 0:
        return 1
    else:
        return -1
def sigmoid(input_value):
    return 1 / (1 + math.exp(-input_value))
def relu(input_value):
    if input_value > 0:
        return input_value
    else:
        return 0
# summation unit
def sum(input_a, input_b, bias, weight_a, weight_b):
    return bias + weight_a*input_a + weight_b*input_b
# calculate error
def calc_err(target, output):
    return target - output
# update weights and bias
def upd(bias, weight_a, weight_b, input_a, input_b, error, learning_rate):
    bias = bias + learning_rate*error
    weight_a = weight_a + learning_rate*error*input_a
    weight_b = weight_b + learning_rate*error*input_b
    return bias, weight_a, weight_b
# calculate SSE
def sse_calc(data, target, bias, weight_a, weight_b, activation):
    sse = 0
    for i in range(len(data)):
        input_a, input_b = data[i]
        net = sum(input_a, input_b, bias, weight_a, weight_b)
        output = activation(net)
        error = calc_err(target[i], output)
        sse += error**2
    return sse
# train the perceptron
def train(data, target, activation, bias, weight_a, weight_b, learning_rate):
    for epoch in range(1, 1001):
        for i in range(len(data)):
            input_a, input_b = data[i]
            net = sum(input_a, input_b, bias, weight_a, weight_b)
            output = activation(net)
            error = calc_err(target[i], output)
            bias, weight_a, weight_b = upd(bias, weight_a, weight_b, input_a, input_b, error, learning_rate)
        # check convergence
        sse = sse_calc(data, target, bias, weight_a, weight_b, activation)
        if sse <= 0.002:
            break
    return epoch, bias, weight_a, weight_b

# set AND gate inputs
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# set target values
target = [0, 0, 0, 1]
# set bipolar target values
bipolar_target = [-1, -1, -1, 1]
# set initial weights
initial_bias = 10
initial_weight_a = 0.2
initial_weight_b = -0.75
# set learning rate
learning_rate = 0.05
# train with bipolar step
epoch_bipolar, final_bias_bipolar, final_weight_a_bipolar, final_weight_b_bipolar = train(data, bipolar_target, bipolar, initial_bias, initial_weight_a, initial_weight_b, learning_rate)
# train with sigmoid
epoch_sigmoid, final_bias_sigmoid, final_weight_a_sigmoid, final_weight_b_sigmoid = train(data, target, sigmoid, initial_bias, initial_weight_a, initial_weight_b, learning_rate)
# train with relu
epoch_relu, final_bias_relu, final_weight_a_relu, final_weight_b_relu = train(data, target, relu, initial_bias, initial_weight_a, initial_weight_b, learning_rate)
print("iterations / epochs to converge")
print("bipolar step :", epoch_bipolar)
print("sigmoid      :", epoch_sigmoid)
print("relu         :", epoch_relu)
print("\nfinal weights")
print("bipolar step:")
print(final_bias_bipolar, final_weight_a_bipolar, final_weight_b_bipolar)
print("\nsigmoid:")
print(final_bias_sigmoid, final_weight_a_sigmoid, final_weight_b_sigmoid)
print("\nrelu:")
print(final_bias_relu, final_weight_a_relu, final_weight_b_relu)