import random
from statistics import mean, median, multimode

n = [random.randint(100, 150) for i in range(100)]
print("Random Numbers:",n)

avg = mean(n)
med = median(n)
mode = multimode(n)
print("mean:", avg)
print("median:", med)
print("mode:", mode)