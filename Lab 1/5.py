import random
from statistics import mean, median, multimode
def statistic(n):
    avg = mean(n)
    med = median(n)
    mode = multimode(n)
    print("Mean:", avg)
    print("Median:", med)
    print("Mode:", mode)

n = [random.randint(100, 150) for i in range(100)]
print("Random Numbers:", n)
statistic(n)