import numpy as nump
import pandas as exe
import matplotlib.pyplot as plot
import time

file = "Lab_Session_Data.xlsx"
def mean(arr):
    total = 0
    for x in arr:
        total += x
    return total / len(arr)

def var(arr):
    m = mean(arr)
    total = 0
    for x in arr:
        total += (x - m) ** 2
    return total / len(arr)

def mean_variance(file):
    irctc = exe.read_excel(file, sheet_name='IRCTC Stock Price')
    price = irctc["Price"].values
    print("numpy mean:", nump.mean(price), "numpy var:", nump.var(price))
    print("own mean:", mean(price), "own var:", var(price))
    return irctc

def compare(irctc, runs):
    price = irctc["Price"].values
    t0 = time.perf_counter()
    for _ in range(runs):
        nump.mean(price)
    t_mean = (time.perf_counter() - t0) / runs

    t0 = time.perf_counter()
    for _ in range(runs):
        mean(price)
    own_mean = (time.perf_counter() - t0) / runs

    t0 = time.perf_counter()
    for _ in range(runs):
        nump.var(price)
    t_var = (time.perf_counter() - t0) / runs

    t0 = time.perf_counter()
    for _ in range(runs):
        var(price)
    own_var = (time.perf_counter() - t0) / runs
    print("numpy mean time:", t_mean, "own mean time:", own_mean)
    print("numpy var time:", t_var, "own var time:", own_var)

def wednesday_mean(irctc):
    wed_price = irctc.loc[irctc["Day"] == "Wed", "Price"]
    print("population mean:", irctc["Price"].mean())
    print("wednesday mean:", wed_price.mean())
    return wed_price.mean()

def april_mean(irctc):
    apr_price = irctc.loc[irctc["Month"] == "Apr", "Price"]
    print("population mean:", irctc["Price"].mean())
    print("april mean:", apr_price.mean())
    return apr_price.mean()

def loss(irctc):
    p_loss = (irctc["Chg%"].apply(lambda x: x < 0)).mean()
    print("P(loss):", p_loss)
    return p_loss

def wednesday_profit(irctc):
    profit = irctc["Chg%"].apply(lambda x: x > 0)
    wed = irctc["Day"] == "Wed"
    profit_wed = profit[wed].mean()
    print("P(profit on Wednesday):", profit_wed)
    return profit_wed

def conditional_profit(irctc):
    profit = irctc["Chg%"].apply(lambda x: x > 0)
    wed = irctc["Day"] == "Wed"
    p_profit_and_wed = (profit & wed).mean()
    p_wed = wed.mean()
    p_profit_given_wed = p_profit_and_wed / p_wed
    print("P(Profit | Wednesday):", p_profit_given_wed)
    return p_profit_given_wed

def scatter_plot(irctc):
    plot.scatter(irctc["Day"], irctc["Chg%"])
    plot.xlabel("Day")
    plot.ylabel("Chg%")
    plot.title("Chg% vs Day")
    plot.show()

irctc = mean_variance(file)
compare(irctc,10)
wednesday_mean(irctc)
april_mean(irctc)
loss(irctc)
wednesday_profit(irctc)
conditional_profit(irctc)
scatter_plot(irctc)