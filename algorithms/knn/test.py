import numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from multiprocess.pool import Pool, ThreadPool
from multiprocess import get_context
from time import time



def multithreaded_gamma(x, l, u, number_of_steps, num_thread):
    st, rg = time(), (u - l) / num_thread
    ls = [(x, l + i * rg, min(u, l + i * rg + rg), number_of_steps // num_thread) for i in range(num_thread)]
    with ThreadPool(num_thread) as p:
        res = p.starmap(calculate_gamma, ls)
    print('Took {}s'.format(time() - st))
    return sum(res)

def calculate_gamma(x, bound_1, bound_2, number_of_steps):
    X = [i * (bound_2 - bound_1) / number_of_steps + bound_1 for i in range(number_of_steps)]
    res = sum(pow(i, x - 1) * 2.71828 ** -i for i in X) * ((bound_2 - bound_1) / number_of_steps)
    return res




def multiprocess_gamma(x, l, u, number_of_steps, num_process):
    st, rg = time(), (u - l) / num_process
    ls =  [(x, l + i * rg, min(u, l + i * rg + rg), number_of_steps // num_process) for i in range(num_process)]
    with Pool(num_process) as p:
        res = p.starmap(calculate_gamma, ls)
    print('Took {}s'.format(time() - st))
    return sum(res)

if __name__ == "__main__":
    result = multiprocess_gamma(6, 0, 1000, 10_000_000, 2)
    print(result)