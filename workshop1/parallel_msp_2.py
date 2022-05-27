import numpy as np
import time
from os import cpu_count
import multiprocessing as mp


def kadane(arr, start, finish, left, right, n):

    Sum = 0
    maxSum = -999999999999
    i = None

    finish[0] = -1
    local_start = 0

    for i in range(n):
        Sum += arr[i]
        if Sum < 0:
            Sum = 0
            local_start = i + 1
        elif Sum > maxSum:
            maxSum = Sum
            start[0] = local_start
            finish[0] = i

    if finish[0] != -1:
        return (maxSum, start[0], finish[0], left, right)

    maxSum = arr[0]
    start[0] = finish[0] = 0

    for i in range(1, n):
        if arr[i] > maxSum:
            maxSum = arr[i]
            start[0] = finish[0] = i
    return (maxSum, start[0], finish[0], left, right)


def _loopy_boi(M, ROW, COL, left, pool):

    maxSum, finalLeft = -999999999999, None
    finalRight, finalTop, finalBottom = None, None, None
    right, i = None, None

    start = [0]
    finish = [0]

    temp = [0] * ROW

    res_list = []
    for right in range(left, COL):
        for i in range(ROW):
            temp[i] += M[i][right]
            
        res = pool.apply_async(
            kadane,
            (
                temp,
                start,
                finish,
                left,
                right,
                ROW,
            ),
        )

        res_list.append(res.get())

    # Sum = kadane(temp, start, finish, ROW)

    for x in res_list:
        if x[0] > maxSum:
            maxSum = x[0]
            finalLeft = x[3]
            finalRight = x[4]
            finalTop = x[1]
            finalBottom = x[2]

    return (maxSum, finalTop, finalLeft, finalBottom, finalRight)


def findMaxSum(M, ROW, COL, processors=cpu_count()):

    assert processors != 0, "Can't use 0 processors"
    assert processors <= cpu_count(), "Too many processors"

    maxSum, finalLeft = -999999999999, None
    finalRight, finalTop, finalBottom = None, None, None
    left, right, i = None, None, None

    temp = [None] * ROW
    start = [0]
    finish = [0]

    res_list = []

    with mp.Pool(processors) as pool:
        for left in range(COL):
            res_list.append(_loopy_boi(M, ROW, COL, left, pool))

    res = None

    for x in res_list:
        if x[0] > maxSum:
            res = x
            maxSum = x[0]

    print(res)

    return res


def matrix_Generator(N, out=True):
    print("The dimension : " + str(N))  # printing dimension

    res = [
        list(np.random.randint(-1, 2, N)) for i in range(N)
    ]  # using list comprehension - matrix creation of n * n

    if out:
        print("The created matrix of N * N: \n")  # print result
        column = ""
        divider = ""
        for x in range(N):
            if x == 0:
                column = "       {:4d}".format(x)
                divider = "     ------"
            else:
                column = column + "{:4d}".format(x)
                divider = divider + "----"
        print(column)
        print(divider)

        for x in range(len(res)):
            row = ""
            for y in range(len(res[x])):
                if y == 0:
                    row = row + "{:4d} | {:4d}".format(x, res[x][y])
                else:
                    row = row + "{:4d}".format(res[x][y])
            print(row)
        print("")
    return res


# Driver Code
if __name__ == "__main__":
    # N * N matrix
    N = 10 ** 2 * 5
    ROW = N
    COL = N
    M = matrix_Generator(N, False)

    t_start = time.time()
    maxSum, finalTop, finalLeft, finalBottom, finalRight = findMaxSum(M, N, N)
    t_end = time.time()

    print("(Top, Left)", "(", finalTop, finalLeft, ")")
    print("(Bottom, Right)", "(", finalBottom, finalRight, ")")
    print("Max sum is:", maxSum)

    print("Time for parallel execution: ", t_end - t_start)
