import numpy as np
import time

def kadane(arr, start, finish, n):

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
        return maxSum

    maxSum = arr[0]
    start[0] = finish[0] = 0

    for i in range(1, n):
        if arr[i] > maxSum:
            maxSum = arr[i]
            start[0] = finish[0] = i
    return maxSum


def findMaxSum(M, ROW, COL):

    maxSum, finalLeft = -999999999999, None
    finalRight, finalTop, finalBottom = None, None, None
    left, right, i = None, None, None

    temp = [None] * ROW
    Sum = 0
    start = [0]
    finish = [0]

    for left in range(COL):

        temp = [0] * ROW

        for right in range(left, COL):
            for i in range(ROW):
                temp[i] += M[i][right]

            Sum = kadane(temp, start, finish, ROW)

            if Sum > maxSum:
                maxSum = Sum
                finalLeft = left
                finalRight = right
                finalTop = start[0]
                finalBottom = finish[0]

    return (maxSum, finalTop, finalLeft, finalBottom, finalRight)


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

    N = 10 ** 4  # N * N matrix
    ROW = N
    COL = N
    M = matrix_Generator(N, False)

    t_start = time.time()
    maxSum, finalTop, finalLeft, finalBottom, finalRight = findMaxSum(M)
    t_end = time.time()

    print("(Top, Left)", "(", finalTop, finalLeft, ")")
    print("(Bottom, Right)", "(", finalBottom, finalRight, ")")
    print("Max sum is:", maxSum)

    print("Time for sequential execution: ", t_end - t_start)
