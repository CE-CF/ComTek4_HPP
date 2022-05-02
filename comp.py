import parallel_msp_2
import workshop1
import time

if __name__ == "__main__":
    N = 1000
    CPUS = 4
    ROW = N
    COL = N
    M = workshop1.matrix_Generator(N, False)

    t_start = time.time()
    parMaxSum, finalTop, finalLeft, finalBottom, finalRight = parallel_msp_2.findMaxSum(
        M, ROW, COL, CPUS
    )
    t_end = time.time()

    print("(Top, Left)", "(", finalTop, finalLeft, ")")
    print("(Bottom, Right)", "(", finalBottom, finalRight, ")")
    print("Max sum is:", parMaxSum)

    print("Time for parallel execution: ", t_end - t_start)

    t_start = time.time()
    seqMaxSum, finalTop, finalLeft, finalBottom, finalRight = workshop1.findMaxSum(
        M, ROW, COL
    )
    t_end = time.time()

    print("(Top, Left)", "(", finalTop, finalLeft, ")")
    print("(Bottom, Right)", "(", finalBottom, finalRight, ")")
    print("Max sum is:", seqMaxSum)

    print("Time for execution: ", t_end - t_start)


    assert parMaxSum == seqMaxSum, "[FAILED] MaxSums aren't equal"
