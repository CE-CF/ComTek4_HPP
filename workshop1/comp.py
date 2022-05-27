import parallel_msp_3 as parallel_msp_2
import workshop1
import time
import sys

if __name__ == "__main__":
    N = int(sys.argv[1])
    CPUS = 10
    ROW = N
    COL = N
    M = workshop1.matrix_Generator(N, False)

    print("Doing sequential")
    t_start = time.time()
    seqMaxSum, finalTop, finalLeft, finalBottom, finalRight = workshop1.findMaxSum(
        M, ROW, COL
    )
    t_end = time.time()

    print("(Top, Left)", "(", finalTop, finalLeft, ")")
    print("(Bottom, Right)", "(", finalBottom, finalRight, ")")
    print("Max sum is:", seqMaxSum)

    print("Time for execution: ", t_end - t_start)


    print("Doing parallel")
    t_start = time.time()
    parMaxSum, finalTop, finalLeft, finalBottom, finalRight = parallel_msp_2.findMaxSum(
        M, ROW, COL, CPUS
    )
    t_end = time.time()

    print("(Top, Left)", "(", finalTop, finalLeft, ")")
    print("(Bottom, Right)", "(", finalBottom, finalRight, ")")
    print("Max sum is:", parMaxSum)

    print("Time for parallel execution: ", t_end - t_start)


    assert parMaxSum == seqMaxSum, "[FAILED] MaxSums aren't equal"
