#include <stddef.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#define MATRIX_SIZE 10000

#define RAND(min, max)                                                         \
  (int)(((double)(max - min + 1) / RAND_MAX) * rand() + min)


struct maxSum_t {
  int maxSum;
  int finalTop;
  int finalLeft;
  int finalBottom;
  int finalRight;
};

void initMatrix(int M[MATRIX_SIZE][MATRIX_SIZE]){
  srandom(time(NULL));
  for (int i = 0; i < MATRIX_SIZE; i++){
    for (int j = 0; j < MATRIX_SIZE; j++){
      M[i][j] = RAND(-128,127);
    }
  }
}

void initMatrixPtr(int *M[], size_t N){
  for (int i = 0; i < N; i++){
    M[i] = (int *) malloc(N * sizeof(int));
  }
}

void freeMatrix(int *M[], size_t N){
  for (int i = 0; i < N; i++){
    free(M[i]);
  }
}
void populateMatrix(int *M[], size_t N){
  srandom(time(NULL));
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      M[i][j] = RAND(-128,127);
    }
  }
}


static int kadane(int *arr, int *start, int *finish,int len){
  int sum, maxSum, locStart;
  sum = 0;
  maxSum = -999999999;

  *finish = -1;
  locStart = 0;

  for (int i = 0; i < len; i++){
    sum += arr[i];
    if (sum < 0){
      sum = 0;
      locStart = i + 1;
    }
    else if (sum > maxSum){
      maxSum = sum;
      *start = locStart;
      *finish = i;
    }
  }

  if (*finish != -1){
    return maxSum;
  }

  maxSum = arr[0];
  *start = 0;
  *finish = 0;

  for (int i = 1; i < len; i++){
    if (arr[i] > maxSum){
      maxSum = arr[i];
      *start = i; 
      *finish = i;
    }
  }
  return maxSum;
}

static struct maxSum_t findMaxSum(int *M[MATRIX_SIZE], int ROW, int COL){
  int maxSum = 0, finalLeft = 0, finalRight = 0, finalTop = 0, finalBottom = 0;
  int sum, start, finish;

  int temp[ROW];

  start = 0;
  finish = 0;

  for (int left = 0; left < COL; left++){
    for (int i = 0; i < ROW; i++)
      temp[i] = 0;

    for (int right = left; right < COL; right++){
      for (int i = 0; i < ROW; i++)
        temp[i] += M[i][right];

      sum = kadane(temp, &start, &finish, ROW);
      if (sum > maxSum){
        maxSum = sum;
        finalLeft = left;
        finalRight = right;
        finalTop = start;
        finalBottom = finish;
      }
    }
    printf("left: %i                   \r", MATRIX_SIZE - left);
    fflush(stdout);
  }
  struct maxSum_t res ={
    .maxSum = maxSum,
    .finalBottom = finalBottom,
    .finalLeft = finalLeft,
    .finalRight = finalRight,
    .finalTop = finalTop
  };
  return res;
}


int main(){
  FILE * fp;

  size_t N[4] = {10, 100, 1000, 10000};
  int* matrix[MATRIX_SIZE];

  fp = fopen("results_seq.csv", "w");
  if (fp == NULL){
    exit(-1);
  }


  fputs("ID,N,MaxSum,Time\n", fp);

  for (int j; j < 4; j++) {
    initMatrixPtr(matrix, N[j]);
    for (int i = 0; i < 10; i++){
    printf("Size: %li\n", N[j]);
      struct timeval tval_before, tval_after, tval_result;
      populateMatrix(matrix, N[j]);
      gettimeofday(&tval_before, NULL);
      struct maxSum_t result = findMaxSum(matrix, N[j], N[j]);
      gettimeofday(&tval_after, NULL);
      
      timersub(&tval_after, &tval_before, &tval_result);
      char *fmt = "(Top, Left): (%li,%li)\n(Bottom, Right): (%li,%li)"
      "\nMaxsum: %li\nTook: %ld.%06ld\n";
      printf(fmt, result.finalTop, result.finalLeft,
             result.finalBottom, result.finalRight,result.maxSum,
             (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
      fprintf(fp, "%i,%li,%i,%ld.%06ld\n", i, N[j], result.maxSum,
              (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
      if (tval_result.tv_sec == 0)
        sleep(1);
    }
    freeMatrix(matrix, N[j]);
  }
  fclose(fp);
}
