#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#define MATRIX_SIZE 4

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

void initMatrixPtr(int **M){
  puts("Alloc M");
  M = (int **) malloc(MATRIX_SIZE * sizeof(int*));
  if (M == NULL)
    exit(-1);

  for (int i = 0; i < MATRIX_SIZE; i++){
    printf("Alloc M[%i]\n", i);
    M[i] = (int *) malloc(MATRIX_SIZE * sizeof(int));
  }

  srandom(time(NULL));
  for (int i = 0; i < MATRIX_SIZE; i++){
    for (int j = 0; j < MATRIX_SIZE; j++){
      printf("filling M[%i][%i]\n", i,j);
      M[i][j] = RAND(-128,127);
    }
  }
  puts("Done init");
}

void freeMatrix(int **M){
  puts("Freeing");
  for (int i = 0; i < MATRIX_SIZE; i++){
    printf("Freeing M[%i]\n", i);
    free(M[i]);
  }

  puts("Freeing M");
  free(M);
}

int kadane(int *arr, int *start, int *finish,int len){
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

struct maxSum_t findMaxSum(int M[MATRIX_SIZE][MATRIX_SIZE], int ROW, int COL){
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
  struct timeval tval_before, tval_after, tval_result;
  int matrix[MATRIX_SIZE][MATRIX_SIZE];



  initMatrix(matrix);

  gettimeofday(&tval_before, NULL);
  struct maxSum_t result = findMaxSum(matrix, MATRIX_SIZE, MATRIX_SIZE);
  gettimeofday(&tval_after, NULL);

  timersub(&tval_after, &tval_before, &tval_result);
  char *fmt = "(Top, Left): (%li,%li)\n(Bottom, Right): (%li,%li)\nMaxsum: %li\nTook: %ld.%06ld";
  printf(fmt, result.finalTop, result.finalLeft,
         result.finalBottom, result.finalRight,result.maxSum, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

}
