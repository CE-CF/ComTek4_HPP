#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include "omp.h"

#define MATRIX_SIZE 10000

#define RAND(min, max)                                                         \
  (signed char)(((double)(max - min + 1) / RAND_MAX) * rand() + min)

/* #define TESTING  */


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

void initMatrixPtr(signed char *M[], size_t N){
  for (int i = 0; i < N; i++){
    M[i] = (signed char *) malloc(N * sizeof(signed char));
  }
}

void freeMatrix(signed char *M[], size_t N){
  for (int i = 0; i < N; i++){
    free(M[i]);
  }
}
void populateMatrix(signed char *M[], size_t N){
  srandom(time(NULL));
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      M[i][j] = RAND(-9,9);
    }
  }
}


static int kadane(signed char *arr, int *start, int *finish,int len){
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

static struct maxSum_t loopy_boi(signed char *M[], int ROW, int COL, int *start, int *finish, int _start, int maxSum){
  signed char temp[ROW];
  struct maxSum_t results;
  int sum;
  for (int i = 0; i < ROW; i++)
    temp[i] = 0;
  
  for (int right = _start; right < COL; right++){
    for (int i = 0; i < ROW; i++)
      temp[i] += M[i][right];
    
    sum = kadane(temp, start, finish, ROW);
    if (sum > maxSum){
      /* printf("Thread[%i]\t New maxSum: %i\tOldMaxsum: %i\n", */
      /*        omp_get_thread_num(), sum, maxSum); */
      maxSum = sum;
      results.maxSum = sum;
      results.finalLeft = _start;
      results.finalRight = right;
      results.finalTop = *start;
      results.finalBottom = *finish;
    }
  }
  return results;
}

static struct maxSum_t findMaxSum(signed char *M[], int ROW, int COL){
  int maxSum = 0, finalLeft = 0, finalRight = 0, finalTop = 0, finalBottom = 0;
  int sum, start = 0, finish = 0;
  struct maxSum_t results[COL];
  memset(results, 0, sizeof(struct maxSum_t) * COL);


  int p = omp_get_max_threads();

  double work = COL/p;

  int minWork = (int) floor(work);

  int remainingWork = COL - minWork * p;


  float sliceSize = (float) minWork / 2;
  int startSS = (int) floor(sliceSize);
  int endSS = (int) ceil(sliceSize);

  int fend = COL - remainingWork;

  for (int i = 0; i < p; i++){
    #pragma omp parallel for shared(results) private(maxSum) schedule(dynamic,1)
    for (int j = i*startSS; j < (i+1) * startSS; j++){
      results[j] = loopy_boi(M, ROW, COL, &start, &finish, j, maxSum);
      /* printf("%i\n", results[j].maxSum); */
      /* printf("Thread[%i]\tDone with col: %i\n", omp_get_thread_num(), j); */
    }

    #pragma omp parallel for shared(results) private(maxSum) schedule(dynamic,1)
    for (int j = fend - (i+1)*endSS; j < fend - i * endSS; j++){
      results[j] = loopy_boi(M, ROW, COL, &start, &finish, j, maxSum);
      /* printf("%i\n", results[j].maxSum); */
      /* printf("Thread[%i]\tDone with col: %i\n", omp_get_thread_num(), j); */
    }
  }
  #pragma omp parallel for shared(results) private(maxSum) schedule(dynamic,1)
  for(int j = fend; j < COL; j++){
      results[j] = loopy_boi(M, ROW, COL, &start, &finish, j, maxSum);
      /* printf("%i\n", results[j].maxSum); */
      /* printf("Thread[%i]\tDone with col: %i\n", omp_get_thread_num(), j); */
  }


/* #pragma omp parallel for shared(results) private(maxSum) schedule(dynamic,1) */
/*   for (int left = 0; left < COL; left++){ */
/*     for (int i = 0; i < ROW; i++) */
/*       temp[i] = 0; */

/*     for (int right = left; right < COL; right++){ */
/*       for (int i = 0; i < ROW; i++) */
/*         temp[i] += M[i][right]; */
      
/*       sum = kadane(temp, &start, &finish, ROW); */
/*       if (sum > maxSum){ */
/*         /\* printf("Thread[%i]\t New maxSum: %i\tOldMaxsum: %i\n", *\/ */
/*         /\*        omp_get_thread_num(), sum, maxSum); *\/ */
/*         maxSum = sum; */
/*         results[left].maxSum = sum; */
/*         results[left].finalLeft = left; */
/*         results[left].finalRight = right; */
/*         results[left].finalTop = start; */
/*         results[left].finalBottom = finish; */
/*       } */
/*     } */
/*     printf("Thread[%i]\tDone with col: %i\n", omp_get_thread_num(), left); */
/*   } */

  int oldMax = 0;
  struct maxSum_t res;
  for(int i = 0; i < COL; i++)
      /* printf("New maxSum: %i\n", */
      /*        results[i].maxSum); */
  for(int i = 0; i < COL; i++)
    if (results[i].maxSum > oldMax ){
      /* printf("New maxSum: %i\tOldMaxsum: %i\n", */
      /*        results[i].maxSum, oldMax); */
      oldMax = results[i].maxSum;
      res = results[i];
    }
  
  return res;
}


int main(){
  FILE * fp;


#ifndef TESTING
  fp = fopen("results_par.csv", "w");
  if (fp == NULL){
    exit(-1);
  }

  size_t N[4] = {10, 100, 1000, 10000};

  fputs("ID,N,MaxSum,Time\n", fp);

  for (int j; j < 4; j++) {
    signed char* matrix[N[j]];
    initMatrixPtr(matrix, N[j]);
    for (int i = 0; i < 10; i++){
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
      fprintf(fp, "%i,%ln,%i,%ld.%06ld\n", i, N, result.maxSum,
              (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
      if (tval_result.tv_sec == 0)
        sleep(1);
    }
    freeMatrix(matrix, N[j]);
  }
  fclose(fp);
#else
  size_t N = 10000;
  printf("Size: %li\n", N);
  signed char* matrix[N];
  initMatrixPtr(matrix, N);
  struct timeval tval_before, tval_after, tval_result;
  populateMatrix(matrix, N);
  gettimeofday(&tval_before, NULL);
  struct maxSum_t result = findMaxSum(matrix, N, N);
  gettimeofday(&tval_after, NULL);

  timersub(&tval_after, &tval_before, &tval_result);
  char *fmt = "(Top, Left): (%li,%li)\n(Bottom, Right): (%li,%li)"
    "\nMaxsum: %li\nTook: %ld.%06ld";
  printf(fmt, result.finalTop, result.finalLeft,
         result.finalBottom, result.finalRight,result.maxSum,
         (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
  freeMatrix(matrix, N);
#endif
}
