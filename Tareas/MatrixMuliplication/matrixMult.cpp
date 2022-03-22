// Modificado por Alberto Josué Ortiz Rosales 
// 22 Mar 2022
// 
// Este código tiene la intención de mostrar el
// uso de las instrucciones avx en la multiplicación 
// de matrices para poder paralelizar la multiplicacion
// de filas y columnas. Se compara esta implementación con 
// la forma normal de multiplicar una matriz.
// 
// Compilación :
//  $ g++ -o matrixMult matrixMult.cpp -march=native
//
// Se puede agregar las bandera de optiminación -O, -O2 y -O3
//
// Ejecución :
// $ ./matrixMult

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

#define ROWS1 200
#define COLS1 200
#define COLS2 200

int main()
{

	float diff, tmp;

	float w[ROWS1][COLS1];
	float x[COLS1][COLS2];
	float y[ROWS1][COLS2];
	float T[COLS2][COLS1]; // TRANSPUESTA DE X
	float scratchpad[8];
	for (int i = 0; i < ROWS1; i++)
	{
		// Genera una matriz pseudo aleatoria
		for (int j = 0; j < COLS1; j++)
		{
			w[i][j] = (float)(rand() % 1000) / 800.0f;
		}
	}
	for (int i = 0; i < COLS1; i++)
	{
		for (int j = 0; j < COLS2; j++)
		{
			//x[i][j] = (i == j) ? 2.0 : 0.0;
			x[i][j] = (float)(rand() % 1000) / 800.0f;
		}
	}

	clock_t t1, t2;

	t1 = clock();
	// Multiplicación normal
	for (int i = 0; i < ROWS1; i++)
	{
		for (int j = 0; j < COLS2; j++)
		{
			tmp = 0;
			for (int k = 0; k < COLS1; k++)
			{
				tmp += w[i][k] * x[k][j];
			}
			y[i][j] = tmp;
		}
	}
	t2 = clock();
	printf("Normal:      %f \n", (double)(t2 - t1) / CLOCKS_PER_SEC);

	// Quitar los comentarios para imprimir la matriz resultante

	/* for (int i = 0; i < ROWS1; i++)
	{
		for (int j = 0; j < COLS2; j++)
		{
			printf("%3.2lf ", y[i][j]);
		}
		printf("\n");
	}
	printf("\n \n");
	*/

	// Hay que transponer para poder usar la función 
	// _mm256_loadu_ps ya que copia los contenidos de la 
	// memmoria al registro de _m256 de manera lineal. ya que 
	// x se necesitan las columnas al transponer se puede recorrer
	// las columnas de manera lineal. Hay otra opción la cual 
	// consiste en copiar las columnas en un arreglo.
	for (int i = 0; i < COLS1; ++i)
	{
		for (int j = 0; j < COLS2; ++j)
		{
			T[j][i] = x[i][j];
		}
	}

	// Registros de 256 bits
	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
	ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

	t1 = clock();
	const int reduced_64 = COLS1 - COLS1 % 64;  // tamaño máximo para realizar operaciones de 64 elementos en 64
	const int reduced_32 = COLS1 - COLS1 % 32;  // tamaño máximo para realizar operaciones de 32 elementos en 32
	const int reduced_16 = COLS1 - COLS1 % 16;  // tamaño máximo para realizar operaciones de 16 elementos en 16
	const int reduced_8 = COLS1 - COLS1 % 8;    // tamaño máximo para realizar operaciones de 8 elementos en 8
	
	for (int i = 0; i < ROWS1; i++)
	{
		for (int l = 0; l < COLS2; l++)
		{
			float res = 0;
			for (int j = 0; j < reduced_64; j += 64)
			{
				// Carga de los datos de la columna de B
				ymm8  = _mm256_loadu_ps(&T[l][j]);
				ymm9  = _mm256_loadu_ps(&T[l][j + 8]);
				ymm10 = _mm256_loadu_ps(&T[l][j + 16]);
				ymm11 = _mm256_loadu_ps(&T[l][j + 24]);
				ymm12 = _mm256_loadu_ps(&T[l][j + 32]);
				ymm13 = _mm256_loadu_ps(&T[l][j + 40]);
				ymm14 = _mm256_loadu_ps(&T[l][j + 48]);
				ymm15 = _mm256_loadu_ps(&T[l][j + 56]);
				// Carga de los datos de la fila de A
				ymm0 = _mm256_loadu_ps(&w[i][j]);
				ymm1 = _mm256_loadu_ps(&w[i][j + 8]);
				ymm2 = _mm256_loadu_ps(&w[i][j + 16]);
				ymm3 = _mm256_loadu_ps(&w[i][j + 24]);
				ymm4 = _mm256_loadu_ps(&w[i][j + 32]);
				ymm5 = _mm256_loadu_ps(&w[i][j + 40]);
				ymm6 = _mm256_loadu_ps(&w[i][j + 48]);
				ymm7 = _mm256_loadu_ps(&w[i][j + 56]);
				// Multiplicacion de la fila por columna
				ymm0 = _mm256_mul_ps(ymm0, ymm8 );
				ymm1 = _mm256_mul_ps(ymm1, ymm9 );
				ymm2 = _mm256_mul_ps(ymm2, ymm10);
				ymm3 = _mm256_mul_ps(ymm3, ymm11);
				ymm4 = _mm256_mul_ps(ymm4, ymm12);
				ymm5 = _mm256_mul_ps(ymm5, ymm13);
				ymm6 = _mm256_mul_ps(ymm6, ymm14);
				ymm7 = _mm256_mul_ps(ymm7, ymm15);
				// Reducción de los productos
				ymm0 = _mm256_add_ps(ymm0, ymm1);
				ymm2 = _mm256_add_ps(ymm2, ymm3);
				ymm4 = _mm256_add_ps(ymm4, ymm5);
				ymm6 = _mm256_add_ps(ymm6, ymm7);
				ymm0 = _mm256_add_ps(ymm0, ymm2);
				ymm4 = _mm256_add_ps(ymm4, ymm6);
				ymm0 = _mm256_add_ps(ymm0, ymm4);

		__builtin_ia32_storeups256(scratchpad, ymm0);
		// _mm256_store_ps(scratchpad, ymm0); Produce segmentación de memoria cuando no 
		// se ultiliza alguna bandera de optimización 
				// reducción final
				for (int k = 0; k < 8; k++)
					res += scratchpad[k];
			}
			for (int j = reduced_64; j < reduced_32; j += 32)
			{
				ymm8  = _mm256_loadu_ps(&T[l][j]);
				ymm9  = _mm256_loadu_ps(&T[l][j + 8]);
				ymm10 = _mm256_loadu_ps(&T[l][j + 16]);
				ymm11 = _mm256_loadu_ps(&T[l][j + 24]);

				ymm0 = _mm256_loadu_ps(&w[i][j]);
				ymm1 = _mm256_loadu_ps(&w[i][j + 8]);
				ymm2 = _mm256_loadu_ps(&w[i][j + 16]);
				ymm3 = _mm256_loadu_ps(&w[i][j + 24]);

				ymm0 = _mm256_mul_ps(ymm0, ymm8);
				ymm1 = _mm256_mul_ps(ymm1, ymm9);
				ymm2 = _mm256_mul_ps(ymm2, ymm10);
				ymm3 = _mm256_mul_ps(ymm3, ymm11);

				ymm0 = _mm256_add_ps(ymm0, ymm1);
				ymm2 = _mm256_add_ps(ymm2, ymm3);
				ymm0 = _mm256_add_ps(ymm0, ymm2);

				__builtin_ia32_storeups256(scratchpad, ymm0);
				for (int k = 0; k < 8; k++)
					res += scratchpad[k];
			}
			for (int j = reduced_32; j < reduced_16; j += 16)
			{
				ymm8 = _mm256_loadu_ps(&T[l][j]);
				ymm9 = _mm256_loadu_ps(&T[l][j + 8]);
			

				ymm0 = _mm256_loadu_ps(&w[i][j]);
				ymm1 = _mm256_loadu_ps(&w[i][j + 8]);
			

				ymm0 = _mm256_mul_ps(ymm0, ymm8);
				ymm1 = _mm256_mul_ps(ymm1, ymm9);
	

				ymm0 = _mm256_add_ps(ymm0, ymm1);

				__builtin_ia32_storeups256(scratchpad, ymm0);
				for (int k = 0; k < 8; k++)
					res += scratchpad[k];
			}
			for (int j = reduced_16; j < reduced_8; j += 8)
			{
				ymm8 = _mm256_loadu_ps(&T[l][j]);
			
			

				ymm0 = _mm256_loadu_ps(&w[i][j]);
			

				ymm0 = _mm256_mul_ps(ymm0, ymm8);

				__builtin_ia32_storeups256(scratchpad, ymm0);
				for (int k = 0; k < 8; k++)
					res += scratchpad[k];
			}
			for (int j = reduced_8; j < COLS1; j++)
			{
				res += w[i][j] * T[l][j];
			}
			y[i][l] = res;
		}
	}
	t2 = clock();
	printf("Intrinsics:  %f  \n", (double)(t2 - t1) / CLOCKS_PER_SEC);

	/* for (int i = 0; i < ROWS1; i++)
	{
		for (int j = 0; j < COLS2; j++)
		{
			printf("%3.2lf ", y[i][j]);
		}
		printf("\n");
	}
	printf("\n \n");
	*/

	return 0;
}