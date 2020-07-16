/*/* Fecha: 10 de Diciembre de 2019
 * Autor: Javier Carillo 
 * Contexto:
 *    Codigo para estimar el valor de pi mediante simulación MonteCarlo en paralelo para el proyecto final del curso Programacion y Analisis 
 *    de Algoritmos impartida en el CIMAT Unidad Monterrey.
 *    
 * Librerías empleadas en el desarrollo del código 
 * omp se emplea para el paralelizamiento del código
 * Math para usar la función coseno
 * Rcpp para conectar el código de R y C
*/

#include <omp.h>
#include <math.h>
#include <Rcpp.h>

//Para importar todo el espacio de nombre de la librería Rcpp

using namespace Rcpp;

//Definimos la  variable semilla que nos ayudará a calcular números aleatorios más adelante y definimos un valor teórico de Pi  

int semilla;
double CPI = 3.14159265358979323846;

/* El header [[Rcpp.... es necesario para identificar las funciones de c.
 * 
 * La función random genera un número aleatorio entre el 0 y el número rango, los número empleados para esto son número grandes que permiten 
 * obtener un periodo de convergencia largo. 
 *  
 */

//[[Rcpp::export]]
double random(double rango) {
  semilla = (semilla * 1103515245 + 12345) & 2147483647;
  return ((double)semilla) / (2147483647/rango);
}

/*La función pi_montecarlo toma como parámetro un entero n que indica el tamaño de la simulación y devuelve un valor aproximado 
 *para pi, el método consiste en una simulación Montecarlo, se generan dos número aleatorio y se cuentan el número de veces
 *que la suma del cuadrado de ambos es menor a 1, representando el número de veces que un dardo cayó dentro de un círculo de
 *radio 1, la teoría del método se describe en el anexo. 
 *
 *La función realiza una paralelización del ciclo for empleado en la generación de números random, esto nos ayuda a reducir
 *el tiempo de la simulación al repartir el trabajo en los hilos disponibles. 
 *Se inicializa la semilla con un valor aleatorio donde la aleatoriedad depende del número de hilos ocupados.
 *La variable numIn se encarga de contar el número de dardos que caen dentro del círculo. 
 *Por último se calcula y se devuelve el valor aproximado de Pi 
 */

//[[Rcpp::export]]
double pi_montecarlo(int n){
  int i, numIn;
  double x, y, pi;
  
  numIn = 0;
  
  #pragma omp threadprivate(semilla)
  #pragma omp parallel private(x, y)  reduction(+:numIn)
  {
    semilla = 25234 + 17 * omp_get_thread_num();
    #pragma omp for
    
    for (i = 0; i <= n; i++) {
      x = (double)random(1.0);  
      y = (double)random(1.0);
      if (x*x + y*y <= 1) numIn++;
    }
  }
  
  pi = 4.0*numIn / n;
  return pi;
}

/*La función buffon toma como parámetro un entero n que indica el tamaño de la simulación y devuelve un valor aproximado 
 *para pi, el método consiste en una simulación del tipo Montecarlo, donde se realiza el conteo de cuántas veces una aguja cae
 *dentro de una caja, dada la geometría del problema se generan dos número aleatorios (x,theta) x entre 0 y 1 y theta entre 0 y pi/2
 *se cuentan el número de veces que la aguja choca con alguna de las paredes y no cae dentro de la caja, esto se hace tomando en cuanta los casos 
 *que se describen en el anexo. 
 *
 *Al igual que en pi_montecarlo, la función realiza una paralelización del ciclo for empleado en la generación de números random, esto nos
 *ayuda a reducir el tiempo de la simulación al repartir el trabajo en los hilos disponibles. 
 *
 *Se inicializa la semilla con un valor aleatorio donde la aleatoriedad depende del número de hilos ocupados.
 *
 *La variable cruzan se encarga de contar el número de agujas que no caen dentro de la caja. 
 *
 *Por último se calcula y se devuelve el valor aproximado de Pi 
 */

//[[Rcpp::export]]
double buffon(int n) {
  int i, cruzan;
  double x, theta, pi;
  
  cruzan = 0;
  
  #pragma omp threadprivate(semilla)
  #pragma omp parallel private(x, theta)  reduction(+:cruzan)
  {
    semilla = 25234 + 17 * omp_get_thread_num();
    #pragma omp for
  
    for (i = 0; i <= n; i++) {
      x = (double)random(1.0);
      theta = (double)random(1.0)*CPI/2;
      if (x+cos(theta)/2 >= 1) {cruzan++;}
      else {
        if(x-cos(theta)/2 <= 0) {cruzan++;}
      }
    }
  }
  pi = (double)2*n/cruzan;
  //printf("El número de agujas que cruzan: %d de un total de %d\n", cruzan, n);
  return pi;
}


/*Se abre un chunk de R para crear el equivalente de las funciones de c en R y comparárlas, en este caso se realiza la comparación entre el método 
 * montecarlo usual, digamos el caso "sencillo" contra el equivalente en R y el método de Buffon que igual es una simulación Montecarlo
 * 
 * Se logró observar que el método más eficiente es el pi_montecarlo en su versión en C. 
 * 
 * Así mismo se logró observar que el método para generar un número aleatorio en R es más eficiente que usando C.
 */

/*** R

#La función pi_montecarlo_R toma como argumento un entero n que indica el tamaño de la muestra y devuelve la estimación de Pi realizando un proceso 
#análogo a pi_montecarlo, pero con la sintaxis de R.

pi_montecarlo_R=function(n){
  x=runif(n)
  y=runif(n)
  z=sqrt(x^2+y^2)
  return (length(which(z<=1))*4/length(z))
}

buffon_R=function(n){
  x=runif(n)
  theta=runif(n)*3.141593/2
  cuenta=length(which(x+cos(theta)/2>=1))+length(which(x-cos(theta)/2<=0))
  pi=2*n/cuenta
  return (pi)
}

N = 2000000
pi_montecarlo(N)
pi_montecarlo_R(N)
buffon(N)
buffon_R(N)


#La librería Microbenchamark nos permite comparar el tiempo empleado por cada una de las funciones. 

library(microbenchmark)
microbenchmark(random(1),runif(1))
microbenchmark(pi_montecarlo(N),pi_montecarlo_R(N))
microbenchmark(buffon(N),buffon_R(N))

*/