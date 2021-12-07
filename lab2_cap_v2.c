// To compile: make game
// To run: ./a.out [width] [height] [input_file]

#define _DEFAULT_SOURCE

#define GEN_LIMIT 0

#define CHECK_SIMILARITY
#define SIMILARITY_FREQUENCY 3

#define true 1
#define false 0

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

//Includes necesarios para OpenMP y MPI
#include <omp.h>
#include <mpi.h>

//Función inalterada
void perror_exit(const char *message)
{
    perror(message);
    exit(EXIT_FAILURE);
}

//Función inalterada
void print_to_file(unsigned char *univ, int width, int height)
{
    FILE *fout = fopen("./game_output.out", "w"); // printing the result to a file with
                                                  // 1 or 0 (1 being an alive cell and 0 a dead cell)
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            fprintf(fout, "%c", univ[i*width+j]);
        }
        fprintf(fout, "\n");
    }

    fflush(fout);
    fclose(fout);
}

//Función inalterada
void show(unsigned char **univ, int width, int height)
{
    // Prints the result in stdout, using various VT100 escape codes
    printf("\033[H");

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            printf((univ[y][x] == '1') ? "\033[07m  \033[m" : "  ");
        }

        printf("\033[E");
    }

    fflush(stdout);
}

//Añadido un parámetro para comprobar o no los vecinos que no se tiene localmente
void evolve(unsigned char **univ, unsigned char **new_univ, int width, int height, int check_neigh)
{
    if(!check_neigh) {  //Para calcular la evolución de las filas de las que se tienen sus vecinos localmente
        // Generate new generation: keep it in new_univ
        #pragma omp parallel for schedule(static)
        for (int y = 2; y < height-2; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int neighbors = 0;

                for (int y1 = y - 1; y1 <= y + 1; y1++)
                {
                    for (int x1 = x - 1; x1 <= x + 1; x1++)
                    {
                        int x2 = x1, y2 = y1;
                        if (x1 == -1)
                            x2 = width - 1;
                        if (y1 == -1)
                            y2 = height - 1;
                        if (x1 == width)
                            x2 = 0;
                        if (y1 == height)
                            y2 = 0;

                        if (univ[y2][x2] == '1')
                            neighbors++;
                    }
                }

                if (univ[y][x] == '1')
                    neighbors--;

                if (neighbors == 3 || (neighbors == 2 && (univ[y][x] == '1')))
                {
                    new_univ[y][x] = '1';
                }
                else
                {
                    new_univ[y][x] = '0';
                }
            }
        }
        
    } else {    //Para la primera y última fila almacenadas en cada proceso es necesario obtener información de los vecinos.
        
        //Primera fila local
        #pragma omp parallel for schedule(static)
        for (int x = 0; x < width; x++)
        {
            int neighbors = 0;

            for (int y1 = 1 - 1; y1 <= 1 + 1; y1++)
            {
                for (int x1 = x - 1; x1 <= x + 1; x1++)
                {
                    int x2 = x1, y2 = y1;
                    if (x1 == -1)
                        x2 = width - 1;
                    if (y1 == -1)
                        y2 = height - 1;
                    if (x1 == width)
                        x2 = 0;
                    if (y1 == height)
                        y2 = 0;

                    if (univ[y2][x2] == '1')
                        neighbors++;
                }
            }

            if (univ[1][x] == '1')
                neighbors--;

            if (neighbors == 3 || (neighbors == 2 && (univ[1][x] == '1')))
            {
                new_univ[1][x] = '1';
            }
            else
            {
                new_univ[1][x] = '0';
            }
        }
        
        //Última fila local
        #pragma omp parallel for schedule(static)
        for (int x = 0; x < width; x++)
        {
            int neighbors = 0;

            for (int y1 = (height-2) - 1; y1 <= (height-2) + 1; y1++)
            {
                for (int x1 = x - 1; x1 <= x + 1; x1++)
                {
                    int x2 = x1, y2 = y1;
                    if (x1 == -1)
                        x2 = width - 1;
                    if (y1 == -1)
                        y2 = height - 1;
                    if (x1 == width)
                        x2 = 0;
                    if (y1 == height)
                        y2 = 0;

                    if (univ[y2][x2] == '1')
                        neighbors++;
                }
            }

            if (univ[height-2][x] == '1')
                neighbors--;

            if (neighbors == 3 || (neighbors == 2 && (univ[height-2][x] == '1')))
            {
                new_univ[height-2][x] = '1';
            }
            else
            {
                new_univ[height-2][x] = '0';
            }
        }
    }
}

int empty(unsigned char **univ, int width, int height)
{
    // Checks if local is empty or not (a.k a. all the cells are dead)    
    int check = 0;
    #pragma omp parallel for shared(check, univ) schedule(static)
    for (int y = 1; y < height-1; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (univ[y][x] == '1')
                check = 1;
        }
    }
    
    int result;
    MPI_Allreduce(&check, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(result == 0){
        return true;
    }
    return false;
}

int similarity(unsigned char **univ, unsigned char **new_univ, int width, int height)
{
    // Check if the new generation is the same with the previous generation
    int check = 0;
    #pragma omp parallel for shared(check, univ, new_univ) schedule(static)
    for (int y = 1; y < height-1; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (univ[y][x] != new_univ[y][x])
                check = 1;
        }
    }
    
    int result;
    MPI_Allreduce(&check, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(result == 0){
        return true;
    }
    return false;

}

void game(int width, int height, char *fileArg)
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int *nrows_proc;    //Número de filas que se envía a cada proceso
    int *ncols_proc;    //Número de columnas que se envía a cada proceso
    int *nelem_proc;    //Número de elementos por proceso
    int *offset;        //Sobrante de la división

    if(rank == 0) {
        nrows_proc = (int*) malloc(size*sizeof(int));
        ncols_proc = (int*) malloc(size*sizeof(int));
        nelem_proc = (int*) malloc(size*sizeof(int));
        
        //Se calcula el número de filas por proceso equitativamente
        for(int i = 0; i < size; i++){
            nrows_proc[i] = height/size;
        }
        //El resto se reparte entre todos los procesos empezando por el primero
        for(int i = 0; i < height%size; i++) {
            nrows_proc[i]++;
        }
        
        //Se calcula el número de columnas por proceso equitativamente
        for(int i = 0; i < size; i++){
            ncols_proc[i] = width/size;
        }
        //El resto se reparte entre todos los procesos empezando por el primero
        for(int i = 0; i < width%size; i++) {
            ncols_proc[i]++;
        }
        
        //Se calcula cuantas posiciones tiene cada proceso 
        for(int i = 0; i < size; i++) {
            nelem_proc[i] = nrows_proc[i] * ncols_proc[i];
        }
        //AQUÍ HAY PROBLEMAS SEGURO
        //Calcular los offset para cada proceso para luego hacer el envío
        offset = (int*) malloc(size*sizeof(int));
        for(int i = 0; i < size; i++) {
            if(i == 0) {
                offset[i] = 0;
            } else {
                offset[i] = offset[i-1] + nelem_proc[i-1];
            }
        }
    }
    
    /*
     * La estructura de los datos localmente en cada proceso será la siguiente:
     * cada proceso tendrá un número de filas nrows_proc a recibir y a mayores
     * tendrá una fila encima y debajo (en caso de ser necesario) para trabajar
     * con los vecinos que le llegan de otros procesos. El ancho será el mismo
     * para todos pues se distribuye por filas, no columnas.
     * Ejemplo:
     * 
     * 000000000000
     * 0nrows_proc0
     * 0    x     0
     * 0ncols_proc0
     * 000000000000
     * 
     */

    //Cada proceso recibe el número de filas y columnas asignado
    int local_nrow;
    int local_ncol;
    MPI_Scatter(nrows_proc, 1, MPI_INT, &local_nrow, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(ncols_proc, 1, MPI_INT, &local_ncol, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate space for the two game arrays (one for current generation, the other for the new one)
    unsigned char *univ = malloc(width * height * sizeof(unsigned char)),
                  *aux_univ = malloc((2+local_nrow)*(2+local_ncol)*sizeof(unsigned char));
    if (univ == NULL)
        perror_exit("malloc: ");

    //El proceso principal lee el fichero como el programa original
    if(rank == 0){
        clock_t t_start = clock();
        
        //Inalterado
        FILE *filePtr = fopen(fileArg, "r");
        if (filePtr == NULL)
            perror_exit("fopen: ");

        // Populate univ with its contents
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width;)
            {
                char c = fgetc(filePtr);
                if ((c != EOF) && (c != '\n'))
                {
                    univ[y*width+x] = c;
                    x++;
                }
            }
        }
        fclose(filePtr);
        filePtr = NULL;
        
        double msecs = ((float)clock() - t_start) / CLOCKS_PER_SEC * 1000.0f;
        printf("File reading time:\t%.2f msecs\n", msecs);
    }

    //Se envían los datos leídos teniendo en cuenta el offset y dejando la primera fila libre
    MPI_Scatterv(univ, nelem_proc, offset, MPI_UNSIGNED_CHAR, aux_univ+width, local_nrow*local_ncol, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    free(univ);

    //AQUÍ VAMOS A TENER PROBLEMAS
    //Para facilitar el tratamiento y reutilizar la algoritmia se tranforman los datos recibidos a forma matricial
    unsigned char** local_univ = malloc((2+local_nrow)*sizeof(unsigned char*));         //Universo actual local
    unsigned char** local_new_univ = malloc((2+local_nrow)*sizeof(unsigned char*));     //Universo evolucionado local
    for(int i = 0; i < local_nrow+2; i++){
        local_univ[i] = &(aux_univ[local_ncol*i]);
        local_new_univ[i] = malloc(local_ncol*sizeof(unsigned char));
        if (local_univ[i] == NULL || local_new_univ[i] == NULL)
            perror_exit("malloc: ");
    }

    //Se definen los vecinos de cada proceso
    int up_neigh = rank - 1;
    if(up_neigh < 0){
        up_neigh = size - 1;
    }
    int down_neigh = rank + 1;
    if(down_neigh >= size){
        down_neigh = 0;
    }
    int left_neigh = rank - 1;
    if(left_neigh < 0){
        left_neigh = size - 1;
    }
    int right_neigh = rank + 1;
    if(right_neigh >= size){
        right_neigh = 0;
    }

    //Inalterado
    int generation = 1;
#ifdef CHECK_SIMILARITY
    int counter = 0;
#endif
    
    MPI_Request requests[8];        //Peticiones que cada proceso, recibe 2 filas y envía otras tantas
    MPI_Status statuses[8];         //Estado de cada petición
    
    double local_tstart, local_tfinish, local_TotalTime, result_time;   //Tiempos de inicio, fin y diferencia para cada proceso y el tiempo final
    MPI_Barrier(MPI_COMM_WORLD);                                        //Sincronizamos antes de comenzar a contar
    local_tstart = MPI_Wtime();                                         //Comienza el timer de MPI

    //LAS DIAGONALES
    while ((!empty(local_univ, local_ncol+2, local_nrow+2)) && (generation <= GEN_LIMIT))
    {
        //Se envían las dos filas que requieren los vecinos
        MPI_Send_init(&(local_univ[1][0]), local_ncol+2, MPI_UNSIGNED_CHAR, up_neigh, 1, MPI_COMM_WORLD, &requests[0]);
        MPI_Send_init(&(local_univ[local_nrow][0]), local_ncol+2, MPI_UNSIGNED_CHAR, down_neigh, 1, MPI_COMM_WORLD, &requests[1]);
        //Se envían las dos columnas que requieren los vecinos
        MPI_Send_init(&(local_univ[0][1]), local_nrow+2, MPI_UNSIGNED_CHAR, left_neigh, 1, MPI_COMM_WORLD, &requests[2]);
        MPI_Send_init(&(local_univ[0][local_ncol]), local_nrow+2, MPI_UNSIGNED_CHAR, right_neigh, 1, MPI_COMM_WORLD, &requests[3]);

        //Se reciben las filas requeridas de los vecinos
        MPI_Recv_init(&(local_univ[0][0]), local_ncol+2, MPI_UNSIGNED_CHAR, up_neigh, 1, MPI_COMM_WORLD, &requests[4]);
        MPI_Recv_init(&(local_univ[local_nrow+1][0]), local_ncol+2, MPI_UNSIGNED_CHAR, down_neigh, 1, MPI_COMM_WORLD, &requests[5]);
        //Se reciben las columnas requeridas de los vecinos
        MPI_Recv_init(&(local_univ[0][0]), local_nrow+2, MPI_UNSIGNED_CHAR, left_neigh, 1, MPI_COMM_WORLD, &requests[6]);
        MPI_Recv_init(&(local_univ[0][local_ncol+1]), local_nrow+2, MPI_UNSIGNED_CHAR, right_neigh, 1, MPI_COMM_WORLD, &requests[7]);

        MPI_Startall(8, requests); //Se ejecutan todas las peticiones

        //Si se tienen suficientes filas localmente se puede calcular parte de la evolución
        if(local_nrow >= 3 && local_ncol >= 3){
            evolve(local_univ, local_new_univ, local_ncol+2, local_nrow+2, false);
        }

        //Por último se calcula la evolución de la primera y última fila que se reciben de los vecinos
        MPI_Waitall(8, requests, statuses);                                     //Se espera a que se completen las peticioens
        evolve(local_univ, local_new_univ, local_ncol+2, local_nrow+2, true);   //Se evolucionan las filas recibidas

//Código original adaptado a los datos locales
#ifdef CHECK_SIMILARITY
        counter++;
        if (counter == SIMILARITY_FREQUENCY)
        {
            if (similarity(local_univ, local_new_univ, local_ncol+2, local_nrow+2))
                break;
            counter = 0;
        }
#endif

        //Código original adaptado a los datos locales
        unsigned char **local_temp_univ = local_univ;
        local_univ = local_new_univ;
        local_new_univ = local_temp_univ;
        
        generation++;
    }
    
    /*
     * Finalizado el bucle se procede a recuperar los datos calculados,
     * reestructurarlos en una matriz y escribirlos a fichero.
     */
    
    //Se recolectan las submatrices en un array
    unsigned char *local_result_univ = malloc(local_ncol*local_nrow*sizeof(unsigned char));
    for (int i = 0; i < local_nrow; i++){
        for (int j = 0; j < local_ncol; j++){
            local_result_univ[i*local_ncol+j] = local_univ[i+1][j];
        }
    }

    //La tarea root crea una matriz donde recibir las de todos los procesos
    unsigned char *result_univ;
    if (rank == 0) {
        result_univ = malloc((height*width)*sizeof(unsigned char));
    }
    //Se reciben las porciones de cada proceso
    MPI_Gatherv(local_result_univ, local_nrow*local_ncol, MPI_UNSIGNED_CHAR, result_univ, nelem_proc, offset, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    free(local_result_univ);
    
    //Se sincronizan los procesos y se calcula el tiempo de ejecución para cada uno
    MPI_Barrier(MPI_COMM_WORLD);
    local_tfinish = MPI_Wtime();
    local_TotalTime = local_tfinish - local_tstart;
    
    //Se elige el tiempo máximo de todos los proceso ejecutados
    MPI_Reduce(&local_TotalTime, &result_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double result_time_msecs = result_time * 1000.0f;

    //La tarea principal imprime los mismos datos que el código original
    if (rank == 0){
        printf("Finished.\n\n");
        printf("Generations:\t%d\n", generation - 1);
        //Se cambia el tiempo por el de MPI
        //printf("Execution time:\t%e msecs\n", result_time);
        printf("Execution time:\t%.2f msecs\n", result_time_msecs);
    }
    
    //Se escribe al fichero de salida
    if(rank == 0) {
        // show (univ, width, height);
        print_to_file(result_univ, width, height);
        free(result_univ);
    }
}

int main(int argc, char *argv[])
{
    int width = 0, height = 0;

    //Inicializar MPI
    MPI_Init(&argc, &argv);
    //Iniciar controlador de errores
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    //Inalterado
    if (argc > 1)
        height = atoi(argv[1]);
    if (argc > 2)
        width = atoi(argv[2]);

    if (width <= 0)
        width = 30;
    if (height <= 0)
        height = 30;

    if (argc > 3)
        game(width, height, argv[3]);

    //Se indica cuando termina cada proceso
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Finished on %d process\n", rank);
    fflush(stdout);

    //Finalmente se para MPI
    MPI_Finalize();
    
    return 0;
}
