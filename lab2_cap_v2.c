// To compile: make game
// To run: ./a.out [width] [height] [input_file]

#define _DEFAULT_SOURCE

#define GEN_LIMIT 1

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

//Función inalterada
void evolve(unsigned char **univ, unsigned char **new_univ, int width, int height)
{
    // Generate new generation: keep it in new_univ
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; y++)
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
}

//Función modificada
int empty(unsigned char **univ, int width, int height)
{
    // Checks if local is empty or not (a.k a. all the cells are dead)    
    int check = 0;
    #pragma omp parallel for shared(check, univ) schedule(dynamic)
    for (int y = 1; y < height-1; y++)  //No necesitamos verificar el exterior de la matriz pues es para los vecinos
    {
        for (int x = 1; x < width-1; x++)
        {
            if (univ[y][x] == '1')
                check = 1;
        }
    }
    
    //Se calcula la cantidad de filas vacias
    int result;
    MPI_Allreduce(&check, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    //Si es cero el universo entero está vacío
    if(result == 0){
        return true;
    }
    return false;
}

//Función modificada
int similarity(unsigned char **univ, unsigned char **new_univ, int width, int height)
{
    // Check if the new generation is the same with the previous generation
    int check = 0;
    #pragma omp parallel for shared(check, univ, new_univ) schedule(dynamic)
    for (int y = 1; y < height-1; y++)  //No necesitamos verificar el exterior de la matriz pues es para los vecinos
    {
        for (int x = 1; x < width-1; x++)
        {
            if (univ[y][x] != new_univ[y][x])
                check = 1;
        }
    }
    
    //Se calcula la cantidad de filas diferentes encontradas
    int result;
    MPI_Allreduce(&check, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    //Si es cero el universo antiguo y el nuevo son iguales
    if(result == 0){
        return true;
    }
    return false;
}

unsigned char** allocate_memory(int rows, int columns)
{
    unsigned char *data = malloc(rows * columns * sizeof(unsigned char));
    unsigned char** arr = malloc(rows * sizeof(unsigned char *));
    for (int i = 0; i < rows; i++)
        arr[i] = &(data[i*columns]);

    return arr;
}

void create_mpi_datatype(MPI_Datatype* data_type, int start_n, int start_m, int subsize_n, int subsize_m, int local_nrow, int local_ncol)
{
    int sizes[2] = {local_nrow+2, local_ncol+2};
    int subsizes[2] = {subsize_n, subsize_m};
    int starts[2] = {start_n, start_m};

    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, data_type);
    MPI_Type_commit(data_type);
}

void game(int width, int height, char *fileArg)
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {1, 1}; /*Periodicity in both dimensions*/
    MPI_Comm COMM_2D;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &COMM_2D);
    
    int *horizontal = malloc(dims[0] * sizeof(int));
    int *vertical = malloc(dims[1] * sizeof(int));

    int *disp_hor = malloc(dims[0] * sizeof(int));
    int *disp_ver = malloc(dims[1] * sizeof(int));
    
    int *nrows_proc; //Number of rows for the i-th process [local_nrow]
    int *ncols_proc; //Number of columns for the i-th process [local_ncol]
    
    if(rank == 0) {
        nrows_proc = (int *) malloc(size * sizeof(int));
        ncols_proc = (int *) malloc(size * sizeof(int));
        
        for (int i = 0; i < size; i++) {
            nrows_proc[i] = height/dims[0];
            ncols_proc[i] = width/dims[1];
        }
        for (int i = 0; i < (height%dims[0]); i++) {
            for (int j = 0; j < dims[1]; j++) {
                nrows_proc[i*dims[1]+j]++;
            }
        }
        for (int i = 0; i < (width%dims[1]); i++) {
            for (int j = 0; j < dims[0]; j++) {
                ncols_proc[i+dims[0]*j]++;
            }
        }
    }
    
    //Eje X
    int hor_proc = (int)width/dims[0];
    for(int i = 0; i < dims[0]; i++) {
        horizontal[i] = hor_proc;
    }
    for(int i = 0; i < width%dims[0]; i++) {
        horizontal[i]++;
    }
    disp_hor[0] = 0;
    for(int i = 1; i < dims[0]; i++) {
        disp_hor[i] = disp_hor[i-1] + horizontal[i-1];
    }
    
    //Eje Y
    int ver_proc = (int)height/dims[1];
    for(int i = 0; i < dims[1]; i++) {
        vertical[i] = ver_proc;
    }
    for(int i = 0; i < height%dims[1]; i++) {
        vertical[i]++;
    }
    disp_ver[0] = 0;
    for(int i = 1; i < dims[1]; i++) {
        disp_ver[i] = disp_ver[i-1] + vertical[i-1];
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
    int local_nrow, local_ncol;
    MPI_Scatter(nrows_proc, 1, MPI_INT, &local_nrow, 1, MPI_INT, 0, COMM_2D);
    MPI_Scatter(ncols_proc, 1, MPI_INT, &local_ncol, 1, MPI_INT, 0, COMM_2D);

    // Allocate space for the two game arrays (one for current generation, the other for the new one)
    unsigned char **univ = malloc(height * sizeof(unsigned char *));
    unsigned char *univ_aplanado = malloc(width * height * sizeof(unsigned char));
    for (int i = 0; i < height; i++)
    {
        univ[i] = malloc(width * sizeof(unsigned char));
        if (univ[i] == NULL)
            perror_exit("malloc: ");
    }

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
                    univ[y][x] = c;
                    x++;
                }
            }
        }
        
        //Se aplana el universo leído, se prodría añadir directamente al aplanado pero por no modificar el código original se prefiere esta forma
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                univ_aplanado[y*width+x] = univ[y][x];
            }
        }
        
        fclose(filePtr);
        filePtr = NULL;
        
        double msecs = ((float)clock() - t_start) / CLOCKS_PER_SEC * 1000.0f;
        printf("File reading time:\t%.2f msecs\n", msecs);
    }
    
    //Escribir la matriz
    int BUFF_SIZE = horizontal[0] * vertical[0];
    unsigned char *univ_send;
    int pos, x, y;
    
    if(rank == 0) {
        for(int i = 0; i < dims[1]; i++) {
            for(int j = 0; j < dims[0]; j++) {
                univ_send = malloc(BUFF_SIZE * sizeof(unsigned char *));
                
                for(int k = 0; k < vertical[i]; k++) {
                    for(int m = 0; m < horizontal[j]; m++) {
                        pos = k * horizontal[j] + m;
                        x = disp_hor[j] + m;
                        y = disp_ver[i] * width + width * k;
                        
                        univ_send[pos] = univ_aplanado[x+y];
                    }
                }
                
                MPI_Send(univ_send, BUFF_SIZE, MPI_UNSIGNED_CHAR, i*dims[0]+j, 0, MPI_COMM_WORLD);
                free(univ_send);
            }
        }
    }
    
    unsigned char *univ_recv = malloc(BUFF_SIZE * sizeof(unsigned char *));
    MPI_Recv(univ_recv, BUFF_SIZE, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
    /*
    if(rank == 0) {
        printf("\n%d\t%d\n\n", local_nrow, local_ncol);
        for(int i=0;i<local_nrow*local_ncol;++i)
        {
            printf("%c",univ_recv[i]);
            printf("\n");
        }
    }
    
    if(rank == 1) {
        printf("\n%d\t%d\n\n", local_nrow, local_ncol);
        for(int i=0;i<local_nrow*local_ncol;++i)
        {
            printf("%c",univ_recv[i]);
            printf("\n");
        }
    }
    
    if(rank == 2) {
        printf("\n%d\t%d\n\n", local_nrow, local_ncol);
        for(int i=0;i<local_nrow*local_ncol;++i)
        {
            printf("%c",univ_recv[i]);
            printf("\n");
        }
    }
    
    if(rank == 3) {
        printf("\n%d\t%d\n\n", local_nrow, local_ncol);
        for(int i=0;i<local_nrow*local_ncol;++i)
        {
            printf("%c",univ_recv[i]);
            printf("\n");
        }
    }
    */
    
    //unsigned char** local_univ = allocate_memory(local_nrow+2, local_ncol+2);
    unsigned char **local_univ = malloc(local_nrow+2 * sizeof(unsigned char *));
    for (int i = 0; i < local_nrow+2; i++)
    {
        local_univ[i] = malloc(local_ncol+2 * sizeof(unsigned char));
        if (local_univ[i] == NULL)
            perror_exit("malloc: ");
    }
    
    
    /*
    for(int i = 0; i < local_nrow+2; i++) {
        for(int j = 0; j < local_ncol+2; j++) {
            local_univ[i][j] = '2';
        }
    }
    */
    
    for(int i = 0; i < local_nrow; i++) {
        for(int j = 0; j < local_ncol; j++) {
            local_univ[i+1][j+1] = univ_recv[j+i*local_ncol];
        }
    }
    
    /*
    printf("\n");
    if(rank == 0) {
        for(int i = 0; i < local_nrow+2; i++) {
            for(int j = 0; j < local_ncol+2; j++) {
                printf("%c",local_univ[i][j]);
            }
            printf("\n");
        }
    }
    
    printf("\n");
    if(rank == 1) {
        for(int i = 0; i < local_nrow+2; i++) {
            for(int j = 0; j < local_ncol+2; j++) {
                printf("%c",local_univ[i][j]);
            }
            printf("\n");
        }
    }
    
    printf("\n");
    if(rank == 2) {
        for(int i = 0; i < local_nrow+2; i++) {
            for(int j = 0; j < local_ncol+2; j++) {
                printf("%c",local_univ[i][j]);
            }
            printf("\n");
        }
    }
    
    printf("\n");
    if(rank == 3) {
        for(int i = 0; i < local_nrow+2; i++) {
            for(int j = 0; j < local_ncol+2; j++) {
                printf("%c",local_univ[i][j]);
            }
            printf("\n");
        }
    }
    */
    
    //unsigned char **local_new_univ = allocate_memory(local_nrow+2, local_ncol+2);
    unsigned char **local_new_univ = malloc(local_nrow+2 * sizeof(unsigned char *));
    for (int i = 0; i < local_nrow+2; i++)
    {
        local_new_univ[i] = malloc(local_ncol+2 * sizeof(unsigned char));
        if (local_new_univ[i] == NULL)
            perror_exit("malloc: ");
    }

    //Create 4 datatypes for sending
    MPI_Datatype left_column_send, up_row_send, right_column_send, down_row_send;
    create_mpi_datatype(&left_column_send, 1, 1, local_nrow, 1, local_nrow, local_ncol);
    create_mpi_datatype(&up_row_send, 1, 1, 1, local_ncol, local_nrow, local_ncol);
    create_mpi_datatype(&right_column_send, 1, local_ncol, local_nrow, 1, local_nrow, local_ncol);
    create_mpi_datatype(&down_row_send, local_nrow, 1, 1, local_ncol, local_nrow, local_ncol);

    //Create 4 datatypes for receiving
    MPI_Datatype left_column_recv, up_row_recv, right_column_recv, down_row_recv;
    create_mpi_datatype(&left_column_recv, 1, 0, local_nrow, 1, local_nrow, local_ncol);
    create_mpi_datatype(&up_row_recv, 0, 1, 1, local_ncol, local_nrow, local_ncol);
    create_mpi_datatype(&right_column_recv, 1, local_ncol+1, local_nrow, 1, local_nrow, local_ncol);
    create_mpi_datatype(&down_row_recv, local_nrow+1, 1, 1, local_ncol, local_nrow, local_ncol);
    
    //Calcular los vecinos de cada submatriz local
    int up_neigh, down_neigh, left_neigh, right_neigh;                      //Filas y columnas
    int up_left_neigh, up_right_neigh, down_left_neigh, down_right_neigh;   //Esquinas
    int neigh_coords[2];
    int corners[2];
    
    //Finding top/bottom neighbours
    MPI_Cart_shift(COMM_2D, 0, 1, &up_neigh, &down_neigh);

    //Finding left/right neighbours
    MPI_Cart_shift(COMM_2D, 1, 1, &left_neigh, &right_neigh);

    //Finding top-left corner
    MPI_Cart_coords(COMM_2D, rank, 2, neigh_coords);
    corners[0] = neigh_coords[0] - 1;
    corners[1] = neigh_coords[1] - 1;
    if(corners[0] < 0)
        corners[0] = dims[0] - 1;
    if (corners[1] < 0)
        corners[1] = dims[1] - 1;
    MPI_Cart_rank(COMM_2D, corners, &up_left_neigh);
    
    //Finding top-right corner
    MPI_Cart_coords(COMM_2D, rank, 2, neigh_coords);
    corners[0] = neigh_coords[0] - 1;
    corners[1] = (neigh_coords[1]+1) % dims[1] ;
    if(corners[0] < 0)
        corners[0] = dims[0] - 1;
    MPI_Cart_rank(COMM_2D, corners, &up_right_neigh);

    //Finding bottom-left corner
    MPI_Cart_coords(COMM_2D, rank, 2, neigh_coords);
    corners[0] = (neigh_coords[0]+1) % dims[0];
    corners[1] = neigh_coords[1] - 1;
    if (corners[1] < 0)
        corners[1] = dims[1] - 1;
    MPI_Cart_rank(COMM_2D, corners, &down_left_neigh);
    
    //Finding bottom-right corner
    MPI_Cart_coords(COMM_2D, rank, 2, neigh_coords);
    corners[0] = (neigh_coords[0]+1) % dims[0];
    corners[1] = (neigh_coords[1]+1) % dims[1];
    MPI_Cart_rank(COMM_2D, corners, &down_right_neigh);
    
    //Inalterado
    int generation = 1;
#ifdef CHECK_SIMILARITY
    int counter = 0;
#endif
    
    double local_tstart, local_tfinish, local_TotalTime, result_time;   //Tiempos de inicio, fin y diferencia para cada proceso y el tiempo final
    MPI_Barrier(MPI_COMM_WORLD);                                        //Sincronizamos antes de comenzar a contar
    local_tstart = MPI_Wtime();                                         //Comienza el timer de MPI

    //LAS DIAGONALES
    while ((!empty(local_univ, local_ncol+2, local_nrow+2)) && (generation <= GEN_LIMIT))
    {
        MPI_Send(&(local_univ[0][0]), 1, left_column_send, left_neigh, 1, COMM_2D);
        MPI_Send(&(local_univ[0][0]), 1, up_row_send, up_neigh, 1, COMM_2D);
        MPI_Send(&(local_univ[0][0]), 1, right_column_send, right_neigh, 1, COMM_2D);
        MPI_Send(&(local_univ[0][0]), 1, down_row_send, down_neigh, 1, COMM_2D);
        MPI_Send(&(local_univ[1][1]), 1, MPI_UNSIGNED_CHAR, up_left_neigh, 1, COMM_2D);
        MPI_Send(&(local_univ[1][local_ncol]), 1, MPI_UNSIGNED_CHAR, up_right_neigh, 1, COMM_2D);
        MPI_Send(&(local_univ[local_nrow][local_ncol]), 1, MPI_UNSIGNED_CHAR, down_right_neigh, 1, COMM_2D);
        MPI_Send(&(local_univ[local_nrow][1]), 1, MPI_UNSIGNED_CHAR, down_left_neigh, 1, COMM_2D);

        MPI_Recv(&(local_univ[0][0]), 1, left_column_recv, left_neigh, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Recv(&(local_univ[0][0]), 1, up_row_recv, up_neigh, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Recv(&(local_univ[0][0]), 1, right_column_recv, right_neigh, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Recv(&(local_univ[0][0]), 1, down_row_recv, down_neigh, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Recv(&(local_univ[0][0]), 1, MPI_UNSIGNED_CHAR, up_left_neigh, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Recv(&(local_univ[0][local_ncol+1]), 1, MPI_UNSIGNED_CHAR, up_right_neigh, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Recv(&(local_univ[local_nrow+1][local_ncol+1]), 1, MPI_UNSIGNED_CHAR, down_right_neigh, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Recv(&(local_univ[local_nrow+1][0]), 1, MPI_UNSIGNED_CHAR, down_left_neigh, 1, COMM_2D, MPI_STATUS_IGNORE);
        
        evolve(local_univ, local_new_univ, local_ncol+2, local_nrow+2);     //Se evoluciona con los datos recibidos

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
        unsigned char **temp = local_univ;
        local_univ = local_new_univ;
        local_new_univ = temp;
        
        generation++;
    }
    
    printf("\n");
    if(rank == 0) {
        for(int i = 0; i < local_nrow+2; i++) {
            for(int j = 0; j < local_ncol+2; j++) {
                printf("%c",local_univ[i][j]);
            }
            printf("\n");
        }
    }
    
    printf("\n");
    if(rank == 1) {
        for(int i = 0; i < local_nrow+2; i++) {
            for(int j = 0; j < local_ncol+2; j++) {
                printf("%c",local_univ[i][j]);
            }
            printf("\n");
        }
    }
    
    printf("\n");
    if(rank == 2) {
        for(int i = 0; i < local_nrow+2; i++) {
            for(int j = 0; j < local_ncol+2; j++) {
                printf("%c",local_univ[i][j]);
            }
            printf("\n");
        }
    }
    
    printf("\n");
    if(rank == 3) {
        for(int i = 0; i < local_nrow+2; i++) {
            for(int j = 0; j < local_ncol+2; j++) {
                printf("%c",local_univ[i][j]);
            }
            printf("\n");
        }
    }
    
    /*
     * Finalizado el bucle se procede a recuperar los datos calculados,
     * reestructurarlos en una matriz y escribirlos a fichero.
     */
    
    /*
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
    */
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
    
    /*
    //Se escribe al fichero de salida
    if(rank == 0) {
        // show (univ, width, height);
        print_to_file(result_univ, width, height);
        free(result_univ);
    }
    */
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
 
