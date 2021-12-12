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

char** local_matrix;
int local_N;
int local_M;
int thread_count;

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
    for (int y = 1; y < height-1; y++)  //No necesitamos verificar la primera y última fila pues son para los vecinos
    {
        for (int x = 0; x < width; x++)
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
    for (int y = 1; y < height-1; y++)  //No necesitamos verificar la primera y última fila pues son para los vecinos
    {
        for (int x = 0; x < width; x++)
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

void print_local_matrix(void)
{

	int i,j;
	for(i=1;i<=local_N;++i)
	{
		for(j=1;j<=local_M;++j)
			printf("%c",local_matrix[i][j]);
		printf("\n");
    }
}

unsigned char** allocate_memory(int rows,int columns)
{
	int i;
	unsigned char *data = malloc(rows*columns*sizeof(unsigned char));
    unsigned char** arr = malloc(rows*sizeof(unsigned char *));
    for (i=0; i<rows; i++)
        arr[i] = &(data[i*columns]);

	return arr;
}

void create_datatype(MPI_Datatype* derivedtype,int start1,int start2,int subsize1,int subsize2)
{
	const int array_of_bigsizes[2] = {local_N+2,local_M+2};
	const int array_of_subsizes[2] = {subsize1,subsize2};
	const int array_of_starts[2] = {start1,start2};

	MPI_Type_create_subarray(2,array_of_bigsizes,array_of_subsizes,array_of_starts,MPI_ORDER_C, MPI_UNSIGNED_CHAR,derivedtype);
	MPI_Type_commit(derivedtype);
}

void game(int width, int height, char *fileArg)
{
    int size, rank, i, j;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int dims[2] = {0,0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {1,1}; /*Periodicity in both dimensions*/
    int my_coords[2];
    MPI_Comm comm_2D;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_2D);
    MPI_Cart_coords(comm_2D, rank, 2, my_coords);

    const int NPROWS = dims[0]; /* Number of 'block' rows */
    const int NPCOLS = dims[1]; /* Number of 'block' cols */
    
    int *num_x = malloc(dims[0] * sizeof(int));
    int *num_y = malloc(dims[1] * sizeof(int));

    int *disp_x = malloc(dims[0] * sizeof(int));
    int *disp_y = malloc(dims[1] * sizeof(int));
    
    int *num_rows; /* Number of rows for the i-th process [local_N]*/
    int *num_cols; /* Number of columns for the i-th process [local_M]*/
    
    if(rank == 0) {
        num_rows = (int *) malloc(size * sizeof(int));
        num_cols = (int *) malloc(size * sizeof(int));
        
        int i,j;
        for (i=0; i<size; i++) {
            num_rows[i] = height/NPROWS;
            num_cols[i] = width/NPCOLS;
        }
        for (i=0; i<(height%NPROWS); i++) {
            for (j=0; j<NPCOLS; j++) {
                num_rows[i*NPCOLS+j]++;
            }
        }
        for (i=0; i<(width%NPCOLS); i++) {
            for (j=0; j<NPROWS; j++) {
                num_cols[i+NPROWS*j]++;
            }
        }
    }
    
    //Eje X
    int N_x_proc = (int)width/dims[0];
    for(int i = 0; i < dims[0]; i++) {
        num_x[i] = N_x_proc;
    }
    for(int i = 0; i < width%dims[0]; i++) {
        num_x[i]++;
    }
    disp_x[0] = 0;
    for(int i = 1; i < dims[0]; i++) {
        disp_x[i] = disp_x[i-1] + num_x[i-1];
    }
    
    //Eje Y
    int N_y_proc = (int)height/dims[1];
    for(int i = 0; i < dims[1]; i++) {
        num_y[i] = N_y_proc;
    }
    for(int i = 0; i < height%dims[1]; i++) {
        num_y[i]++;
    }
    disp_y[0] = 0;
    for(int i = 1; i < dims[1]; i++) {
        disp_y[i] = disp_y[i-1] + num_y[i-1];
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
    // Scatter dimensions,displacement,extent of each process
	MPI_Scatter(num_rows,1,MPI_INT,&local_N,1,MPI_INT,0,comm_2D);
	MPI_Scatter(num_cols,1,MPI_INT,&local_M,1,MPI_INT,0,comm_2D);

    // Allocate space for the two game arrays (one for current generation, the other for the new one)
    unsigned char **univ = malloc(height * sizeof(unsigned char *));
    unsigned char *univ_aplanado = malloc(width * height * sizeof(unsigned char));
    for (int i = 0; i < height; i++)
    {
        univ[i] = malloc(width * sizeof(unsigned char));
        if (univ[i] == NULL)
            perror_exit("malloc: ");
    }

    MPI_Barrier(MPI_COMM_WORLD);
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
    MPI_Barrier(MPI_COMM_WORLD);
    
    //Escribir la matriz
    int buf_size = num_x[0] * num_y[0];
    unsigned char *matrix_buf;
    int index, x, y;
    
    if(rank == 0) {
        for(int i = 0; i < dims[1]; i++) {
            for(int j = 0; j < dims[0]; j++) {
                matrix_buf = malloc(buf_size * sizeof(unsigned char *));
                
                for(int k = 0; k < num_y[i]; k++) {
                    for(int m = 0; m < num_x[j]; m++) {
                        index = k * num_x[j] + m;
                        
                        x = disp_x[j] + m;
                        y = disp_y[i] * width + width * k;
                        
                        matrix_buf[index] = univ_aplanado[x+y];
                    }
                }
                
                MPI_Send(matrix_buf, buf_size, MPI_UNSIGNED_CHAR, i*dims[0]+j, 0, MPI_COMM_WORLD);
                free(matrix_buf);
            }
        }
    }
    
    unsigned char *matrix_recv = malloc(buf_size * sizeof(unsigned char *));
    MPI_Recv(matrix_recv, buf_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank == 0) {
        printf("\n%d\t%d\n\n", local_N, local_M);
        for(int i=0;i<local_N*local_M;++i)
        {
            printf("%c",matrix_recv[i]);
            printf("\n");
        }
    }
    
    if(rank == 1) {
        printf("\n%d\t%d\n\n", local_N, local_M);
        for(int i=0;i<local_N*local_M;++i)
        {
            printf("%c",matrix_recv[i]);
            printf("\n");
        }
    }
    
    if(rank == 2) {
        printf("\n%d\t%d\n\n", local_N, local_M);
        for(int i=0;i<local_N*local_M;++i)
        {
            printf("%c",matrix_recv[i]);
            printf("\n");
        }
    }
    
    if(rank == 3) {
        printf("\n%d\t%d\n\n", local_N, local_M);
        for(int i=0;i<local_N*local_M;++i)
        {
            printf("%c",matrix_recv[i]);
            printf("\n");
        }
    }
    
    unsigned char **local_matrix = allocate_memory(local_N+2,local_M+2);
    if (local_matrix == NULL)
        perror_exit("malloc: ");
    
    for(int i = 0; i < local_N+2; i++) {
        for(int j = 0; j < local_M+2; j++) {
            local_matrix[i][j] = '2';
        }
    }
    
    for(int i = 0; i < local_N; i++) {
        for(int j = 0; j < local_M; j++) {
            local_matrix[i+1][j+1] = matrix_recv[j+i*local_M];
        }
    }
    

    printf("\n");
    if(rank == 0) {
        for(int i = 0; i < local_N+2; i++) {
            for(int j = 0; j < local_M+2; j++) {
                printf("%c",local_matrix[i][j]);
            }
            printf("\n");
        }
    }
    
    printf("\n");
    if(rank == 1) {
        for(int i = 0; i < local_N+2; i++) {
            for(int j = 0; j < local_M+2; j++) {
                printf("%c",local_matrix[i][j]);
            }
            printf("\n");
        }
    }
    
    printf("\n");
    if(rank == 2) {
        for(int i = 0; i < local_N+2; i++) {
            for(int j = 0; j < local_M+2; j++) {
                printf("%c",local_matrix[i][j]);
            }
            printf("\n");
        }
    }
    
    printf("\n");
    if(rank == 3) {
        for(int i = 0; i < local_N+2; i++) {
            for(int j = 0; j < local_M+2; j++) {
                printf("%c",local_matrix[i][j]);
            }
            printf("\n");
        }
    }
    
    unsigned char **next_gen = allocate_memory(local_N+2,local_M+2);

    //Create 4 datatypes for sending
    MPI_Datatype firstcolumn_send,firstrow_send,lastcolumn_send,lastrow_send;
    create_datatype(&firstcolumn_send,1,1,local_N,1);
    create_datatype(&firstrow_send,1,1,1,local_M);
    create_datatype(&lastcolumn_send,1,local_M,local_N,1);
    create_datatype(&lastrow_send,local_N,1,1,local_M);

    //Create 4 datatypes for receiving
    MPI_Datatype firstcolumn_recv,firstrow_recv,lastcolumn_recv,lastrow_recv;
    create_datatype(&firstcolumn_recv,1,0,local_N,1);
    create_datatype(&firstrow_recv,0,1,1,local_M);
    create_datatype(&lastcolumn_recv,1,local_M+1,local_N,1);
    create_datatype(&lastrow_recv,local_N+1,1,1,local_M);

    //Find ranks of my 8 neighbours
    int left,right,bottom,top,topleft,topright,bottomleft,bottomright;
    int source,dest,disp=1;
    //int my_coords[2];
    int corner_coords[2];
    int corner_rank;

    printf("HASTA AQUÍ\n");
    
    //Finding top/bottom neighbours
    MPI_Cart_shift(comm_2D,0,disp,top,bottom);

    //Finding left/right neighbours
    MPI_Cart_shift(comm_2D,1,disp,left,right);

    //Finding top-right corner
    MPI_Cart_coords(comm_2D,rank,2,my_coords);
    corner_coords[0] = my_coords[0] -1;
    corner_coords[1] = (my_coords[1] + 1) % NPCOLS ;
    if(corner_coords[0] < 0)
        corner_coords[0] = NPROWS -1;
    MPI_Cart_rank(comm_2D,corner_coords,topright);

    //Finding top-left corner
    MPI_Cart_coords(comm_2D,rank,2,my_coords);
    corner_coords[0] = my_coords[0] - 1;
    corner_coords[1] = my_coords[1] - 1 ;
    if(corner_coords[0]<0)
        corner_coords[0] = NPROWS -1;
    if (corner_coords[1]<0)
        corner_coords[1] = NPCOLS -1;
    MPI_Cart_rank(comm_2D,corner_coords,topleft);

    //Finding bottom-right corner
    MPI_Cart_coords(comm_2D,rank,2,my_coords);
    corner_coords[0] = (my_coords[0] + 1) % NPROWS ;
    corner_coords[1] = (my_coords[1] + 1) % NPCOLS ;
    MPI_Cart_rank(comm_2D,corner_coords,bottomright);

    //Finding bottom-left corner
    MPI_Cart_coords(comm_2D,rank,2,my_coords);
    corner_coords[0] = (my_coords[0] + 1) % NPROWS ;
    corner_coords[1] = my_coords[1] - 1 ;
    if (corner_coords[1]<0)
        corner_coords[1] = NPCOLS -1;
    MPI_Cart_rank(comm_2D,corner_coords,bottomleft);
    
    //16 requests , 16 statuses
    MPI_Request array_of_requests[16];
    MPI_Status array_of_statuses[16];
    
    //Inalterado
    int generation = 1;
#ifdef CHECK_SIMILARITY
    int counter = 0;
#endif
    
    
    double local_tstart, local_tfinish, local_TotalTime, result_time;   //Tiempos de inicio, fin y diferencia para cada proceso y el tiempo final
    MPI_Barrier(MPI_COMM_WORLD);                                        //Sincronizamos antes de comenzar a contar
    local_tstart = MPI_Wtime();                                         //Comienza el timer de MPI

    //LAS DIAGONALES
    while ((!empty(local_matrix, local_M+2, local_N+2)) && (generation <= GEN_LIMIT))
    {
        MPI_Send_init(&(local_matrix[0][0]),1,				firstcolumn_send,left,			1,comm_2D,&array_of_requests[0]);
        MPI_Send_init(&(local_matrix[0][0]),1,				firstrow_send,	top,			1,comm_2D,&array_of_requests[1]);
        MPI_Send_init(&(local_matrix[0][0]),1,				lastcolumn_send,right,			1,comm_2D,&array_of_requests[2]);
        MPI_Send_init(&(local_matrix[0][0]),1,				lastrow_send,	bottom,			1,comm_2D,&array_of_requests[3]);
        MPI_Send_init(&(local_matrix[1][1]),1,				MPI_UNSIGNED_CHAR,		topleft,		1,comm_2D,&array_of_requests[4]);
        MPI_Send_init(&(local_matrix[1][local_M]),1,		MPI_UNSIGNED_CHAR,		topright,		1,comm_2D,&array_of_requests[5]);
        MPI_Send_init(&(local_matrix[local_N][local_M]),1,	MPI_UNSIGNED_CHAR,		bottomright,	1,comm_2D,&array_of_requests[6]);
        MPI_Send_init(&(local_matrix[local_N][1]),1,		MPI_UNSIGNED_CHAR,		bottomleft,		1,comm_2D,&array_of_requests[7]);

        MPI_Recv_init(&(local_matrix[0][0]),1,				firstcolumn_recv,left,			1,comm_2D,&array_of_requests[8]);
        MPI_Recv_init(&(local_matrix[0][0]),1,				firstrow_recv,	top,			1,comm_2D,&array_of_requests[9]);
        MPI_Recv_init(&(local_matrix[0][0]),1,				lastcolumn_recv,right,			1,comm_2D,&array_of_requests[10]);
        MPI_Recv_init(&(local_matrix[0][0]),1,				lastrow_recv,	bottom,			1,comm_2D,&array_of_requests[11]);
        MPI_Recv_init(&(local_matrix[0][0]),1,				MPI_UNSIGNED_CHAR,		topleft,		1,comm_2D,&array_of_requests[12]);
        MPI_Recv_init(&(local_matrix[0][local_M+1]),1,		MPI_UNSIGNED_CHAR,		topright,		1,comm_2D,&array_of_requests[13]);
        MPI_Recv_init(&(local_matrix[local_N+1][local_M+1]),1,MPI_UNSIGNED_CHAR,		bottomright,	1,comm_2D,&array_of_requests[14]);
        MPI_Recv_init(&(local_matrix[local_N+1][0]),1,		MPI_UNSIGNED_CHAR,		bottomleft,		1,comm_2D,&array_of_requests[15]);
        
        if(rank==0)
        {
            printf("Generation:%d\n",generation+1);
            for(i=0;i<local_M;++i)
                putchar('~');
            putchar('\n');
            print_local_matrix();
        }
        
        //Start all requests [8 sends + 8 receives]
        MPI_Startall(16,array_of_requests);
        //Make sure all requests are completed
        MPI_Waitall(16,array_of_requests,array_of_statuses);
        
        evolve(local_matrix, next_gen, local_M+2, local_N+2);     //Se evoluciona con los datos recibidos

//Código original adaptado a los datos locales
#ifdef CHECK_SIMILARITY
        counter++;
        if (counter == SIMILARITY_FREQUENCY)
        {
            if (similarity(local_matrix, next_gen, local_M+2, local_N+2))
                break;
            counter = 0;
        }
#endif

        //Código original adaptado a los datos locales
        unsigned char **temp = local_matrix;
        local_matrix = next_gen;
        next_gen = temp;
        
        generation++;
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
 
