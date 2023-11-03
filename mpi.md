# How to check if mpi is installed on the device

> $ ompi_info

# Running code on multigpu branch

> $ mpirun -np <# of processes>  ./CellDiv 20 inp.json <# sections in x> <#setions in y> <# sections in z>

Multiplication of the last three temrs should be equal to the number of processors
