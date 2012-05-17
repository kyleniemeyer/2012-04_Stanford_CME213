#!/bin/sh
#PBS -l nodes=1:ppn=4

########### UPDATE THESE VARIABLES ###############
# the directory where your files are
pa4_home=/home/dbtsai/2012-04_stanford_cme213/hw/hw4/programming
##################################################

cd $pa4_home
#echo $USER
echo $pa4_home
######### ADD YOUR EXECUTION SCRIPT HERE #########
# Set the number of threads
# Clean up the directory
make clean
# Compile the program
make
# Run Mergesort

for i in 1 2 4 8 16 32 64
do
	echo "\n"
	echo $i
	export OMP_NUM_THREADS=$i
#	./mergesort 100000 3000000 48000000 1
	./radixsort 1000000 8
	./radixsort 16000000 8
done
# Run radixsort
#./radixsort 1000000
