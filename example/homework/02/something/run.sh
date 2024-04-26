#!/bin/bash

mpic++ main.cpp -o main

mpirun -np 4 ./main 10 10 25
