#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import time
import math
import os
import subprocess

def solve_tsp_lkh(input_file, output_file, execution_number, tsp_t):

    tsp_instance = "NAME : ***\n"\
                   "COMMENT : #####\n"\
                   "TYPE : TSP\n"\
                   "DIMENSION : ***\n"\
                   "EDGE_WEIGHT_TYPE : CEIL_2D\n"\
                   "NODE_COORD_SECTION\n"

    f = open(input_file, 'r')
    lines = f.readlines()
    f.close()

    # lines[0]  // PROBLEM NAME: 	a280-TTP
    problem_name = lines[0].split()[-1]
    # lines[1]  // KNAPSACK DATA TYPE: bounded strongly corr
    # lines[2]  // DIMENSION:	280
    number_of_cities = int(lines[2].split()[-1])
    # lines[3]  // NUMBER OF ITEMS: 	27
    # lines[4]  // CAPACITY OF KNAPSACK: 	25936
    # lines[5]  // MIN SPEED: 0.1
    # lines[6]  // MAX SPEED: 1
    # lines[7]  // RENTING RATIO: 	5.61
    # lines[8]  // EDGE_WEIGHT_TYPE: CEIL_2D
    # lines[9]  // NODE_COORD_SECTION(INDEX, X, Y):

    tsp_instance_file = "./TSPComponent/" + output_file.split('/')[-1] + "_%02d.tsp" %(int(execution_number))

    f = open(tsp_instance_file, 'w')
    f.write(tsp_instance.replace("DIMENSION : ***", "DIMENSION : %d" % (number_of_cities)).replace("NAME : ***","NAME : %s" % (problem_name)))
    for line in lines[10:10+number_of_cities]:
        f.write(line)
    f.write("EOF\n")
    f.close()    
    
    par = "PROBLEM_FILE = ***\n"\
          "PATCHING_C = 3\n"\
          "PATCHING_A = 2\n"\
          "CANDIDATE_SET_TYPE = QUADRANT\n"\
          "SEED = ***\n"\
          "RUNS = 1\n"\
          "TIME_LIMIT = ***\n"\
          "TOUR_FILE = ***\n"

    tsp_solution_file = output_file + ".tsp.sol"
    par = par.replace("PROBLEM_FILE = ***", "PROBLEM_FILE = %s" % (tsp_instance_file)).replace("SEED = ***", "SEED = %d" %(int(execution_number))).replace("TIME_LIMIT = ***", "TIME_LIMIT = %d" %(int(tsp_t))).replace("TOUR_FILE = ***", "TOUR_FILE = %s" % (tsp_solution_file))

    par_file = tsp_instance_file + ".par"

    f = open(par_file, 'w')
    f.write(par)
    f.close()

    os.system("./TSPComponent/LKH " + par_file + " >/dev/null")
    
    os.remove(tsp_instance_file)
    os.remove(par_file)

    return tsp_solution_file

def solve_kp_ghdp(input_file, output_file, execution_number, kp_delta):

    kp_solution_file = output_file + ".kp.sol"
    os.system("./KPComponent/KP " + input_file + " " + kp_solution_file + " " + kp_delta)
    return kp_solution_file

def run_algorithm(input_file, output_file, n, n_e, n_m, rho_e, alpha, omega, tsp_t, kp_delta, execution_number, runtime):

    start = time.time()
    tsp_solution_file = solve_tsp_lkh(input_file, output_file, execution_number, tsp_t)
    kp_solution_file = solve_kp_ghdp(input_file, output_file, execution_number, kp_delta)
    end = time.time()
    os.system("./BITTP %s %s %s %s %s %s %s %s %s %s %s %s %.3f" % (input_file, output_file, n, n_e, n_m, rho_e, alpha, omega, execution_number, runtime, tsp_solution_file, kp_solution_file, end - start))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="BI-TTP instance")
    parser.add_argument("output_file", help="output file to store the BI-TTP solutions")
    parser.add_argument("N", help="population size")
    parser.add_argument("N_e", help="elite population size")
    parser.add_argument("N_m", help="mutation population size")
    parser.add_argument("rho_e", help="elite allele inheritance probability")
    parser.add_argument("alpha", help="fraction of the initial population created from TSP and KP solutions")
    parser.add_argument("omega", help="indicates how often the improvement procedure is applied")
    parser.add_argument("tsp_t", help="time limit in seconds to solve the TSP component via LKH algorithm")
    parser.add_argument("kp_delta", help="maximum capacity of the sub-knapsack solved by dynamic programming")
    parser.add_argument("execution_number", help="execution number from [1, 100]")
    parser.add_argument("runtime", help="number of processing hours of the NDS-BRKGA (stopping criterion)")
    args = parser.parse_args()

    run_algorithm(args.input_file, args.output_file, args.N, args.N_e, args.N_m, args.rho_e, args.alpha, args.omega, args.tsp_t, args.kp_delta, args.execution_number, args.runtime)

