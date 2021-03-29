# A Non-Dominated Sorting Based Customized Random-Key Genetic Algorithm for the Bi-Objective Traveling Thief Problem

This project contains the code of the Non-Dominated Sorting Based Customized Random-Key Genetic Algorithm (NDS-BRKGA), which is described in detail in [Chagas et al. (2020)](https://doi.org/10.1007/s10732-020-09457-7), for solving the Bi-Objective Traveling Thief Problem (BI-TTP).

### Compiling the code

Before running the NDS-BRKGA, it is needed to compile its code as well as the code of its sub-algorithms (see our paper for further details). To this end, just run the following command:

```console
$ sh compile.sh
```

### Usage:

```console
$ python launcher.py <instance> <output> <N> <N_e> <N_m> <rho_e> <alpha> <omega> <tsp_t> <kp_delta> <execution_number> <runtime>

  where:

        <instance>              : Instance name
        <output>                : Two output files are generated: <output>.x contains the list of all non-dominated solutions and <output>.f the list of objective values
        <N>                     : Population size
        <N_e>                   : Elite population size in terms of <N> (e.g. <N_e> = 0.1 --> 0.1<N> elite individuals)
        <N_m>                   : Mutant population size in terms of <N> (e.g. <N_m> = 0.1 --> 0.1<N> mutant individuals)
        <rhoe_e>                : Elite allele inheritance probability
        <alpha>                 : Fraction of the initial population created from TSP and KP solutions
        <omega>                 : Frequency of local search procedure
        <tsp_t>:                : Time limit in seconds to solve the TSP component via LKH algorithm
        <kp_delta>              : Maximum capacity of the sub-knapsack solved by the dynamic programming algorithm
        <execution_number>      : Execution number of the NDS-BRKGA
        <runtime>               : Number of processing hours of the NDS-BRKGA (stopping criterion)
```

Example:

```console
$ python launcher.py instances/a280_n279 a280_n279.sol 1000 0.6 0.0 0.8 0.3 50 300 50000 1 5
```
