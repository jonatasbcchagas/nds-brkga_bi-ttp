#include <iostream>
#include <cstdio>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <fstream>

#include "data.h"
#include "ndsbrkga.h"

using namespace std;

int main(int argc, char **argv) {

    if(argc < 15 ) {
        clog << "\n       Usage:\n                     ./BITTP   input_file   output_file   N   N_e   N_m   rho_e   alpha   improv_phase_often   delta   execution_number  runtime   tsp_solution_file   kp_solution_file   preprocessing_time   store_all_nds" << endl << endl;
        exit(0);
    }
        
    const string instanceFileName = argv[1];
    const string solutionFileName = argv[2];
    int n, execution_number, repair_operator_id, freqImprovPhase, store_all_nds;
    double n_e, n_m, rhoe, alpha, runtime, preprocessing_time = 0.0;
    sscanf(argv[3], "%d", &n);
    sscanf(argv[4], "%lf", &n_e);
    sscanf(argv[5], "%lf", &n_m);
    sscanf(argv[6], "%lf", &rhoe);    
    sscanf(argv[7], "%lf", &alpha);
    sscanf(argv[8], "%d", &freqImprovPhase);    
    sscanf(argv[9], "%d", &execution_number);
    sscanf(argv[10], "%lf", &runtime);
    string tspSolutionFileName = argv[11];
    string kpSolutionFileName = argv[12];
    sscanf(argv[13], "%lf", &preprocessing_time);    
    sscanf(argv[14], "%d", &store_all_nds);    
    
    string solutionFileNameF = solutionFileName; solutionFileNameF += ".f";    
    string solutionFileNameX = solutionFileName; solutionFileNameX += ".x";
    
    
    Data::getInstance().readData(instanceFileName);   

    const unsigned chromosomeSize = Data::getInstance().numCities - 1 + Data::getInstance().numItems;    
    
    long unsigned rng_seed[] = {
                                    269070,  99470, 126489, 644764, 547617, 642580,  73456, 462018, 858990, 756112, 
                                    701531, 342080, 613485, 131654, 886148, 909040, 146518, 782904,   3075, 974703, 
                                    170425, 531298, 253045, 488197, 394197, 519912, 606939, 480271, 117561, 900952, 
                                    968235, 345118, 750253, 420440, 761205, 130467, 928803, 768798, 640300, 871462, 
                                    639622,  90614, 187822, 594363, 193911, 846042, 680779, 344008, 759862, 661168, 
                                    223420, 959508,  62985, 349296, 910428, 964420, 422964, 384194, 985214,  57575, 
                                    639619,  90505, 435236, 465842, 102567, 189997, 741017, 611828, 699223, 335142, 
                                     52119,  49256, 324523, 348215, 651525, 517999, 830566, 958538, 880422, 390645, 
                                    148265, 807740, 934464, 524847, 408760, 668587, 257030, 751580,  90477, 594476, 
                                    571216, 306614, 308010, 661191, 890429, 425031,  69108, 435783,  17725, 335928
                                };

    MTRand rng(rng_seed[execution_number-1]);  // initialize the random number generator

    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // initialize the NSBRKGA-based heuristic
    NDSBRKGA < Decoder, MTRand > ndsbrkga(chromosomeSize, n, n_e, n_m, rhoe, alpha, tspSolutionFileName, kpSolutionFileName, rng);

    int generation = 0;

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration < double > time_span = duration_cast < duration < double > > (t2 - t1);
        
    while(1) {
       
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double> >(t2 - t1);
        
        ndsbrkga.evolve();  // evolve the population for one generation    
        
        if(store_all_nds == 1) ndsbrkga.saveReferenceSolutionSet(solutionFileNameX, solutionFileNameF);
        
        if(generation % freqImprovPhase == 0) {
            ndsbrkga.localSearch();
            if(store_all_nds == 1) ndsbrkga.saveReferenceSolutionSet(solutionFileNameX, solutionFileNameF);
        }
        
        generation += 1;
        
        if((double)time_span.count() + preprocessing_time >= runtime*3600) {
            if(store_all_nds == 0) ndsbrkga.saveReferenceSolutionSet(solutionFileNameX, solutionFileNameF);
            exit(0); 
        }
    }
       
    return 0;
}

