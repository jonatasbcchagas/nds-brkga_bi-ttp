#include <iostream>
#include <cstdio>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <list>

#include "data.h"
#include "ndsbrkga.h"

using namespace std;

class NonDominatedSet {

    private:
        
            list < pair < double, double > > allNDS;

    public:
        
            NonDominatedSet() {};
            
            int getRelation(pair < double, double > a, pair < double, double > b) {
                
                int val = 0;
            
                if (a.first < b.first) {
                    if (val == -1) return 0;
                    val = 1;
                } else if (a.first > b.first) {
                    if (val == 1) return 0;
                    val = -1;
                }
                
                if (a.second < b.second) {
                    if (val == -1) return 0;
                    val = 1;
                } else if (a.second > b.second) {
                    if (val == 1) return 0;
                    val = -1;
                }

                return val;
            }
            
            void add(pair < double, double > s) {

                bool isAdded = true;

                list < pair < double, double > > :: iterator it = allNDS.begin();
                
                for(; it != allNDS.end(); ++it) {
                    
                    pair < double, double > other = *it;

                    int rel = getRelation(s, other);

                    // if dominated by or equal in design space
                    if (rel == -1 || (rel == 0 && s == other)) {
                        isAdded = false;
                        break;
                    } else if (rel == 1) it = allNDS.erase(it);

                }

                if (isAdded) allNDS.push_back(s);
            }

            void updateNDS(vector < pair < double, double > > &inputNDS) {
                
                for(int i = 0; i < (int)inputNDS.size(); ++i) {
                    add(inputNDS[i]);
                }
            }
            
            void printNDS(ofstream &fout) {
                
                allNDS.sort();
                list < pair < double, double > > :: iterator it = allNDS.begin();                
                for(; it != allNDS.end(); ++it) {
                    fout << fixed << setprecision(5) << fabs(it->first) << ' ' << setprecision(0) << fabs(it->second) << endl;
                }
            }
};


int main(int argc, char **argv) {

    if(argc < 14 ) {
        clog << "\n       Usage:\n                     ./BITTP   input_file   output_file   N   N_e   N_m   rho_e   alpha   improv_phase_often   delta   execution_number  runtime   tsp_solution_file   kp_solution_file   preprocessing_time" << endl << endl;
        exit(0);
    }
        
    const string instanceFileName = argv[1];
    const string solutionFileName = argv[2];
    int n, execution_number, repair_operator_id, freqImprovPhase;
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
    
    Data::getInstance().readData(instanceFileName);   
    
    NonDominatedSet NDS;
    vector < pair < double, double > > inputNDS;
    

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

    int saveID = max(1, (int)(ceil(preprocessing_time/600.0)));

    char fileName[100];       
    ofstream fout;
    
    inputNDS.clear();
    ndsbrkga.storeReferenceSolutionSet(inputNDS);
    NDS.updateNDS(inputNDS);
            
    while(1) {
        
        generation += 1;
               
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double> >(t2 - t1);
        
        ndsbrkga.evolve();  // evolve the population for one generation 
        
        inputNDS.clear();
        ndsbrkga.storeReferenceSolutionSet(inputNDS);
        NDS.updateNDS(inputNDS);
        
        if(generation % freqImprovPhase == 0) {
            ndsbrkga.localSearch();
            inputNDS.clear();
            ndsbrkga.storeReferenceSolutionSet(inputNDS);
            NDS.updateNDS(inputNDS);
        }
        
        if((double)time_span.count() + preprocessing_time >= runtime*3600) {
            sprintf(fileName, "%s.f.allnds", solutionFileName.c_str());
            ofstream fout(fileName);            
            NDS.printNDS(fout);            
            fout.close();
            sprintf(fileName, "%s.f", solutionFileName.c_str());
            fout.open(fileName);
            ndsbrkga.saveReferenceSolutionSet(fout, false);
            fout.close();
            sprintf(fileName, "%s.x", solutionFileName.c_str());
            fout.open(fileName);
            ndsbrkga.saveReferenceSolutionSet(fout, true);
            fout.close();
            exit(0); 
        }
    }
    
        
    return 0;
}

