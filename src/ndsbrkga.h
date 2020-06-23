#ifndef NDSBRKGA_H
#define NDSBRKGA_H

#include "data.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <climits>
#include <fstream>
#include <iomanip>

const double INF = 987654321;
const double EPS = 1E-5;

using namespace std;

class Population;

//=====================================================================================================================//

class Decoder {
    
    public:

            static Decoder& getInstance() {
                static Decoder instance;                                        
                return instance;
            }
            
            Decoder(Decoder const&)         = delete;
            void operator=(Decoder const&)  = delete;
                
            void decode(std::pair < std::vector < double >, std::vector < double > >&);    
            void localSearch(std::pair < std::vector < double >, std::vector < double > >&, Population*);            
            bool compare(const std::pair < std::vector < double >, std::vector < double > >&, const std::pair < std::vector < double >, std::vector < double > >&) const;                    
            void save(const std::pair < std::vector < double >, std::vector < double > >&, ofstream&) const;
            
            static std::default_random_engine generator;
  
            const double alpha = 0.5;  
            
    private:
        
            void repairOperator(std::pair < std::vector < double >, std::vector < double > >&, std::vector < std::pair < double, int > >&, std::vector < int >&, std::vector < int >&, std::vector < int >&, int) const;
            
            Decoder() {};
};


//=====================================================================================================================//

class MTRand {
    
    // Data
    public:
        typedef unsigned long uint32;  // unsigned integer type, at least 32 bits
        
        enum { N = 624 };       // length of state vector
        enum { SAVE = N + 1 };  // length of array for save()

    protected:
        enum { M = 397 };  // period parameter
        
        uint32 state[N];   // internal state
        uint32 *pNext;     // next value to get from state
        int left;          // number of values left before reload needed

    // Methods
    public:
        
        MTRand( const uint32 oneSeed );  // initialize with a simple uint32
        MTRand( uint32 *const bigSeed, uint32 const seedLength = N );  // or array
        MTRand();  // auto-initialize with /dev/urandom or time() and clock()
        MTRand( const MTRand& o );  // copy
        
        // Do NOT use for CRYPTOGRAPHY without securely hashing several returned
        // values together, otherwise the generator state can be learned after
        // reading 624 consecutive values.
        
        // Access to 32-bit random numbers
        uint32 randInt();                     // integer in [0,2^32-1]
        uint32 randInt( const uint32 n );     // integer in [0,n] for n < 2^32
        //double rand();                        // real number in [0,1] -- disabled by rtoso
        //double rand( const double n );        // real number in [0,n]    -- disabled by rtoso
        double randExc();                     // real number in [0,1)
        double randExc( const double n );     // real number in [0,n)
        double randDblExc();                  // real number in (0,1)
        double randDblExc( const double n );  // real number in (0,n)
        double operator()();                  // same as rand53()
        
        // Access to 53-bit random numbers (capacity of IEEE double precision)
        double rand();        // calls rand53() -- modified by rtoso
        double rand53();      // real number in [0,1)
        
        // Access to nonuniform random number distributions
        double randNorm( const double mean = 0.0, const double stddev = 1.0 );
        
        // Re-seeding functions with same behavior as initializers
        void seed( const uint32 oneSeed );
        void seed( uint32 *const bigSeed, const uint32 seedLength = N );
        void seed();
        
        // Saving and loading generator state
        void save( uint32* saveArray ) const;  // to array of size SAVE
        void load( uint32 *const loadArray );  // from such array
        friend std::ostream& operator <<( std::ostream& os, const MTRand& mtrand );
        friend std::istream& operator>>( std::istream& is, MTRand& mtrand );
        MTRand& operator=( const MTRand& o );

    protected:
        
        void initialize( const uint32 oneSeed );
        void reload();
        uint32 hiBit( const uint32 u ) const { return u & 0x80000000UL; }
        uint32 loBit( const uint32 u ) const { return u & 0x00000001UL; }
        uint32 loBits( const uint32 u ) const { return u & 0x7fffffffUL; }
        uint32 mixBits( const uint32 u, const uint32 v ) const { return hiBit(u) | loBits(v); }
        uint32 magic( const uint32 u ) const { return loBit(u) ? 0x9908b0dfUL : 0x0UL; }
        uint32 twist( const uint32 m, const uint32 s0, const uint32 s1 ) const { return m ^ (mixBits(s0,s1)>>1) ^ magic(s1); }
        static uint32 hash( time_t t, clock_t c );
};

// Functions are defined in order of usage to assist inlining

inline MTRand::uint32 MTRand::hash( time_t t, clock_t c ) {
    
    // Get a uint32 from t and c
    // Better than uint32(x) in case x is floating point in [0,1]
    // Based on code by Lawrence Kirby (fred@genesis.demon.co.uk)
    
    static uint32 differ = 0;  // guarantee time-based seeds will change
    
    uint32 h1 = 0;
    unsigned char *p = (unsigned char *) &t;
    for( size_t i = 0; i < sizeof(t); ++i ) {
        h1 *= UCHAR_MAX + 2U;
        h1 += p[i];
    }
    uint32 h2 = 0;
    p = (unsigned char *) &c;
    for( size_t j = 0; j < sizeof(c); ++j ){
        h2 *= UCHAR_MAX + 2U;
        h2 += p[j];
    }
    return ( h1 + differ++ ) ^ h2;
}

inline void MTRand::initialize( const uint32 seed ) {
    
    // Initialize generator state with seed
    // See Knuth TAOCP Vol 2, 3rd Ed, p.106 for multiplier.
    // In previous versions, most significant bits (MSBs) of the seed affect
    // only MSBs of the state array.  Modified 9 Jan 2002 by Makoto Matsumoto.
    register uint32 *s = state;
    register uint32 *r = state;
    register int i = 1;
    *s++ = seed & 0xffffffffUL;
    for( ; i < N; ++i ) {
        *s++ = ( 1812433253UL * ( *r ^ (*r >> 30) ) + i ) & 0xffffffffUL;
        r++;
    }
}

inline void MTRand::reload() {
    
    // Generate N new values in state
    // Made clearer and faster by Matthew Bellew (matthew.bellew@home.com)
    static const int MmN = int(M) - int(N);  // in case enums are unsigned
    register uint32 *p = state;
    register int i;
    for( i = N - M; i--; ++p ) {
        *p = twist( p[M], p[0], p[1] );
    }
    for( i = M; --i; ++p ) {
        *p = twist( p[MmN], p[0], p[1] );
    }
    *p = twist( p[MmN], p[0], state[0] );
    
    left = N, pNext = state;
}

inline void MTRand::seed( const uint32 oneSeed ) {
    
    // Seed the generator with a simple uint32
    initialize(oneSeed);
    reload();
}

inline void MTRand::seed( uint32 *const bigSeed, const uint32 seedLength ) {
    
    // Seed the generator with an array of uint32's
    // There are 2^19937-1 possible initial states.  This function allows
    // all of those to be accessed by providing at least 19937 bits (with a
    // default seed length of N = 624 uint32's).  Any bits above the lower 32
    // in each element are discarded.
    // Just call seed() if you want to get array from /dev/urandom
    initialize(19650218UL);
    register int i = 1;
    register uint32 j = 0;
    register int k = ( N > seedLength ? N : seedLength );
    for( ; k; --k ) {
        state[i] =
        state[i] ^ ( (state[i-1] ^ (state[i-1] >> 30)) * 1664525UL );
        state[i] += ( bigSeed[j] & 0xffffffffUL ) + j;
        state[i] &= 0xffffffffUL;
        ++i;  ++j;
        if( i >= N ) { state[0] = state[N-1];  i = 1; }
        if( j >= seedLength ) j = 0;
    }
    for( k = N - 1; k; --k ) {
        state[i] =
        state[i] ^ ( (state[i-1] ^ (state[i-1] >> 30)) * 1566083941UL );
        state[i] -= i;
        state[i] &= 0xffffffffUL;
        ++i;
        if( i >= N ) { state[0] = state[N-1];  i = 1; }
    }
    state[0] = 0x80000000UL;  // MSB is 1, assuring non-zero initial array
    reload();
}

inline void MTRand::seed() {
    
    // Seed the generator with an array from /dev/urandom if available
    // Otherwise use a hash of time() and clock() values
    
    // First try getting an array from /dev/urandom
    FILE* urandom = fopen( "/dev/urandom", "rb" );
    if( urandom ) {
        uint32 bigSeed[N];
        register uint32 *s = bigSeed;
        register int i = N;
        register bool success = true;
        while( success && i-- ) {
            success = fread( s++, sizeof(uint32), 1, urandom );
        }
        fclose(urandom);
        if( success ) { seed( bigSeed, N );  return; }
    }
    
    // Was not successful, so use time() and clock() instead
    seed( hash( time(NULL), clock() ) );
}

inline MTRand::MTRand( const uint32 oneSeed ) { seed(oneSeed); }

inline MTRand::MTRand( uint32 *const bigSeed, const uint32 seedLength ) { seed(bigSeed,seedLength); }

inline MTRand::MTRand() { seed(); }

inline MTRand::MTRand( const MTRand& o ) {
    
    register const uint32 *t = o.state;
    register uint32 *s = state;
    register int i = N;
    for( ; i--; *s++ = *t++ ) {}
    left = o.left;
    pNext = &state[N-left];
}

inline MTRand::uint32 MTRand::randInt() {
    
    // Pull a 32-bit integer from the generator state
    // Every other access function simply transforms the numbers extracted here
    
    if( left == 0 ) reload();
    --left;
    
    register uint32 s1;
    s1 = *pNext++;
    s1 ^= (s1 >> 11);
    s1 ^= (s1 <<  7) & 0x9d2c5680UL;
    s1 ^= (s1 << 15) & 0xefc60000UL;
    return ( s1 ^ (s1 >> 18) );
}

inline MTRand::uint32 MTRand::randInt( const uint32 n ) {
    
    // Find which bits are used in n
    // Optimized by Magnus Jonsson (magnus@smartelectronix.com)
    uint32 used = n;
    used |= used >> 1;
    used |= used >> 2;
    used |= used >> 4;
    used |= used >> 8;
    used |= used >> 16;
    
    // Draw numbers until one is found in [0,n]
    uint32 i;
    do {
        i = randInt() & used;  // toss unused bits to shorten search
    } while( i > n );
    return i;
}

//inline double MTRand::rand()
//    { return double(randInt()) * (1.0/4294967295.0); }

//inline double MTRand::rand( const double n )
//    { return rand() * n; }

inline double MTRand::randExc() { return double(randInt()) * (1.0/4294967296.0); }

inline double MTRand::randExc( const double n ) { return randExc() * n; }

inline double MTRand::randDblExc() { return ( double(randInt()) + 0.5 ) * (1.0/4294967296.0); }

inline double MTRand::randDblExc( const double n ) { return randDblExc() * n; }

inline double MTRand::rand53() {
    uint32 a = randInt() >> 5, b = randInt() >> 6;
    return ( a * 67108864.0 + b ) * (1.0/9007199254740992.0);  // by Isaku Wada
}

inline double MTRand::rand() { return rand53(); }

inline double MTRand::randNorm( const double mean, const double stddev ) {
    // Return a real number from a normal (Gaussian) distribution with given
    // mean and standard deviation by polar form of Box-Muller transformation
    double x, y, r;
    do
    {
        x = 2.0 * rand53() - 1.0;
        y = 2.0 * rand53() - 1.0;
        r = x * x + y * y;
    }
    while ( r >= 1.0 || r == 0.0 );
    double s = sqrt( -2.0 * log(r) / r );
    return mean + x * s * stddev;
}

inline double MTRand::operator()() {
    return rand53();
}

inline void MTRand::save( uint32* saveArray ) const {
    register const uint32 *s = state;
    register uint32 *sa = saveArray;
    register int i = N;
    for( ; i--; *sa++ = *s++ ) {}
    *sa = left;
}

inline void MTRand::load( uint32 *const loadArray ) {
    register uint32 *s = state;
    register uint32 *la = loadArray;
    register int i = N;
    for( ; i--; *s++ = *la++ ) {}
    left = *la;
    pNext = &state[N-left];
}

inline std::ostream& operator <<( std::ostream& os, const MTRand& mtrand ) {
    register const MTRand::uint32 *s = mtrand.state;
    register int i = mtrand.N;
    for( ; i--; os << *s++ << "    " ) {}
    return os << mtrand.left;
}

inline std::istream& operator>>( std::istream& is, MTRand& mtrand ) {
    register MTRand::uint32 *s = mtrand.state;
    register int i = mtrand.N;
    for( ; i--; is >> *s++ ) {}
    is >> mtrand.left;
    mtrand.pNext = &mtrand.state[mtrand.N-mtrand.left];
    return is;
}

inline MTRand& MTRand::operator=( const MTRand& o ) {
    if( this == &o ) return (*this);
    register const uint32 *t = o.state;
    register uint32 *s = state;
    register int i = N;
    for( ; i--; *s++ = *t++ ) {}
    left = o.left;
    pNext = &state[N-left];
    return (*this);
}

//=====================================================================================================================//

class Population {
    
    template< class Decoder, class RNG >
    friend class NDSBRKGA;
    friend class Decoder;

    public:
    
            unsigned getN() const;    // Size of each chromosome
            unsigned getP() const;    // Size of population

            //double operator()(unsigned i, unsigned j) const;    // Direct access to allele j of chromosome i

            // These methods REQUIRE fitness to be sorted, and thus a call to sortFitness() beforehand
            // (this is done by NDSBRKGA, so rest assured: everything will work just fine with NDSBRKGA).
            const std::pair < std::vector < double >, std::vector < double > >& getChromosome(unsigned i) const;         // Returns i-th best chromosome
            
            void saveReferenceSolutionSet(ofstream &fout, bool printsol) const;
            
            void storeReferenceSolutionSet(vector < pair < double, double > > &inputNDS) const;
            
            void clear();
            
    private:
        
            Population();
            Population(const Population& other);
            Population(unsigned n, unsigned p);
            ~Population();
            
            std::vector < std::pair < std::vector < double >, std::vector < double > > > population;                    // Population as vectors of prob.
            
            std::vector < std::pair < std::pair < unsigned, double >, unsigned > > fitness;    // Fitness < < level, crowding_distance > , id_chromosome_position > >

            int getRelation(unsigned p, unsigned q);              
            bool equalsInDesignSpace(unsigned p, unsigned q);     
            
            void sortFitness(unsigned eliteSize);                                                // Sorts 'fitness' by its first parameter
                    
            std::pair < std::vector < double >, std::vector < double > >& getChromosome(unsigned i);    // Returns a chromosome

            double& operator()(unsigned i, unsigned j);                                               // Direct access to allele j of chromosome i
            std::pair < std::vector < double >, std::vector < double > > & operator()(unsigned i);    // Direct access to chromosome i
};

inline Population::Population() { }

inline Population::Population(const Population& pop) : population(pop.population), fitness(pop.fitness) { }

inline Population::Population(const unsigned n, const unsigned p) : population(p, std::make_pair(std::vector < double >(n, 0.0), std::vector < double > ())), fitness(p) {
    if(p == 0) { std::clog << "Population size p cannot be zero." << endl; exit(0); }
    if(n == 0) { std::clog << "Chromosome size n cannot be zero." << endl; exit(0); }
}

inline Population::~Population() { }

inline unsigned Population::getN() const {
    return population[0].first.size();
}

inline unsigned Population::getP() const {
    return population.size();
}

inline const std::pair < std::vector < double >, std::vector < double > >& Population::getChromosome(unsigned i) const {
    return population[ fitness[i].second ];
}

inline std::pair < std::vector < double >, std::vector < double > >& Population::getChromosome(unsigned i) {
    return population[ fitness[i].second ];
}

inline void Population::saveReferenceSolutionSet(ofstream &fout, bool printsol) const {
    
    std::vector < std::pair < double, double > > vt;
    
    for(unsigned i = 0; i < population.size(); ++i) {        
        if(fitness[i].first.first != 0) break;
        if(printsol) Decoder::getInstance().save( population[ fitness[i].second ], fout );
        else {
            //for(unsigned j = 0; j < population[0].second.size(); ++j) {
                //fout << fixed << setw(40) << setprecision(20) << fabs(population[ fitness[i].second ].second[j]) << ' ';
            //}
            //fout << endl;
            vt.push_back(std::make_pair(fabs(population[ fitness[i].second ].second[0]), fabs(population[ fitness[i].second ].second[1])));
        }
    }    
    
    sort(vt.begin(), vt.end());
    
    for(unsigned i = 0; i < vt.size(); ++i) {       
        fout << fixed << setprecision(5) << vt[i].first << ' ' << fixed << setprecision(0) << vt[i].second << endl;
    }
}


inline void Population::storeReferenceSolutionSet(std::vector < pair < double, double > > &inputNDS) const {
    
    std::vector < std::pair < double, double > > vt;
    
    for(unsigned i = 0; i < population.size(); ++i) {        
        if(fitness[i].first.first != 0) break;
        inputNDS.push_back(std::make_pair(population[ fitness[i].second ].second[0], population[ fitness[i].second ].second[1]));
    }
}

inline void Population::clear() {
    
    std::vector < std::pair < std::vector < double >, std::vector < double > > >().swap(population);
    std::vector < std::pair < std::pair < unsigned, double >, unsigned > >().swap(fitness);   
}

inline int Population::getRelation(unsigned p, unsigned q) {
    
    /*
    
    if p dominates q:
        return 1 
    else if q dominates p:
        return -1
    else:
        return 0     
    
    */

    int val = 0;
    for(unsigned i = 0; i < population[p].second.size(); ++i) {
        if(population[p].second[i] + EPS < population[q].second[i]) {
            if(val == -1) return 0;
            val = 1;
        } 
        else if(population[p].second[i] - EPS > population[q].second[i]) {
            if (val == 1) return 0;
            val = -1;
        }
    }
    return val;
}

inline bool Population::equalsInDesignSpace(unsigned p, unsigned q) {

    for(unsigned i = 0; i < population[p].second.size(); ++i) {
        if(fabs(population[p].second[i] - population[q].second[i]) > EPS) {
            return false;
        }
    }
    
    return true;
    // return Decoder::getInstance().compare(population[p], population[q]);
}

inline void Population::sortFitness(unsigned eliteSize) {

    std::vector < std::vector < unsigned > > graph; // (p -> q) if p dominates q
    graph.resize(getP());
    
    std::vector < int > degree(getP(), 0);
    
    std::vector < std::vector < unsigned > > levels;
    levels.push_back(std::vector < unsigned > ());
    
    unsigned levelId = 0;
    unsigned numberOfObjectives = population[0].second.size();
    
    std::vector < bool > removed(getP(), false);    
    
    for(unsigned p = 0; p < getP(); ++p) {
        
        if(removed[p]) continue;
        
        for(unsigned q = p + 1; q < getP(); ++q) {
            
            if(removed[q]) continue;
            
            int relation = getRelation(p, q);
            
            if(relation == 1) {
                graph[p].push_back(q);
                degree[q] += 1;
            } 
            else if(relation == -1) {
                graph[q].push_back(p);
                degree[p] += 1;
            }
            else if(equalsInDesignSpace(p, q)) {
                removed[q] = true;
            }
        }
    }
    
    for(unsigned p = 0; p < getP(); ++p) {
        if(removed[p]) continue;
        if(degree[p] == 0) {
            levels[levelId].push_back(p);
        }
    }  
           
    while(1) {
        
        levels.push_back(std::vector < unsigned > ());
        
        for(unsigned i = 0; i < levels[levelId].size(); ++i) {           
            unsigned p = levels[levelId][i];            
            for(unsigned j = 0; j < graph[p].size(); ++j) {            
                unsigned q = graph[p][j];                
                degree[q] -= 1;                
                if(degree[q] == 0) {
                    levels[levelId + 1].push_back(q);
                }
            }
        }
        
        if(levels.back().size() == 0) {
            levels.erase(--levels.end());
            break;
        }
        
        levelId += 1;        
    }
    
    for(unsigned p = 0; p < getP(); ++p) {
        fitness[p].second = p;
        fitness[p].first.second = -population[p].second[0];
        fitness[p].first.first = INF;
    }
    
    for(unsigned i = 0; i < levels.size(); ++i) {
        for(unsigned j = 0; j < levels[i].size(); ++j) {
            fitness[ levels[i][j] ].first.first = i;            
        }
    }
    
    unsigned total = 0;
    for(unsigned i = 0; i < levels.size(); ++i) {
        if(total < eliteSize && total + levels[i].size() > eliteSize) {
            for(unsigned j = 0; j < levels[i].size(); ++j) {
                fitness[ levels[i][j] ].first.second = 0.0;            
            }
            for(unsigned k = 0; k < numberOfObjectives; ++k) {  
                std::vector < std::pair < double, unsigned > > vt;
                for(unsigned j = 0; j < levels[i].size(); ++j) {
                    vt.push_back(std::make_pair(population[ levels[i][j] ].second[k], levels[i][j]));
                }
                sort(vt.begin(), vt.end());
                
                double fmin = vt.front().first;                
                double fmax = vt.back().first;    
                                
                fitness[ vt.front().second ].first.second = INF;
                fitness[ vt.back().second  ].first.second = INF;
                
                for(unsigned l = 1; l < vt.size() - 1; ++l) {
                    fitness[ vt[l].second ].first.second += (population[ vt[l + 1].second ].second[k] - population[ vt[l - 1].second ].second[k]) / (fmax - fmin);                    
                }                
            }
            break;
        }
        else {
            total += levels[i].size();
        }
    }
    
    sort(fitness.begin(), fitness.end(), [](const std::pair < std::pair < unsigned, double >, unsigned > &a, const std::pair < std::pair < unsigned, double >, unsigned > &b) { 
        if(a.first.first < b.first.first) return true;
        else if(a.first.first == b.first.first && a.first.second > b.first.second) return true;
        return false;});
}

//inline double Population::operator()(unsigned chromosome, unsigned allele) const {
//    return population[chromosome][allele];
//}

inline double& Population::operator()(unsigned chromosome, unsigned allele) {
    return population[chromosome].first[allele];
}

inline std::pair < std::vector < double >, std::vector < double > >& Population::operator()(unsigned chromosome) {
    return population[chromosome];
}

//=====================================================================================================================//

template< class Decoder, class RNG >
class NDSBRKGA {
    
    public:
        /*
        * Default constructor
        * Required hyperparameters:
        * - n: number of genes in each chromosome
        * - p: number of elements in each population
        * - pe: pct of elite items into each population
        * - pm: pct of mutants introduced at each generation into the population
        * - rhoe: probability that an offspring inherits the allele of its elite parent
        *
        * Optional parameters:
        * - K: number of independent Populations
        * - MAX_THREADS: number of threads to perform parallel decoding
        *                WARNING: Decoder::decode() MUST be thread-safe; safe if implemented as
        *                + double Decoder::decode(std::vector < double >& chromosome) const
        */
        NDSBRKGA(unsigned n, unsigned p, double pe, double pm, double rhoe, double alpha, std::string tspSolutionFileName, std::string kpSolutionFileName, RNG& refRNG, unsigned K = 1, unsigned MAX_THREADS = 1);

        /**
        * Destructor
        */
        ~NDSBRKGA();

        /**
        * Resets all populations with brand new keys
        */
        void reset();

        /**
        * Evolve the current populations following the guidelines of NDSBRKGAs
        * @param generations number of generations (must be even and nonzero)
        * @param J interval to exchange elite chromosomes (must be even; 0 ==> no synchronization)
        * @param M number of elite chromosomes to select from each population in order to exchange
        */
        void evolve(unsigned generations = 1);
        
        void localSearch();

        /**
        * Exchange elite-solutions between the populations
        * @param M number of elite chromosomes to select from each population
        */
        void exchangeElite(unsigned M);

        /**
        * Returns the current population
        */
        const Population& getPopulation(unsigned k = 0) const;
                
        void saveReferenceSolutionSet(ofstream &fout, bool printsol = true) const;
        
        void storeReferenceSolutionSet(std::vector < pair < double, double > > &inputNDS) const;
        

        // Return copies to the internal parameters:
        unsigned getN() const;
        unsigned getP() const;
        unsigned getPe() const;
        unsigned getPm() const;
        unsigned getPo() const;
        double getRhoe() const;
        double getAlpha() const;
        
        unsigned getK() const;
        unsigned getMAX_THREADS() const;

    private:

        // Hyperparameters:
        const unsigned n;    // number of genes in the chromosome
        const unsigned p;    // number of elements in the population
        const unsigned pe;    // number of elite items in the population
        const unsigned pm;    // number of mutants introduced at each generation into the population
        const double rhoe;    // probability that an offspring inherits the allele of its elite parent
        const double alpha;   // fraction of the initial population created from TSP and KP solutions
 
        const string tspSolutionFileName; 
        const string kpSolutionFileName;         

        // Templates:
        RNG& refRNG;                  // reference to the random number generator
       
        // Parallel populations parameters:
        const unsigned K;                  // number of independent parallel populations
        const unsigned MAX_THREADS;        // number of threads for parallel decoding

        // Data:
        std::vector < Population* > previous;       // previous populations
        std::vector < Population* > current;        // current populations

        // Local operations:
        void initialize(const unsigned i);        // initialize current population 'i' with random keys
        void evolution(Population& curr, Population& next, const unsigned k);
};

template< class Decoder, class RNG >
NDSBRKGA< Decoder, RNG >::NDSBRKGA(unsigned _n, unsigned _p, double _pe, double _pm, double _rhoe, double _alpha, std::string _tspSolutionFileName, std::string _kpSolutionFileName, RNG& rng, unsigned _K, unsigned MAX) : n(_n), p(_p), pe(unsigned(_pe * p)), pm(unsigned(_pm * p)), rhoe(_rhoe), alpha(_alpha), tspSolutionFileName(_tspSolutionFileName), kpSolutionFileName(_kpSolutionFileName), refRNG(rng), K(_K), MAX_THREADS(MAX), previous(K, 0), current(K, 0) {

    // Error check:
    //using std::range_error;
    if(n == 0) { std::clog << "Chromosome size equals zero." << endl; exit(0); }
    if(p == 0) { std::clog << "Population size equals zero." << endl; exit(0); }
    if(pe == 0) { std::clog << "Elite-set size equals zero." << endl; exit(0); }
    if(pe > p) { std::clog << "Elite-set size greater than population size (pe > p)." << endl; exit(0); }
    if(pm > p) { std::clog << "Mutant-set size (pm) greater than population size (p)." << endl; exit(0); }
    if(pe + pm > p) { std::clog << "elite + mutant sets greater than population size (p)." << endl; exit(0); }
    if(K == 0) { std::clog << "Number of parallel populations cannot be zero." << endl; exit(0); }

    // Initialize and decode each chromosome of the current population, then copy to previous:
    for(unsigned i = 0; i < K; ++i) {

        // Allocate:
        current[i] = new Population(n, p);

        // Initialize:
        initialize(i);

        // Then just copy to previous:
        previous[i] = new Population(*current[i]);
    }
}

template< class Decoder, class RNG >
NDSBRKGA< Decoder, RNG >::~NDSBRKGA() {
    for(unsigned i = 0; i < K; ++i) { delete current[i]; delete previous[i]; }
}

template< class Decoder, class RNG >
const Population& NDSBRKGA< Decoder, RNG >::getPopulation(unsigned k) const {
    return (*current[k]);
}

template< class Decoder, class RNG >
void NDSBRKGA< Decoder, RNG >::reset() {
    for(unsigned i = 0; i < K; ++i) { initialize(i); }
}

template< class Decoder, class RNG >
void NDSBRKGA< Decoder, RNG >::evolve(unsigned generations) {
    if(generations == 0) { std::clog << "Cannot evolve for 0 generations." << endl; exit(0); }

    for(unsigned i = 0; i < generations; ++i) {
        for(unsigned j = 0; j < K; ++j) {
            evolution(*current[j], *previous[j], j);    // First evolve the population (curr, next)
            std::swap(current[j], previous[j]);        // Update (prev = curr; curr = prev == next)
        }
    }
}

template< class Decoder, class RNG >
void NDSBRKGA< Decoder, RNG >::exchangeElite(unsigned M) {
    
    if(M == 0 || M >= p) { std::clog << "M cannot be zero or >= p." << endl; exit(0); }

    for(unsigned i = 0; i < K; ++i) {
        
        // Population i will receive some elite members from each Population j below:
        
        unsigned dest = p - 1;    // Last chromosome of i (will be updated below)
        
        for(unsigned j = 0; j < K; ++j) {
            if(j == i) { continue; }

            // Copy the M individual from Population j to Population i:
            for(unsigned _m = 0; _m < M; ++_m) {
                
                unsigned m = (refRNG.randInt(pe - 1));
                
                const std::pair < std::vector < double >, std::vector < double > >& bestOfJ = current[j]->getChromosome(m);
                
                current[i]->population[dest] = bestOfJ;
                
                current[i]->fitness[dest].second = current[j]->fitness[m].second;
                
                --dest;
            }
        }
    }

    for(int j = 0; j < int(K); ++j) { current[j]->sortFitness(getPe()); }
}

template< class Decoder, class RNG >
inline void NDSBRKGA< Decoder, RNG >::initialize(const unsigned k) {
    
    vector < int > tour, stolenItems;

    //========================================================================================//
    // read tsp solution found by LKH
    
    ifstream fin(tspSolutionFileName.c_str());

    if(!fin) {
        clog << "ERROR!" << endl;
        clog << tspSolutionFileName << endl;
        exit(0);
    }
            
    string line;
    getline(fin, line); // NAME : a280.2613.tour
    getline(fin, line); // COMMENT : Length = 2613
    getline(fin, line); // COMMENT : Found by LKH [Keld Helsgaun] Thu Jan 17 13:00:40 2019
    getline(fin, line); // TYPE : TOUR
    getline(fin, line); // DIMENSION : 280
    getline(fin, line); // TOUR_SECTION
    
    int tmp;    
    while(1) {
        fin >> tmp;
        if(tmp == -1) break;
        tour.push_back(tmp);
    }
    fin.close();
    //========================================================================================//
    
    //========================================================================================//
    // read kp solution found our greedy heuristic + dynamic programming
    
    fin.open(kpSolutionFileName.c_str());
    if(!fin) {
        clog << "ERROR!" << endl;
        clog << kpSolutionFileName << endl;
        exit(0);
    }
    
    fin >> tmp; // total profit
    while(fin >> tmp) {
        stolenItems.push_back(tmp);
    }
    fin.close();
    //========================================================================================//
    
    std::vector < std::vector < double > > chromosomes;
            
    vector < double > chromosome1(Data::getInstance().numCities + Data::getInstance().numItems, 0.0);
    vector < double > chromosome2(Data::getInstance().numCities + Data::getInstance().numItems, 0.0);
    
    double _delta = 1.0/(int)tour.size();
    
    double key = 0.0;    
    for (unsigned i = 1; i < tour.size(); i++) {
        chromosome1[ tour[i] - 2 ] = key;
        key += _delta;
    }

    key = 0.0;
    for (unsigned i = tour.size() - 1; i >= 1; i--) {
        chromosome2[ tour[i] - 2 ] = key;
        key += _delta;
    }
    
    vector < int > cityPositions(Data::getInstance().numCities+1, 0);
    for(int i = 0; i < (int)tour.size(); ++i) {
        cityPositions[tour[i]] = i;
    }
    
    vector < pair < double, int > > orderItems;
    
    for(int i = 0; i < (int)stolenItems.size(); ++i) {    
        double weight = Data::getInstance().items[stolenItems[i]].weight;
        double profit = Data::getInstance().items[stolenItems[i]].profit;
        orderItems.push_back(make_pair(-profit/weight, stolenItems[i]));
    }
    
    sort(orderItems.begin(), orderItems.end());
    
    vector < int > indices;
    
    std::default_random_engine generator(refRNG.randInt(1000000));
    
    std::uniform_int_distribution < int > distribution(0, (int)orderItems.size()-2);
       
    if(getP() * getAlpha() >= 1) {
        indices.push_back(-1);
    }   
    
    if(getP() * getAlpha() >= 2) {
        indices.push_back((int)orderItems.size()-1);
    }   
    
    if(getP() * getAlpha() >= 3) {
        indices.push_back(-2);
    }   
         
    while((int)indices.size() < getP() * getAlpha()) {
        indices.push_back(distribution(generator));
    }   
    
    std::set < int > st(indices.begin(), indices.end());
    
    if(st.find(-2) != st.end()) chromosomes.push_back(chromosome1);
    if(st.find(-1) != st.end()) chromosomes.push_back(chromosome2);
    
    for(int i = 0; i < (int)orderItems.size(); ++i) {
        chromosome1[Data::getInstance().numCities - 1 + orderItems[i].second - 1] = 1.0;
        chromosome2[Data::getInstance().numCities - 1 + orderItems[i].second - 1] = 1.0;
        
        if(st.find(i) != st.end()) {
            
            std::pair < std::vector < double >, std::vector < double > > _chromosome1;
            _chromosome1.first = chromosome1;
            
            std::pair < std::vector < double >, std::vector < double > > _chromosome2;
            _chromosome2.first = chromosome2;        
            
            Decoder::getInstance().decode(_chromosome1);
            Decoder::getInstance().decode(_chromosome2);        
            
            if(_chromosome1.second[0] < _chromosome2.second[0]) chromosomes.push_back(chromosome1);
            else chromosomes.push_back(chromosome2);
        }
    }    
    
    unsigned pos = 0;
    for(unsigned i = 0; i < (int)chromosomes.size(); ++i) { 
        for(unsigned j = 0; j < n; ++j) { (*current[k])(pos, j) = chromosomes[i][j]; }
        pos += 1;
    }
    for(; pos < getP(); ++pos) {        
        for(unsigned j = 0; j < n; ++j) { (*current[k])(pos, j) = refRNG.rand(); }
    }
    
    // Decode:
    #ifdef _OPENMP
        #pragma omp parallel for num_threads(MAX_THREADS)
    #endif
    for(int j = 0; j < int(p); ++j) {
        Decoder::getInstance().decode((*current[k])(j));
    }
    
    current[k]->sortFitness(getPe());
}

template< class Decoder, class RNG >
inline void NDSBRKGA< Decoder, RNG >::saveReferenceSolutionSet(ofstream &fout, bool printsol) const {
     
    for(unsigned k = 0; k < K; ++k) {
        current[k]->saveReferenceSolutionSet(fout, printsol);  
    }
}

template< class Decoder, class RNG >
inline void NDSBRKGA< Decoder, RNG >::storeReferenceSolutionSet(std::vector < pair < double, double > > &inputNDS) const {
     
    for(unsigned k = 0; k < K; ++k) {
        current[k]->storeReferenceSolutionSet(inputNDS);  
    }
}

template< class Decoder, class RNG >
inline void NDSBRKGA< Decoder, RNG >::evolution(Population& curr, Population& next, const unsigned k) {
    
    //clog << "K: " << k << endl;
    
    // We now will set every chromosome of 'current', iterating with 'i':
    unsigned i = 0;    // Iterate chromosome by chromosome
    unsigned j = 0;    // Iterate allele by allele

    // 2. The 'pe' best chromosomes are maintained, so we just copy these into 'current':
    while(i < pe) {
        next(i) = curr(curr.fitness[i].second);        
        next.fitness[i].first = curr.fitness[i].first;
        next.fitness[i].second = i;       
        ++i;
    }

    // 3. We'll mate 'p - pe - pm' pairs; initially, i = pe, so we need to iterate until i < p - pm:
    while(i < p - pm) {
        
        unsigned eliteParent, anotherParent;
        
        do {
            // Select an elite parent:
            eliteParent = (refRNG.randInt(pe - 1));

            // Select an elite or non-elite parent:
            anotherParent = (refRNG.randInt(p - 1));
            
        } while(eliteParent == anotherParent);

        // Mate:
        for(j = 0; j < n; ++j) {
            const unsigned sourceParent = ((refRNG.rand() < rhoe) ? eliteParent : anotherParent);

            next(i, j) = curr(curr.fitness[sourceParent].second, j);

            //next(i, j) = (refRNG.rand() < rhoe) ? curr(curr.fitness[eliteParent].second, j) :
            //                                      curr(curr.fitness[anotherParent].second, j);
        }

        ++i;
    }

    /*
    while(i < p - pm) {
    
        int parent1, parent2, a, b;
        
        a = (refRNG.randInt(pe - 1));
        do {
            b = (refRNG.randInt(p - 1));
        } while (a == b);

        if(a < pe && b >= pe) parent1 = a;
        else if(b < pe && a >= pe) parent1 = b;
        else parent1 = ((refRNG.rand() < 0.5) ? a : b);

        a = (refRNG.randInt(p - 1));
        do {
            b = (refRNG.randInt(p - 1));
        } while (a == b);

        if(a < pe && b >= pe) parent2 = a;
        else if(b < pe && a >= pe) parent2 = b;
        else if(refRNG.rand() < 0.5) parent2 = a;
        else parent2 = b;
        
        // Mate:
        for(j = 0; j < n; ++j) {
            const unsigned sourceParent = ((refRNG.rand() < 0.5) ? parent1 : parent2);

            next(i, j) = curr(curr.fitness[sourceParent].second, j);

            //next(i, j) = (refRNG.rand() < rhoe) ? curr(curr.fitness[eliteParent].second, j) :
            //                                      curr(curr.fitness[anotherParent].second, j);
        }
        
        ++i;
    }
    */
    // We'll introduce 'pm' mutants:
    while(i < p) {
        for(j = 0; j < n; ++j) { next(i, j) = refRNG.rand(); }
        ++i;
    }
    
    // Time to compute fitness, in parallel:
    #ifdef _OPENMP
        #pragma omp parallel for num_threads(MAX_THREADS)
    #endif
    for(int i = int(pe); i < int(p); ++i) {
        Decoder::getInstance().decode(next.population[i]);
    }

    // Now we must sort 'current' by fitness, since things might have changed:
    next.sortFitness(getPe());
}

template< class Decoder, class RNG >
inline void NDSBRKGA< Decoder, RNG >::localSearch() {
     
    std::default_random_engine generator(refRNG.randInt(1000000));
    
    std::uniform_int_distribution < int > distribution(0, getPe()-1);
    
    for(unsigned k = 0; k < K; ++k) {
        
        Population* all = new Population();
        
        std::set < int > st;
        while((int)st.size() < getPe() * 0.1) {
            st.insert(distribution(generator));
        }
        
        std::set < int > :: iterator it = st.begin();
        for(; it != st.end(); ++it) {
            unsigned i = *it;        
        //for(unsigned i = 0; i < getPe(); ++i) {
            Decoder::getInstance().localSearch(current[k]->population[i], all);
        }   
        for(unsigned i = 0; i < getPe(); ++i) {
            all->population.push_back(current[k]->population[i]);
            //clog << i + 1 << ' ' << current[k]->population[i].second[1] << endl;
        }
        
        /*
        for(unsigned i = 0; i < all->population.size(); ++i) {
            clog << i + 1 << ' ' << all->population.size() << ' ' << all->population[i].second[1] << endl;
        }  
        */
        
        all->fitness.resize(all->population.size());   
        //clog << "sortFitness()" << endl;
        //clog << all->population.size() << endl;
        all->sortFitness(getPe());       

        for(unsigned i = 0; i < getPe(); ++i) {
            current[k]->population[i] = all->population[all->fitness[i].second];
            current[k]->fitness[i].second = i;
        }
        
        all->clear();    
        delete all;
    }
}

template< class Decoder, class RNG >
unsigned NDSBRKGA<Decoder, RNG>::getN() const { return n; }

template< class Decoder, class RNG >
unsigned NDSBRKGA<Decoder, RNG>::getP() const { return p; }

template< class Decoder, class RNG >
unsigned NDSBRKGA<Decoder, RNG>::getPe() const { return pe; }

template< class Decoder, class RNG >
unsigned NDSBRKGA<Decoder, RNG>::getPm() const { return pm; }

template< class Decoder, class RNG >
unsigned NDSBRKGA<Decoder, RNG>::getPo() const { return p - pe - pm; }

template< class Decoder, class RNG >
double NDSBRKGA<Decoder, RNG>::getRhoe() const { return rhoe; }

template< class Decoder, class RNG >
double NDSBRKGA<Decoder, RNG>::getAlpha() const { return alpha; }

template< class Decoder, class RNG >
unsigned NDSBRKGA<Decoder, RNG>::getK() const { return K; }

template< class Decoder, class RNG >
unsigned NDSBRKGA<Decoder, RNG>::getMAX_THREADS() const { return MAX_THREADS; }

#endif
