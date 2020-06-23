#include "ndsbrkga.h"
#include "data.h"

#include <chrono>

default_random_engine Decoder::generator(112358);

void Decoder::repairOperator(std::pair < std::vector < double >, std::vector < double > > &chromosome, std::vector < std::pair < double, int > > &ranking, std::vector < int > &preSelectedItems, std::vector < int > &weight, std::vector < int > &profit, int capacity) const { // clog << "repairOperator1" << endl;
        
    int n = Data::getInstance().numCities;
    int m = Data::getInstance().numItems;
    
    int totalWeight, idItem, currCity;
    
    std::uniform_real_distribution < double > distribution(0.0, 1.0);
    
    totalWeight = 0;
    for(int k = 0; k < (int)preSelectedItems.size(); ++k) {        
        idItem = preSelectedItems[k];
        totalWeight += Data::getInstance().items[idItem].weight;
    }

    capacity = capacity * distribution(generator);

    vector < bool > cityWithoutItems(n + 1, false); 
    
    int pos = 0;    
    while(totalWeight > capacity) {
        currCity = ranking[pos].second;
        totalWeight -= weight[currCity];  
        cityWithoutItems[currCity] = true;        
        weight[currCity] = profit[currCity] = 0;
        pos += 1;
    }
    
    for(int k = 0; k < (int)preSelectedItems.size(); ++k) {        
        idItem = preSelectedItems[k];
        if(cityWithoutItems[Data::getInstance().items[idItem].idCity]) {
            chromosome.first[n - 1 + idItem - 1] = 0.0;
        }
    }
}

void Decoder::decode(std::pair < std::vector < double >, std::vector < double > > &chromosome) {

    int n = Data::getInstance().numCities;
    int m = Data::getInstance().numItems;
    
    std::vector < std::pair < double, int > > ranking(n - 1);
    
    for(int i = 0; i < n - 1; ++i) {
        ranking[i] = std::make_pair(chromosome.first[i], i + 2);
    }

    std::sort(ranking.begin(), ranking.end());
    ranking.push_back(make_pair(1.0, 1)); // end point

    std::vector < int > weight(n + 1, 0.0);
    std::vector < int > profit(n + 1, 0.0);    
    
    int idItem = 1;
    int totalWeight = 0;
    std::vector < int > selectedItems;
    for(int i = n - 1; i < n + m - 1; ++i) {
        if(chromosome.first[i] - EPS >= alpha) {            
            weight[Data::getInstance().items[idItem].idCity] += Data::getInstance().items[idItem].weight;
            profit[Data::getInstance().items[idItem].idCity] += Data::getInstance().items[idItem].profit;  
            totalWeight += Data::getInstance().items[idItem].weight;
            selectedItems.push_back(idItem);
        }
        idItem += 1;
    }
    
    //clog << repairOperatorID << endl;
    //clog << totalWeight << ' ' << Data::getInstance().capacityOfKnapsack << endl;
    
    if(totalWeight > Data::getInstance().capacityOfKnapsack) {
        repairOperator(chromosome, ranking, selectedItems, weight, profit, Data::getInstance().capacityOfKnapsack);
    }    
    
    totalWeight = 0;
    double totalTime = 0.0;
    int totalProfit = 0;
    int currCity, prevCity = 1;    
    
    for(int i = 0; i < n; ++i) {
        currCity = ranking[i].second;
        totalTime += Data::getInstance().getDistance(prevCity, currCity) / (Data::getInstance().maxSpeed - Data::getInstance().v * totalWeight); 
        totalWeight += weight[currCity];
        totalProfit += profit[currCity];
        prevCity = currCity;
    }   

    //clog << totalWeight << ' ' << Data::getInstance().capacityOfKnapsack << endl;
        
    chromosome.second.clear();
    chromosome.second.push_back(totalTime);
    chromosome.second.push_back(-totalProfit);
}

void Decoder::localSearch(std::pair < std::vector < double >, std::vector < double > > &chromosome, Population* allpop) {

    int n = Data::getInstance().numCities;
    int m = Data::getInstance().numItems;
    
    std::uniform_int_distribution < int > distribution1(0, n - 1);

    for(int _i = 0; _i < 100; ++_i) { 
        
        double prevTime = chromosome.second[0];
        
        int a, b;
        do {
            a = distribution1(generator);
            b = distribution1(generator);
        } while (a == b);
        
        if(a > b) std::swap(a, b);
        
        std::reverse(chromosome.first.begin() + a, chromosome.first.begin() + b);
        
        decode(chromosome); 
        
        if(chromosome.second[0] > prevTime) {
            std::reverse(chromosome.first.begin() + a, chromosome.first.begin() + b);
            chromosome.second[0] = prevTime;
        }
        //else clog << prevTime << ' ' << chromosome.second[0] << endl;
    }

    std::uniform_int_distribution < int > distribution2(n, n + m - 1 - 1);
    
    std::pair < std::vector < double >, std::vector < double > > _chromosome;
        
    for(int _i = 0; _i < 100; ++_i) { 
        
        int i = distribution2(generator);
        _chromosome = chromosome;
        
        if(chromosome.first[i] >= alpha) _chromosome.first[i] = 0.0;
        else _chromosome.first[i] = 1.0;
        
        decode(_chromosome); 
        
        bool insert = true;
        for(unsigned l = 0; l < allpop->population.size(); ++l) {
            if(allpop->population[l].second[0] + EPS <= _chromosome.second[0] && allpop->population[l].second[1] + EPS <= _chromosome.second[1]) {
                insert = false;
                break;
            }
        }            
        if(insert) allpop->population.push_back(_chromosome);
    }    
}

bool Decoder::compare(const std::pair < std::vector < double >, std::vector < double > > &chromosomeA, const std::pair < std::vector < double >, std::vector < double > > &chromosomeB) const {
 
    int n = Data::getInstance().numCities;
    int m = Data::getInstance().numItems;
    
    for(int i = n - 1; i < n + m - 1; ++i) {
       if((chromosomeA.first[i] - EPS >= alpha) != (chromosomeB.first[i] - EPS >= alpha)) return false;
    }
    
    std::vector < std::pair < double, int > > rankingA(n - 1);
    std::vector < std::pair < double, int > > rankingB(n - 1);    
    
    for(int i = 0; i < n - 1; ++i) {
        rankingA[i] = std::make_pair(chromosomeA.first[i], i + 2);
        rankingB[i] = std::make_pair(chromosomeB.first[i], i + 2);
    }

    std::sort(rankingA.begin(), rankingA.end());
    std::sort(rankingB.begin(), rankingB.end());

    for(int i = 0; i < n - 1; ++i) {
        if(rankingA[i].second != rankingB[i].second) return false;
    }
 
    return true;
}

void Decoder::save(const std::pair < std::vector < double >, std::vector < double > > &chromosome, ofstream &fout) const {

    int n = Data::getInstance().numCities;
    int m = Data::getInstance().numItems;
    
    std::vector < std::pair < double, int > > ranking(n - 1);
    
    for(int i = 0; i < n - 1; ++i) {
        ranking[i] = std::make_pair(chromosome.first[i], i + 2);
    }

    std::sort(ranking.begin(), ranking.end());

    std::vector < int > tour;
    tour.push_back(1);
    for(int i = 0; i < n - 1; ++i) {
        tour.push_back(ranking[i].second);
    }
    tour.push_back(1);
    
    std::vector < int > weight(n + 1, 0.0);
    std::vector < int > profit(n + 1, 0.0);    
    
    int idItem = 1;
    for(int i = n - 1; i < n + m - 1; ++i) {
        if(chromosome.first[i] - EPS >= alpha) {            
            weight[Data::getInstance().items[idItem].idCity] += Data::getInstance().items[idItem].weight;
            profit[Data::getInstance().items[idItem].idCity] += Data::getInstance().items[idItem].profit;                        
        }
        idItem += 1;
    }
    
    int totalWeight = 0;
    
    for(int i = tour.size() - 1; i >= 0; --i) {
        int currCity = tour[i];
        if(totalWeight + weight[currCity] <= Data::getInstance().capacityOfKnapsack + EPS) {
            totalWeight += weight[currCity];
        }
        else {
            weight[currCity] = profit[currCity] = 0;
        }
    }   
    
    totalWeight = 0;
    double totalTime = 0.0;
    int totalProfit = 0;
    
    for(int i = 1; i < tour.size(); ++i) {
        totalTime += Data::getInstance().getDistance(tour[i-1], tour[i]) / (Data::getInstance().maxSpeed - Data::getInstance().v * totalWeight); 
        totalWeight += weight[tour[i]];
        totalProfit += profit[tour[i]];
    }
    
    fout << '1';
    for(int i = 1; i < tour.size() - 1; ++i) {
        fout << ' ' << tour[i];
    }
    fout << endl;
    
    if(chromosome.first[n - 1] - EPS >= alpha) {     
        fout << '1';
    }
    else fout << '0';
    
    for(int i = n - 1 + 1; i < n + m - 1; ++i) {
        if(chromosome.first[i] - EPS >= alpha) {     
            fout << " 1";
        }
        else fout << " 0";
    }
    fout << endl << endl;
}
