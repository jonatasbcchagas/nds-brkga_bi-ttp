#include "data.h"

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <cmath>

using namespace std;

void Data::readData(string file) {
    
    string line;
    stringstream ss;

    ifstream fin(file.c_str());

    if(!fin) {
        clog << "ERROR!" << endl;
        clog << file << endl;
        exit(0);
    }
    
    getline(fin, line); // PROBLEM NAME:    a280-ThOP

    getline(fin, line); // KNAPSACK DATA TYPE: bounded strongly corr

    getline(fin, line); // DIMENSION:  280
    for(unsigned j=0; j < line.length(); ++j) {
        if(line[j] < '0' || line[j] > '9') line[j] = ' ';
    }
    ss.clear();
    ss << line;
    ss >> numCities;

    getline(fin, line); // NUMBER OF ITEMS:    279
    for(unsigned j=0; j < line.length(); ++j) {
        if(line[j] < '0' || line[j] > '9') line[j] = ' ';
    }
    ss.clear();
    ss << line;
    ss >> numItems;

    getline(fin, line); // CAPACITY OF KNAPSACK:   25936
    for(unsigned j=0; j < line.length(); ++j) {
        if(line[j] < '0' || line[j] > '9') line[j] = ' ';
    }
    ss.clear();
    ss << line;
    ss >> capacityOfKnapsack;
      
    getline(fin, line); // MIN SPEED:  0.1
    for(unsigned j=0; j < line.length(); ++j) {
        if((line[j] < '0' || line[j] > '9') && line[j] != '.') line[j] = ' ';
    }
    ss.clear();
    ss << line;
    ss >> minSpeed;
    
    getline(fin, line); // MAX SPEED:  1
    for(unsigned j=0; j < line.length(); ++j) {
        if((line[j] < '0' || line[j] > '9') && line[j] != '.') line[j] = ' ';
    }
    ss.clear();
    ss << line;
    ss >> maxSpeed;
   
    getline(fin, line); // RENTING RATIO:  5.61
    for(unsigned j=0; j < line.length(); ++j) {
        if((line[j] < '0' || line[j] > '9') && line[j] != '.') line[j] = ' ';
    }
    ss.clear();
    ss << line;
    ss >> rentingRatio;
   
    getline(fin, line); // EDGE_WEIGHT_TYPE:   CEIL_2D
    
    getline(fin, line); // NODE_COORD_SECTION  (INDEX, X, Y): 
    
    v = (maxSpeed - minSpeed)/capacityOfKnapsack;
    
    vertex.resize(numCities+1);
    
    unsigned id;
    for(unsigned i = 1; i <= numCities; ++i) {
        fin >> id;
        fin >> vertex[i].first;
        fin >> vertex[i].second;
    }
    
    getline(fin, line); // '\n'
    getline(fin, line); // ITEMS SECTION    (INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER):
    
    int id_city;
    double p, w;

    items.push_back(Item(0, -987654321, 987654321, -1));
    totalProfit = 0;
    for(unsigned i = 1; i <= numItems; ++i) {
        fin >> id >> p >> w >> id_city;
        items.push_back(Item(id, p, w, id_city));
        totalProfit += p;
    }

    fin.close();
}

int Data::getDistance(int i, int j) {
    
    return (int) ceil(   
                        sqrt(
                            ((vertex[i].first-vertex[j].first)*(vertex[i].first-vertex[j].first)) +
                            ((vertex[i].second-vertex[j].second)*(vertex[i].second-vertex[j].second))
                        )
                     );
}
    