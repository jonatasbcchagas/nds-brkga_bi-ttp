#include <iostream>
#include <cstdio>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <fstream>

#include "../data.h"

using namespace std;

vector < vector < double > > tab;
vector < vector < vector < int > > > items;
        
inline void DP(const string &outputFileName) {
    
    int numItems = Data::getInstance().numItems;
    int cap = Data::getInstance().capacityOfKnapsack;
    
    tab.resize(2);
    tab[0].resize(cap + 1);
    tab[1].resize(cap + 1);
    
    items.resize(2);
    items[0].resize(cap + 1);
    items[1].resize(cap + 1); 
    
    int i, j;

    for(i = 0; i <= cap; i++) {
        tab[0][i] = 0;
        items[0][i].clear();
    }

    for(i = 1; i <= numItems; i++) {
        for(j = 1; j<= cap; j++) {
            if(Data::getInstance().items[i].weight > j) {
                tab[1][j] = tab[0][j];
                items[1][j] = items[0][j];
            }
            else {
                if(tab[0][j-Data::getInstance().items[i].weight] + Data::getInstance().items[i].profit > tab[0][j]) {
                    tab[1][j] = tab[0][j-Data::getInstance().items[i].weight] + Data::getInstance().items[i].profit;
                    items[1][j] = items[0][j-Data::getInstance().items[i].weight];
                    items[1][j].push_back(i);
                }
                else {
                    tab[1][j] = tab[0][j];
                    items[1][j] = items[0][j];                    
                }
            }
        }
        tab[0].swap(tab[1]);
        items[0].swap(items[1]);
    }
    
    int totalProfit = tab[0][cap];
    vector < int > stolenItems;
    
    for(i = 0; i < items[0][cap].size(); ++i) {
        stolenItems.push_back(items[0][cap][i]);
    }
    
    sort(stolenItems.begin(), stolenItems.end());
    
    ofstream fout(outputFileName.c_str());
    
    fout << fixed << setprecision(0) << totalProfit << endl;
        
    for(i = 0; i < (int)stolenItems.size(); ++i) {
        fout << stolenItems[i] << ' ';
    }
    fout << endl;
    
    fout.close();
}

inline void HEU_DP(const string &outputFileName, int finalSlice) {
    
    int numItems = Data::getInstance().numItems;
    int cap = Data::getInstance().capacityOfKnapsack;
    
    int i, j;
    
    vector < pair < double, int > > vt;
    for(int i = 1; i <= numItems; ++i) {            
        double weight = Data::getInstance().items[i].weight;
        double profit = Data::getInstance().items[i].profit;
        int cityID = Data::getInstance().items[i].idCity;
        vt.push_back(make_pair(-profit/weight, i));
    }
    
    sort(vt.begin(), vt.end());
    
    vector < int > stolenItems;
    double totalWeight = 0.0;
    double totalProfit = 0.0;
    int pos = 0;
    
    for(int i = 0; i < (int)vt.size(); ++i) {        
        double weight = Data::getInstance().items[vt[i].second].weight;
        if(totalWeight + weight <= cap - finalSlice) {
            totalWeight += weight;
            totalProfit += Data::getInstance().items[vt[i].second].profit;
            stolenItems.push_back(vt[i].second);
        }
        else {
            pos = i;
            break;
        }
    }    
    
    cap = cap - totalWeight;    
    
    tab.resize(2);
    tab[0].resize(cap + 1);
    tab[1].resize(cap + 1);
    
    items.resize(2);
    items[0].resize(cap + 1);
    items[1].resize(cap + 1);    
    
    for(i = 0; i <= cap; i++) {
        tab[0][i] = 0;
        items[0][i].clear();
    }
      
    
    for(int k = pos; k < (int)vt.size(); k++) {
        
        i = vt[k].second;
        
        for(j = 1; j<= cap; j++) {
            if(Data::getInstance().items[i].weight > j) {
                tab[1][j] = tab[0][j];
                items[1][j] = items[0][j];
            }
            else {
                if(tab[0][j-Data::getInstance().items[i].weight] + Data::getInstance().items[i].profit > tab[0][j]) {
                    tab[1][j] = tab[0][j-Data::getInstance().items[i].weight] + Data::getInstance().items[i].profit;
                    items[1][j] = items[0][j-Data::getInstance().items[i].weight];
                    items[1][j].push_back(i);
                }
                else {
                    tab[1][j] = tab[0][j];
                    items[1][j] = items[0][j];                    
                }
            }
        }
        tab[0].swap(tab[1]);
        items[0].swap(items[1]);
    }
    
    totalProfit += tab[0][cap];
    for(i = 0; i < items[0][cap].size(); ++i) {
        stolenItems.push_back(items[0][cap][i]);
    }
    
    sort(stolenItems.begin(), stolenItems.end());
    
    ofstream fout(outputFileName.c_str());
    
    fout << fixed << setprecision(0) << totalProfit << endl;
        
    for(i = 0; i < (int)stolenItems.size(); ++i) {
        fout << stolenItems[i] << ' ';
    }
    fout << endl;
    
    fout.close();
}

int main(int argc, char **argv) {

    if(argc < 3 || argc > 4) {
        clog << "\n       Usage ./KP instance_file output_file\n\n              or\n\n             ./KP instance_file output_file final_slice " << endl << endl;
        exit(0);
    }
    
    string instanceFileName = argv[1]; 
    string outputFileName = argv[2];
    
    Data::getInstance().readData(instanceFileName); 
    
    if(argc <= 3) {
        DP(outputFileName);
    }
    else {
        int finalSlice;
        sscanf(argv[3], "%d", &finalSlice);      
        HEU_DP(outputFileName, finalSlice);
    }
    
    return 0;
}
