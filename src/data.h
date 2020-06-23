#ifndef DATA_H
#define DATA_H

#include <vector>
#include <string>

using namespace std;

struct Item {
    
    int id;
    int profit;
    int weight;
    int idCity;
        
    Item(int _id, int _profit, int _weight, int _idCity) : id(_id), profit(_profit), weight(_weight), idCity(_idCity) {}
};

class Data {
    
    public:
    
            static Data& getInstance() {
                static Data instance;                                        
                return instance;
            }
            
            Data(Data const&)            = delete;
            void operator=(Data const&)  = delete;
            
            string problemName;
            string knapsackDataType;            
            int numCities;
            int numItems;
            int capacityOfKnapsack;
            double v, minSpeed, maxSpeed;
            double rentingRatio;
            vector < pair < double, double > > vertex;
            
            int totalProfit;
            
            vector < Item > items;

            void readData(string);
            int getDistance(int, int);
            
    private:
        
            Data() {};
};

#endif