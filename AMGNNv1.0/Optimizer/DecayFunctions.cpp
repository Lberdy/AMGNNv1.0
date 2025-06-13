#include<cmath>

class TimeBasedDecay{

public:
    float decayRate = 0.005;

    float calculateLR(float LR, int epoch){

        return LR/(1+decayRate*epoch);

    }

    void save(std::ofstream& fileOut){
        fileOut.write(reinterpret_cast<char*>(&decayRate), sizeof(float));
    }

    void load(std::ifstream& fileIn){
        fileIn.read(reinterpret_cast<char*>(&decayRate), sizeof(float));
    }

};

class StepDecay{

public:
    float dropRate = 0.5;
    int dropEvery = 10;

    float calculateLR(float LR, int epoch){

        return LR*pow(dropRate, floor((float)epoch/dropEvery));

    }

    void save(std::ofstream& fileOut){
        fileOut.write(reinterpret_cast<char*>(&dropRate), sizeof(float));
        fileOut.write(reinterpret_cast<char*>(&dropEvery), sizeof(int));
    }

    void load(std::ifstream& fileIn){
        fileIn.read(reinterpret_cast<char*>(&dropRate), sizeof(float));
        fileIn.read(reinterpret_cast<char*>(&dropEvery), sizeof(int));
    }

};

class ExponentialDecay{

public:
    float decayRate = 0.05;

    float calculateLR(float LR, int epoch){

        return LR*exp(-decayRate*epoch);

    }

    void save(std::ofstream& fileOut){
        fileOut.write(reinterpret_cast<char*>(&decayRate), sizeof(float));
    }

    void load(std::ifstream& fileIn){
        fileIn.read(reinterpret_cast<char*>(&decayRate), sizeof(float));
    }

};

class PolynomialDecay{

public:
    float finalLR = 0.00001;
    float power = 2;

    float calculateLR(float LR, int epoch, int maxEpoch){

        return (LR - finalLR)*pow((1 - (float)epoch/maxEpoch), power) + finalLR;

    }

    void save(std::ofstream& fileOut){
        fileOut.write(reinterpret_cast<char*>(&finalLR), sizeof(float));
        fileOut.write(reinterpret_cast<char*>(&power), sizeof(float));
    }

    void load(std::ifstream& fileIn){
        fileIn.read(reinterpret_cast<char*>(&finalLR), sizeof(float));
        fileIn.read(reinterpret_cast<char*>(&power), sizeof(float));
    }

};

class CosineAnnealingDecay{

public:
    float minLR = 0.00001;

    float calculateLR(float LR ,int epoch, int maxEpoch){

        return minLR + 0.5*(LR - minLR)*(1 + cos(M_PI*(float)epoch/maxEpoch));

    }

    void save(std::ofstream& fileOut){
        fileOut.write(reinterpret_cast<char*>(&minLR), sizeof(float));
    }

    void load(std::ifstream& fileIn){
        fileIn.read(reinterpret_cast<char*>(&minLR), sizeof(float));
    }

};