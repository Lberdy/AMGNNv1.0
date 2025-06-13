#include<vector>
#include<random>
#include<cmath>
#include<fstream>
#include<iostream>

class Matrix{

public:

    static std::vector<float> zeros(int e){

        std::vector<float> zero;
        for(int i = 1; i <= e; i++){
            zero.push_back(0);
        }

        return zero;

    }


    static std::vector<float> ones(int e){

        std::vector<float> one;
        for(int i = 1; i <= e; i++){
            one.push_back(1);
        }

        return one;

    }


    static std::vector<float> Xavier(int fanIn, int fanOut){

        float limit = sqrt(6.0/(fanIn+fanOut));
        std::uniform_real_distribution<float> destribution(-limit,limit);
        std::random_device rd;
        std::default_random_engine engine(rd());

        std::vector<float> xavier;
        for(int i = 1; i <= fanOut*fanIn; i++){
            xavier.push_back(destribution(engine));
        }

        return xavier;

    }

    static std::vector<float> Kaiming(int fanIn, int fanOut){

        float limit = sqrt(6.0/(fanIn));
        
        std::uniform_real_distribution<float> destribution(-limit,limit);
        std::random_device rd;
        std::default_random_engine engine(rd());

        std::vector<float> kaiming;
        for(int i = 1; i <= fanOut*fanIn; i++){
            kaiming.push_back(destribution(engine));
        }

        return kaiming;

    }

    static void dot(std::vector<float> mat1, std::vector<float> mat2, std::vector<float>& values){

        int i = 0;
        int index = 0;
        int size = mat2.size();
        float sum = 0;
        while(i < mat1.size()){
            sum += mat1[i]*mat2[i%size];
            i++;
            if(i % size == 0){
                values[index] = sum;
                index++;
                sum = 0;
            }
        }

    }

};