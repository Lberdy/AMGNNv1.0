#include"../Differentiation.cpp"
#include<mutex>

std::mutex adamMutex;

class Adam{

    int weightNumber = 0;
    int weightOptimized = 0;
    int currentStep = 1;
    std::vector<float> Moment1;
    std::vector<float> Moment2;

public:
    float B1 = 0.9;
    float B2 = 0.999;
    float epsilon = 0.000001;
    float cachedB1 = B1;
    float cachedB2 = B2;

    Adam(const NN& nn){

        for(const HiddenLayer& layer:nn.HiddenLayers){
            weightNumber += layer.InWeights.size();
            weightNumber += layer.biases.size();
        }

        weightNumber += nn.outputLayer.InWeights.size();
        weightNumber += nn.outputLayer.biases.size();

        Moment1 = Matrix::zeros(weightNumber);
        Moment2 = Matrix::zeros(weightNumber);

    }

    Adam(){}

    float optimizeWeight(float weight, float gradient, float LR, int num){

        float M = B1*Moment1[num] + (1 - B1)*gradient;
        float V = B2*Moment2[num] + (1 - B2)*(gradient*gradient);

        float Mbc = M/(1 - cachedB1);
        float Vbc = V/(1 - cachedB2);

        float stepSize = LR*(Mbc/(sqrt(Vbc) + epsilon));
        float newWeight = weight - stepSize;

        Moment1[num] = M;
        Moment2[num] = V;

        {
            std::lock_guard<std::mutex> lock(adamMutex);
            weightOptimized++;
            if(weightOptimized == weightNumber){
                currentStep++;
                weightOptimized = 0;
                cachedB1 *= B1;
                cachedB2 *= B2;
            }
        }

        return newWeight;

    }

    void save(std::ofstream& fileOut){

        fileOut.write(reinterpret_cast<char*>(&weightNumber), sizeof(int));
        fileOut.write(reinterpret_cast<char*>(&B1), sizeof(float));
        fileOut.write(reinterpret_cast<char*>(&B2), sizeof(float));
        fileOut.write(reinterpret_cast<char*>(&epsilon), sizeof(float));

    }

    void load(std::ifstream& fileIn){

        fileIn.read(reinterpret_cast<char*>(&weightNumber), sizeof(int));
        Moment1 = Matrix::zeros(weightNumber);
        Moment2 = Matrix::zeros(weightNumber);
        fileIn.read(reinterpret_cast<char*>(&B1), sizeof(float));
        fileIn.read(reinterpret_cast<char*>(&B2), sizeof(float));
        cachedB1 = B1;
        cachedB2 = B2;
        fileIn.read(reinterpret_cast<char*>(&epsilon), sizeof(float));

    }

};