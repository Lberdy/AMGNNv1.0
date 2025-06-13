#include"../Matrix.cpp"

enum ActivationFunction{
    LINEAR,
    SIGMOID,
    TANH,
    RELU,
    SOFTPLUS
};

class HiddenLayer{

public:
    int neurons;
    std::vector<float> InWeights;
    std::vector<float> biases;
    std::vector<float> values;
    ActivationFunction ActivationFunctionName;

    HiddenLayer(ActivationFunction AFN, int fanIn, int fanOut){

        ActivationFunctionName = AFN;

        neurons = fanOut;

        if(AFN == LINEAR || AFN == SIGMOID || AFN == TANH){
            InWeights = Matrix::Xavier(fanIn, fanOut);
        }else if(AFN == RELU || AFN == SOFTPLUS){
            InWeights = Matrix::Kaiming(fanIn, fanOut);
        }

        biases = Matrix::zeros(fanOut);
        values = Matrix::zeros(fanOut);

    }

    HiddenLayer(){}

private:
    float Linear(float x){
        return x;
    }

    float sigmoid(float x){
        return 1.0/(1 + exp(-x));
    }

    float Tanh(float x){
        return tanh(x);
    }

    float relu(float x){
        return (x > 0)?x:0;
    }

    float softplus(float x){
        return log(1 + exp(x));
    }

public:
    void calculateValues(std::vector<float>& Input){
        
        Matrix::dot(InWeights, Input, values);
        
        for(int i = 0; i < values.size(); i++){
            values[i] += biases[i];
        }

        for(int i = 0; i < values.size(); i++){
            switch(ActivationFunctionName){
                case LINEAR : values[i] = Linear(values[i]); break;
                case SIGMOID : values[i] = sigmoid(values[i]); break;
                case TANH : values[i] = Tanh(values[i]); break;
                case RELU : values[i] = relu(values[i]); break;
                case SOFTPLUS : values[i] = softplus(values[i]); break;
            }
        }

    }

    void save(std::ofstream& fileOut){

        fileOut.write(reinterpret_cast<char*>(&neurons), sizeof(int));

        size_t weightsSize = InWeights.size();
        fileOut.write(reinterpret_cast<char*>(&weightsSize), sizeof(weightsSize));
        for(size_t i = 0; i < weightsSize; i++){
            float value = InWeights[i];
            fileOut.write(reinterpret_cast<char*>(&value), sizeof(float));
        }

        size_t biasesSize = biases.size();
        fileOut.write(reinterpret_cast<char*>(&biasesSize), sizeof(biasesSize));
        for(size_t i = 0; i < biasesSize; i++){
            float value = biases[i];
            fileOut.write(reinterpret_cast<char*>(&value), sizeof(float));
        }

        int AFN = static_cast<int>(ActivationFunctionName);
        fileOut.write(reinterpret_cast<char*>(&AFN), sizeof(int));

    }

    void load(std::ifstream& fileIn){

        fileIn.read(reinterpret_cast<char*>(&neurons), sizeof(int));
        values = Matrix::zeros(neurons);

        size_t weightsSize;
        fileIn.read(reinterpret_cast<char*>(&weightsSize), sizeof(weightsSize));
        for(size_t i = 0; i < weightsSize; i++){
            float value;
            fileIn.read(reinterpret_cast<char*>(&value), sizeof(float));
            InWeights.push_back(value);
        }

        size_t biasesSize;
        fileIn.read(reinterpret_cast<char*>(&biasesSize), sizeof(biasesSize));
        for(size_t i = 0; i < biasesSize; i++){
            float value;
            fileIn.read(reinterpret_cast<char*>(&value), sizeof(float));
            biases.push_back(value);
        }

        int AFN;
        fileIn.read(reinterpret_cast<char*>(&AFN), sizeof(int));
        ActivationFunctionName = static_cast<ActivationFunction>(AFN);

    }

};