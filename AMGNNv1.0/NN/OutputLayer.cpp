#include"HiddenLayer.cpp"

enum TaskType{
    REGRESSION,
    BINARRY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION
};

class OutputLayer{

public:
    int neurons;
    std::vector<float> InWeights;
    std::vector<float> biases;
    std::vector<float> values;
    TaskType taskType;
    bool reluRregression = false;

    OutputLayer(TaskType TT, int fanIn, int fanOut){

        taskType = TT;

        InWeights = Matrix::Xavier(fanIn, fanOut);
        biases = Matrix::zeros(fanOut);
        values = Matrix::zeros(fanOut);

        neurons = fanOut;

    }

    OutputLayer(){}

private:
    float sigmoid(float x){
        return 1.0/(1 + exp(-x));
    }

    float softmax(float x){
        float sumExp = 0;
        for(float v:values){
            sumExp += exp(v);
        }

        return exp(x)/sumExp;

    }

public:
    void calculateValues(std::vector<float>& Input){

        Matrix::dot(InWeights, Input, values);
        for(int i = 0; i < values.size(); i++){
            values[i] += biases[i];
        }

        if(taskType == BINARRY_CLASSIFICATION || taskType == MULTILABEL_CLASSIFICATION){
            for(int i = 0; i < values.size(); i++){
                values[i] = sigmoid(values[i]);
            }
        }else if(taskType == MULTICLASS_CLASSIFICATION){
            std::vector<float> tempValues;
            for(float v:values){
                tempValues.push_back(softmax(v));
            }
            values = tempValues;
        }else{
            if(reluRregression){
                for(int i = 0; i < values.size(); i++){
                    if(values[i] < 0){
                        values[i] = -values[i];
                    }
                }
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

        int ttype = static_cast<int>(taskType);
        fileOut.write(reinterpret_cast<char*>(&ttype), sizeof(int));

        fileOut.write(reinterpret_cast<char*>(&reluRregression), sizeof(bool));

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

        int ttype;
        fileIn.read(reinterpret_cast<char*>(&ttype), sizeof(int));
        taskType = static_cast<TaskType>(ttype);

        fileIn.read(reinterpret_cast<char*>(&reluRregression), sizeof(bool));

    }

};