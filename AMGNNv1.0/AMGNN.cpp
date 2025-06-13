#include"Optimizer/Optimizer.cpp"

enum LearningMethod{
    GRADIENT_DESCENT,
    STOCHASTIC_GRADIENT_DESCENT,
    MINI_BATCH_GRADIENT_DESCENT
};

class AMGNN{

public:
    NN nn;
    Optimizer optimizer;
    TaskType taskType;
    LearningMethod learningMethod;

    AMGNN(ActivationFunction AFN, TaskType TT, int InputNeurons, std::vector<int> HiddenNeurons, int OutputNeurons, Loss_Function LFN, Optimizers opt, int Epoches, LearningMethod LRM)
    : nn(AFN, TT, InputNeurons, HiddenNeurons, OutputNeurons), optimizer(nn, opt, Epoches, LFN){

        taskType = TT;
        learningMethod = LRM;

    }

private:

    AMGNN(){}

    double max(std::vector<float> Input){
        int maxim = Input[0];
        for(float value:Input){
            if(value > maxim){
                maxim = value;
            }
        }
        return maxim;
    }

    void argmax(std::vector<float>& Input){
        float maximaum = max(Input);
        for(int i = 0; i < Input.size(); i++){
            Input[i] = (Input[i] < maximaum)?0:1;
        }
    }

    void NormB_MLClassification(std::vector<float>& Input){
        for(int i = 0; i < Input.size(); i++){
            Input[i] = (Input[i] < 0.5)?0:1;
        }
    }

public:
    void train(std::vector<std::vector<std::vector<float>>>& DataSet){  
        switch(learningMethod){
            case GRADIENT_DESCENT : optimizer.gradientDescent(nn, DataSet); break;
            case STOCHASTIC_GRADIENT_DESCENT : optimizer.stochasticGradientDescent(nn, DataSet); break;
            case MINI_BATCH_GRADIENT_DESCENT : optimizer.miniBatchGradientDescent(nn, DataSet); break;
        }
    }

    std::vector<float> predict(std::vector<float> Input){
        std::vector<float> predicted = nn.predict(Input);
        switch(taskType){
            case MULTICLASS_CLASSIFICATION: argmax(predicted); break;
            case BINARRY_CLASSIFICATION:
            case MULTILABEL_CLASSIFICATION: NormB_MLClassification(predicted); break;
        }
        return predicted;
    }

private:
    void load(std::ifstream& fileIn){

        int ttype;
        fileIn.read(reinterpret_cast<char*>(&ttype), sizeof(int));
        taskType = static_cast<TaskType>(ttype);

        int LM;
        fileIn.read(reinterpret_cast<char*>(&LM), sizeof(int));
        learningMethod = static_cast<LearningMethod>(LM);

        nn.load(fileIn);
        optimizer.load(fileIn);

    }

public:
    void saveModel(std::string fileName){

        std::ofstream fileOut(fileName+".AMGNN", std::ios::binary);

        int ttype = static_cast<int>(taskType);
        fileOut.write(reinterpret_cast<char*>(&ttype), sizeof(int));

        int LM = static_cast<int>(learningMethod);
        fileOut.write(reinterpret_cast<char*>(&LM), sizeof(int));

        nn.save(fileOut);
        optimizer.save(fileOut);

    }

    static AMGNN loadModel(std::string fileName){

        std::ifstream fileIn(fileName, std::ios::binary);

        AMGNN neuralnetrwork;
        neuralnetrwork.load(fileIn);

        return neuralnetrwork;

    }

};