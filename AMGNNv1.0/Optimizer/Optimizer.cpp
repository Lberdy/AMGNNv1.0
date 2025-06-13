#include"Adam.cpp"
#include"AMGO.cpp"
#include<algorithm>

enum Optimizers{
    AMGNNO,
    ADAM
};

std::mutex nnMutex;

class Optimizer{

public:
    int batches = 60;
    Optimizers OptimizerName;
    float learningRate = 0.01;
    float lossEpsilon = 0.01;
    int epoches;

    Adam adam;
    AMGO amgo;

    Differentiation differentiation;
    LossFunction lossFunction;

    size_t parallelOperations = 3;
    size_t parallelBatches = 2;
    size_t parallelStoch = 2;

    Optimizer(NN& nn, Optimizers opt, int maxEpoches, Loss_Function LFN) : adam(nn), lossFunction(LFN){

        OptimizerName = opt;
        epoches = maxEpoches;

    }

    Optimizer(){}

private:
    bool isEnd(NN& nn, int epoch, std::vector<std::vector<float>>& InputData, std::vector<std::vector<float>>& OutputData){

        if(epoch == epoches){
            return true;
        }
        
        float loss = lossFunction.calculateLoss(nn, InputData, OutputData);
        std::cout << "loss : "<<loss<<std::endl;
        if(loss <= lossEpsilon){
            return true;
        }

        return false;

    }

    void optimize(NN& nn, std::vector<std::vector<float>> InputData, std::vector<std::vector<float>> OutputData, int epoch){

        ThreadPooling threadpool(parallelOperations);

        int num = 0;

        for(int i = 0; i < nn.HiddenLayers.size(); i++){

            HiddenLayer& layer = nn.HiddenLayers[i];

            for(int j = 0; j < layer.InWeights.size(); j++){
                threadpool.enqueue(
                    [this,&nn,num,i,j,&InputData,&OutputData,epoch](){
                        float gradient = differentiation.derivate(nn, i, j, true, false, lossFunction, InputData, OutputData);
                        {
                            std::unique_lock<std::mutex> lock(nnMutex);
                            float weight = nn.HiddenLayers[i].InWeights[j];
                            switch(OptimizerName){
                                case AMGNNO : nn.HiddenLayers[i].InWeights[j] = amgo.optimizeWeight(weight, gradient, learningRate, epoch, epoches); break;
                                case ADAM : nn.HiddenLayers[i].InWeights[j] = adam.optimizeWeight(weight, gradient, learningRate, num); break;
                            }
                        }
                    }
                );
                num++;
            }

            for(int j = 0; j < layer.biases.size(); j++){
                threadpool.enqueue(
                    [this,&nn,num,i,j,&InputData,&OutputData,epoch](){
                        float gradient = differentiation.derivate(nn, i, j, false, false, lossFunction, InputData, OutputData);
                        {
                            std::unique_lock<std::mutex> lock(nnMutex);
                            float weight = nn.HiddenLayers[i].biases[j];
                            switch(OptimizerName){
                                case AMGNNO : nn.HiddenLayers[i].biases[j] = amgo.optimizeWeight(weight, gradient, learningRate, epoch, epoches); break;
                                case ADAM : nn.HiddenLayers[i].biases[j] = adam.optimizeWeight(weight, gradient, learningRate, num); break;
                            }
                        }
                    }
                );
                num++;
            }

        }

        for(int i = 0; i < nn.outputLayer.InWeights.size(); i++){
            threadpool.enqueue(
                [this,&nn,num,i,&InputData,&OutputData,epoch](){
                    float gradient = differentiation.derivate(nn, 0, i, true, true, lossFunction, InputData, OutputData);
                    {
                        std::unique_lock<std::mutex> lock(nnMutex);
                        float weight = nn.outputLayer.InWeights[i];
                        switch(OptimizerName){
                            case AMGNNO : nn.outputLayer.InWeights[i] = amgo.optimizeWeight(weight, gradient, learningRate, epoch, epoches); break;
                            case ADAM : nn.outputLayer.InWeights[i] = adam.optimizeWeight(weight, gradient, learningRate, num); break;
                        }
                    }
                }
            );
            num++;
        }

        for(int i = 0; i < nn.outputLayer.biases.size(); i++){
            threadpool.enqueue(
                [this,&nn,num,i,&InputData,&OutputData,epoch](){
                    float gradient = differentiation.derivate(nn, 0, i, false, true, lossFunction, InputData, OutputData);
                    {
                        std::unique_lock<std::mutex> lock(nnMutex);
                        float weight = nn.outputLayer.biases[i];
                        switch(OptimizerName){
                            case AMGNNO : nn.outputLayer.biases[i] = amgo.optimizeWeight(weight, gradient, learningRate, epoch, epoches); break;
                            case ADAM : nn.outputLayer.biases[i] = adam.optimizeWeight(weight, gradient, learningRate, num); break;
                        }
                    }
                }
            );
            num++;
        }

        threadpool.wait_for_all_tasks();

    }

public:
    void gradientDescent(NN& nn, std::vector<std::vector<std::vector<float>>>& DataSet){

        std::vector<std::vector<float>> InputData;
        std::vector<std::vector<float>> OutputData;

        for(std::vector<std::vector<float>>& sample:DataSet){
            InputData.push_back(sample[0]);
            OutputData.push_back(sample[1]);
        }

        int epoch = 0;

        while(!isEnd(nn, epoch, InputData, OutputData)){

            optimize(nn, InputData, OutputData, epoch);

            epoch++;

        }

    }

    void stochasticGradientDescent(NN& nn, std::vector<std::vector<std::vector<float>>>& DataSet){

        std::vector<std::vector<float>> InputDataCheckEnd;
        std::vector<std::vector<float>> OutputDataCheckEnd;

        ThreadPooling threadpoolStoch(parallelStoch);

        for(std::vector<std::vector<float>>& sample:DataSet){
            InputDataCheckEnd.push_back(sample[0]);
            OutputDataCheckEnd.push_back(sample[1]);
        }

        std::random_device rd;
        std::default_random_engine engine(rd());

        int epoch = 0;

        while(!isEnd(nn, epoch, InputDataCheckEnd, OutputDataCheckEnd)){

            std::vector<std::vector<std::vector<float>>> DataSetCopy = DataSet;

            shuffle(DataSetCopy.begin(), DataSetCopy.end(), engine);
            
            for(std::vector<std::vector<float>>& sample:DataSetCopy){
                std::vector<std::vector<float>> input = {sample[0]};
                std::vector<std::vector<float>> output = {sample[1]};
                threadpoolStoch.enqueue(
                    [this,&nn,input,output,epoch](){
                        optimize(nn,input,output,epoch);
                    }   
                );
            }

            threadpoolStoch.wait_for_all_tasks();

            epoch++;

        }

    }

    void miniBatchGradientDescent(NN& nn, std::vector<std::vector<std::vector<float>>>& DataSet){

        std::vector<std::vector<float>> InputDataCheckEnd;
        std::vector<std::vector<float>> OutputDataCheckEnd;

        ThreadPooling batchThreadPool(parallelBatches);

        for(std::vector<std::vector<float>>& sample:DataSet){
            InputDataCheckEnd.push_back(sample[0]);
            OutputDataCheckEnd.push_back(sample[1]);
        }

        std::random_device rd;
        std::default_random_engine engine(rd());

        int epoch = 0;

        while(!isEnd(nn, epoch, InputDataCheckEnd, OutputDataCheckEnd)){

            std::vector<std::vector<std::vector<float>>> DataSetCopy = DataSet;

            std::vector<std::vector<std::vector<std::vector<float>>>> Batches;

            shuffle(DataSetCopy.begin(), DataSetCopy.end(), engine);

            int index = 0;

            while(index < DataSetCopy.size()){

                std::vector<std::vector<std::vector<float>>> Batch;
                std::vector<std::vector<float>> InputBatch;
                std::vector<std::vector<float>> OutputBatch;
                bool first = true;

                while((index % batches != 0 || first) && index < DataSetCopy.size()){
                    InputBatch.push_back(DataSet[index][0]);
                    OutputBatch.push_back(DataSet[index][1]);
                    index++;
                    first = false;
                }

                Batch.push_back(InputBatch);
                Batch.push_back(OutputBatch);

                Batches.push_back(Batch);
                

            }

            for(std::vector<std::vector<std::vector<float>>>& Batch:Batches){
                batchThreadPool.enqueue(
                    [this,&nn,&Batch,epoch](){
                        optimize(nn,Batch[0],Batch[1],epoch);
                    }
                );
            }

            batchThreadPool.wait_for_all_tasks();

            epoch++;

        }

    }

    void save(std::ofstream& fileOut){

        fileOut.write(reinterpret_cast<char*>(&batches), sizeof(int));
        int OPT = static_cast<int>(OptimizerName);
        fileOut.write(reinterpret_cast<char*>(&OPT), sizeof(int));
        fileOut.write(reinterpret_cast<char*>(&learningRate), sizeof(float));
        fileOut.write(reinterpret_cast<char*>(&lossEpsilon), sizeof(float));
        fileOut.write(reinterpret_cast<char*>(&epoches), sizeof(int));

        adam.save(fileOut);
        amgo.save(fileOut);
        differentiation.save(fileOut);
        lossFunction.save(fileOut);

    }

    void load(std::ifstream& fileIn){

        fileIn.read(reinterpret_cast<char*>(&batches), sizeof(int));
        int OPT;
        fileIn.read(reinterpret_cast<char*>(&OPT), sizeof(int));
        OptimizerName = static_cast<Optimizers>(OPT);
        fileIn.read(reinterpret_cast<char*>(&learningRate), sizeof(float));
        fileIn.read(reinterpret_cast<char*>(&lossEpsilon), sizeof(float));
        fileIn.read(reinterpret_cast<char*>(&epoches), sizeof(int));

        adam.load(fileIn);
        amgo.load(fileIn);
        differentiation.load(fileIn);
        lossFunction.load(fileIn);

    }

};