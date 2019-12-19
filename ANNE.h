#ifndef ANNE_H
#define ANNE_H

#include <vector>

namespace ANNE
{
    enum ActivationFunction { ReLU, Tanh, Sigmoid, Linear };

    struct DataSet
    {
        std::vector<std::vector<float>> inputs;
        std::vector<std::vector<float>> outputs;

        DataSet();
        ~DataSet();

        void addData(std::vector<float> input, std::vector<float> output);
    };

    struct Neuron;
    struct Layer;

    struct Synapse
    {
        Neuron * source;
        Neuron * target;
        float weight;

        Synapse(Neuron * source, Neuron * target);
        ~Synapse();
    };

    struct Neuron
    {
        std::vector<Synapse *> inputs;
        std::vector<Synapse *> outputs;
        float bias;
        float value;
        float error;

        Neuron();
        ~Neuron();

        void connect(Neuron * neuron);
        void connect(Layer * layer);
        void activate(ActivationFunction function);
        void calculateError();
        void calculateError(float target);
        void updateWeights(float learningRate);
    };

    struct Layer
    {
        std::vector<Neuron *> neurons;

        Layer(unsigned int size = 0);
        ~Layer();

        void connect(Layer * layer);
        void activate(ActivationFunction function);
        void calculateErrors();
        void updateWeights(float learningRate);
    };

    struct Network
    {
        std::vector<Layer *> layers;

        Network(std::vector<unsigned int> sizes = {});
        ~Network();

        std::vector<float> compute(ActivationFunction function, std::vector<float> input, bool round = false);
        void learn(ActivationFunction function, std::vector<float> input, std::vector<float> output, float learningRate);
        void train(ActivationFunction function, DataSet * dataSet, float learningRate, unsigned int epochs);
    };
};

#endif // ANNE_H
