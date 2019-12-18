#ifndef ANNE_H
#define ANNE_H

#include <vector>

namespace ANNE
{
    enum ActivationFunction { ReLU, Tanh, Sigmoid, Linear };

    struct Neuron;
    struct Layer;

    struct Synapse
    {
        Neuron * source;
        Neuron * target;
        float weight;

        Synapse(Neuron * source, Neuron * target);
    };

    struct Neuron
    {
        std::vector<Synapse *> inputs;
        std::vector<Synapse *> outputs;
        float bias;
        float value;
        float error;

        Neuron();
        void connect(Neuron * neuron);
        void connect(Layer * layer);
        void activate(ActivationFunction function);
        void adjust(float learningFactor, float target = 1);
    };

    struct Layer
    {
        std::vector<Neuron> neurons;

        Layer(unsigned int size = 0);
        void connect(Layer * layer);
        void activate(ActivationFunction function);
    };

    struct Network
    {
        std::vector<Layer> layers;

        Network(std::vector<unsigned int> sizes = {});
        void connect();
        std::vector<float> compute(ActivationFunction function, std::vector<float> input, bool round = false);
        void learn(ActivationFunction function, std::vector<float> input, std::vector<float> output, float learningFactor);
        void learn(ActivationFunction function, std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> outputs, float learningFactor, unsigned int epochs);
    };
};

#endif // ANNE_H
