#include "ANNE.h"

ANNE::Synapse::Synapse(Neuron * s, Neuron * t) : source(s), target(t), weight(1) {}

ANNE::Synapse::~Synapse() {}

ANNE::Neuron::Neuron() : inputs(), outputs(), bias(1), value(0) {}

ANNE::Neuron::~Neuron()
{
    for(Synapse * synapse : inputs)
        delete synapse;

    inputs.clear();
    outputs.clear();
}

void ANNE::Neuron::connect(Neuron * neuron)
{
    Synapse * synapse = new Synapse(neuron, this);
    inputs.push_back(synapse);
    neuron->outputs.push_back(synapse);
}

void ANNE::Neuron::connect(Layer * layer)
{
    for(Neuron * neuron : layer->neurons)
        connect(neuron);
}

void ANNE::Neuron::activate(ActivationFunction function)
{
    float sum = bias;
    for(Synapse * synapse : inputs)
        sum += synapse->weight * synapse->source->value;
    if(function == ReLU)
        value = fmax(0.0f, sum);
    else if(function == Tanh)
        value = tanh(sum);
    else if(function == Sigmoid)
        value = 1 / (1 + exp(-sum));
    else if(function == Linear)
        value = sum;
    else
        value = sum;
}

void ANNE::Neuron::calculateError()
{
    float errorSum = 0;
    for(Synapse * output : outputs)
        errorSum += output->target->error * output->weight;
    error = value * (1 - value) * errorSum;
}

void ANNE::Neuron::calculateError(float target)
{
    error = (target - value) * value * (1 - value);
}

void ANNE::Neuron::updateWeights(float learningFactor)
{
    for(Synapse * synapse : inputs)
    {
        Neuron * source = synapse->source;
        synapse->weight += learningFactor * error * source->value;
    }
    bias += learningFactor * error;
}

ANNE::Layer::Layer(unsigned int size) : neurons()
{
    for(unsigned int i = 0; i < size; i++)
        neurons.push_back(new Neuron());
}

ANNE::Layer::~Layer()
{
    for(Neuron * neuron : neurons)
        delete neuron;

    neurons.clear();
}

void ANNE::Layer::connect(Layer * layer)
{
    for(Neuron * neuron : neurons)
        neuron->connect(layer);
}

void ANNE::Layer::activate(ActivationFunction function)
{
    for(Neuron * neuron : neurons)
        neuron->activate(function);
}

void ANNE::Layer::calculateErrors()
{
    for(Neuron * neuron : neurons)
        neuron->calculateError();
}

void ANNE::Layer::updateWeights(float learningFactor)
{
    for(Neuron * neuron : neurons)
        neuron->updateWeights(learningFactor);
}

ANNE::Network::Network(std::vector<unsigned int> sizes)
{
    for(unsigned int size : sizes)
        layers.push_back(new Layer(size));
}

ANNE::Network::~Network()
{
    for(Layer * layer : layers)
        delete layer;

    layers.clear();
}

void ANNE::Network::connect()
{
    for(unsigned int i = 1; i < layers.size(); i++)
        layers[i]->connect(layers[i - 1]);
}

std::vector<float> ANNE::Network::compute(ActivationFunction function, std::vector<float> input, bool round)
{
    Layer * inputLayer = layers[0];
    for(unsigned int i = 0; i < inputLayer->neurons.size(); i++)
        inputLayer->neurons[i]->value = input[i];
    for(unsigned int i = 1; i < layers.size(); i++)
        layers[i]->activate(function);
    Layer * outputLayer = layers[layers.size() - 1];
    std::vector<float> output(outputLayer->neurons.size());
    for(unsigned int i = 0; i < output.size(); i++)
    {
        float value = outputLayer->neurons[i]->value;
        output[i] = round ? ::round(value) : value;
    }
    return output;
}

void ANNE::Network::learn(ActivationFunction function, std::vector<float> input, std::vector<float> output, float learningFactor)
{
    std::vector<float> result = compute(function, input);
    Layer * outputLayer = layers[layers.size() - 1];
    for(unsigned int i = 0; i < outputLayer->neurons.size(); i++)
        outputLayer->neurons[i]->calculateError(output[i]);
    for(unsigned int l = layers.size() - 2; l > 0; l--)
        layers[l]->calculateErrors();
    for(Layer * layer : layers)
        layer->updateWeights(learningFactor);
}

void ANNE::Network::learn(ActivationFunction function, std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> outputs, float learningFactor, unsigned int epochs)
{
    for(unsigned int i = 0; i < epochs; i++)
        for(unsigned int j = 0; j < inputs.size(); j++)
            learn(function, inputs[j], outputs[j], learningFactor);
}
