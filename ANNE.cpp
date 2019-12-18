#include "ANNE.h"

ANNE::Synapse::Synapse(Neuron * s, Neuron * t) : source(s), target(t), weight(1) {}

ANNE::Neuron::Neuron() : inputs(), outputs(), bias(1), value(0) {}

void ANNE::Neuron::connect(Neuron * neuron)
{
    Synapse * synapse = new Synapse(neuron, this);
    inputs.push_back(synapse);
    neuron->outputs.push_back(synapse);
}

void ANNE::Neuron::connect(Layer * layer)
{
    for(Neuron & neuron : layer->neurons)
        connect(&neuron);
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

void ANNE::Neuron::adjust(float learningFactor, float target)
{
    float outputErrorSum = 0;
    for(Synapse * output : outputs)
        outputErrorSum += output->target->error * output->weight;
    error = (target - value) * value * (1 - value) * (outputErrorSum > 0 ? outputErrorSum : 1);
    for(Synapse * synapse : inputs)
    {
        Neuron & source = *synapse->source;
        synapse->weight += learningFactor * error * source.value;
    }
    bias += learningFactor * error;
}

ANNE::Layer::Layer(unsigned int size) : neurons(size) {}

void ANNE::Layer::connect(Layer * layer)
{
    for(Neuron & neuron : neurons)
        neuron.connect(layer);
}

void ANNE::Layer::activate(ActivationFunction function)
{
    for(Neuron & neuron : neurons)
        neuron.activate(function);
}

ANNE::Network::Network(std::vector<unsigned int> sizes)
{
    for(unsigned int size : sizes)
        layers.push_back(Layer(size));
}

void ANNE::Network::connect()
{
    for(unsigned int i = 1; i < layers.size(); i++)
    {
        Layer & layer = layers[i];
        layer.connect(&layers[i - 1]);
    }
}

std::vector<float> ANNE::Network::compute(ActivationFunction function, std::vector<float> input, bool round)
{
    Layer & inputLayer = layers[0];
    for(unsigned int i = 0; i < inputLayer.neurons.size(); i++)
    {
        Neuron & neuron = inputLayer.neurons[i];
        neuron.value = input[i];
    }
    for(unsigned int i = 1; i < layers.size(); i++)
    {
        Layer & layer = layers[i];
        layer.activate(function);
    }
    Layer & outputLayer = layers[layers.size() - 1];
    std::vector<float> output(outputLayer.neurons.size());
    for(unsigned int i = 0; i < output.size(); i++)
    {
        float value = outputLayer.neurons[i].value;
        output[i] = round ? ::round(value) : value;
    }
    return output;
}

void ANNE::Network::learn(ActivationFunction function, std::vector<float> input, std::vector<float> output, float learningFactor)
{
    std::vector<float> result = compute(function, input);
    Layer & outputLayer = layers[layers.size() - 1];
    for(unsigned int i = 0; i < outputLayer.neurons.size(); i++)
    {
        Neuron & neuron = outputLayer.neurons[i];
        neuron.adjust(output[i], learningFactor);
    }
    for(unsigned int l = layers.size() - 2; l > 0; l--)
    {
        Layer & layer = layers[l];
        for(unsigned int i = 0; i < layer.neurons.size(); i++)
        {
            Neuron & neuron = layer.neurons[i];
            neuron.adjust(learningFactor);
        }
    }
}

void ANNE::Network::learn(ActivationFunction function, std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> outputs, float learningFactor, unsigned int epochs)
{
    for(unsigned int i = 0; i < epochs; i++)
        for(unsigned int j = 0; j < inputs.size(); j++)
            learn(function, inputs[j], outputs[j], learningFactor);
}
