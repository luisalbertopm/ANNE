#include <QCoreApplication>

#include <iostream>

#include "ANNE.h"

void print(std::vector<float> input, std::vector<float> output, std::vector<float> result)
{
    std::cout << "Input:";
    for(float value : input)
        std::cout << " " << value;

    std::cout << " ";

    std::cout << "Output:";
    for(float value : output)
        std::cout << " " << value;

    std::cout << " ";

    std::cout << "Result:";
    for(float value : result)
        std::cout << " " << value;

    std::cout << "\n";
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    using namespace ANNE;

    Network network({3, 2, 1});
    network.connect();

    ActivationFunction function = Sigmoid;

    std::vector<std::vector<float>> inputs, outputs;
    inputs.push_back({0, 0, 0}); outputs.push_back({0});
    inputs.push_back({0, 0, 1}); outputs.push_back({1});
    inputs.push_back({0, 1, 0}); outputs.push_back({0});
    inputs.push_back({1, 0, 0}); outputs.push_back({0});
    inputs.push_back({1, 0, 1}); outputs.push_back({1});
    inputs.push_back({1, 1, 0}); outputs.push_back({1});
    inputs.push_back({1, 1, 1}); outputs.push_back({0});

    network.learn(function, inputs, outputs, 0.5, 10000);

    for(unsigned int i = 0; i < inputs.size(); i++)
    {
        std::vector<float> input = inputs[i];
        std::vector<float> output = outputs[i];
        std::vector<float> result = network.compute(function, input, true);
        print(input, output, result);
    }

    return a.exec();
}
