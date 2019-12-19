#include <QCoreApplication>

#include <iostream>

#include "ANNE.h"

using namespace ANNE;

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

void computeAndPrint(ActivationFunction function, Network * network, DataSet * dataSet)
{
    for(unsigned int i = 0; i < dataSet->inputs.size(); i++)
    {
        std::vector<float> input = dataSet->inputs[i];
        std::vector<float> output = dataSet->outputs[i];
        std::vector<float> result = network->compute(function, input, true);
        print(input, output, result);
    }
}

void testTwoOperandsAndGate()
{
    Network network({2, 1});

    ActivationFunction function = Sigmoid;

    DataSet dataSet;
    dataSet.addData({0, 0}, {0});
    dataSet.addData({0, 1}, {0});
    dataSet.addData({1, 0}, {0});
    dataSet.addData({1, 1}, {1});

    network.train(function, &dataSet, 0.5, 1000);
    computeAndPrint(function, &network, &dataSet);
}

void testTwoOperandsXorGate()
{
    Network network({2, 2, 1});

    ActivationFunction function = Sigmoid;

    DataSet dataSet;
    dataSet.addData({0, 0}, {0});
    dataSet.addData({0, 1}, {1});
    dataSet.addData({1, 0}, {1});
    dataSet.addData({1, 1}, {0});

    network.train(function, &dataSet, 0.5, 1000);
    computeAndPrint(function, &network, &dataSet);
}

void testThreeOperandsXorGate()
{
    Network network({3, 3, 2, 1});

    ActivationFunction function = Sigmoid;

    DataSet dataSet;
    dataSet.addData({0, 0, 0}, {0});
    dataSet.addData({0, 0, 1}, {1});
    dataSet.addData({0, 1, 0}, {1});
    dataSet.addData({1, 0, 0}, {1});
    dataSet.addData({1, 0, 1}, {0});
    dataSet.addData({1, 1, 0}, {0});
    dataSet.addData({1, 1, 1}, {0});

    network.train(function, &dataSet, 0.5, 10000);
    computeAndPrint(function, &network, &dataSet);
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    testTwoOperandsAndGate();
    std::cout << "\n";
    testTwoOperandsXorGate();
    std::cout << "\n";
    testThreeOperandsXorGate();
    std::cout << "\n";

    return a.exec();
}
