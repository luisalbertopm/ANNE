#include <iostream>
#include <thread>
#include <mutex>

#include "ANNE.h"

using namespace ANNE;

static std::mutex computeMutex;

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

void computeAndPrint(ActivationFunction function, Network * network, DataSet * dataSet, std::string title)
{
    computeMutex.lock();
    std::cout << title.c_str() << "\n";
    for(unsigned int i = 0; i < dataSet->inputs.size(); i++)
    {
        std::vector<float> input = dataSet->inputs[i];
        std::vector<float> output = dataSet->outputs[i];
        std::vector<float> result = network->compute(function, input, true);
        print(input, output, result);
    }
    std::cout << "\n";
    computeMutex.unlock();
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
    computeAndPrint(function, &network, &dataSet, "TWO OPERANDS AND GATE");
}

void testTwoOperandsOrGate()
{
    Network network({2, 1});

    ActivationFunction function = Sigmoid;

    DataSet dataSet;
    dataSet.addData({0, 0}, {0});
    dataSet.addData({0, 1}, {1});
    dataSet.addData({1, 0}, {1});
    dataSet.addData({1, 1}, {1});

    network.train(function, &dataSet, 0.5, 1000);
    computeAndPrint(function, &network, &dataSet, "TWO OPERANDS OR GATE");
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

    network.train(function, &dataSet, 0.5, 10000);
    computeAndPrint(function, &network, &dataSet, "TWO OPERANDS XOR GATE");
}

void testThreeOperandsAndGate()
{
    Network network({3, 3, 1});

    ActivationFunction function = Sigmoid;

    DataSet dataSet;
    dataSet.addData({0, 0, 0}, {0});
    dataSet.addData({0, 0, 1}, {0});
    dataSet.addData({0, 1, 0}, {0});
    dataSet.addData({1, 0, 0}, {0});
    dataSet.addData({1, 0, 1}, {0});
    dataSet.addData({1, 1, 0}, {0});
    dataSet.addData({1, 1, 1}, {1});

    network.train(function, &dataSet, 0.5, 10000);
    computeAndPrint(function, &network, &dataSet, "THREE OPERANDS AND GATE");
}

void testThreeOperandsOrGate()
{
    Network network({3, 3, 1});

    ActivationFunction function = Sigmoid;

    DataSet dataSet;
    dataSet.addData({0, 0, 0}, {0});
    dataSet.addData({0, 0, 1}, {1});
    dataSet.addData({0, 1, 0}, {1});
    dataSet.addData({1, 0, 0}, {1});
    dataSet.addData({1, 0, 1}, {1});
    dataSet.addData({1, 1, 0}, {1});
    dataSet.addData({1, 1, 1}, {1});

    network.train(function, &dataSet, 0.5, 10000);
    computeAndPrint(function, &network, &dataSet, "THREE OPERANDS OR GATE");
}

void testThreeOperandsXorGate()
{
    Network network({3, 3, 1});

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
    computeAndPrint(function, &network, &dataSet, "THREE OPERANDS XOR GATE");
}

int main(int argc, char * argv[])
{
    std::thread testTwoOperandsAndGateThread(testTwoOperandsAndGate);
    std::thread testTwoOperandsOrGateThread(testTwoOperandsOrGate);
    std::thread testTwoOperandsXorGateThread(testTwoOperandsXorGate);
    std::thread testThreeOperandsAndGateThread(testThreeOperandsAndGate);
    std::thread testThreeOperandsOrGateThread(testThreeOperandsOrGate);
    std::thread testThreeOperandsXorGateThread(testThreeOperandsXorGate);

    testTwoOperandsAndGateThread.join();
    testTwoOperandsOrGateThread.join();
    testTwoOperandsXorGateThread.join();
    testThreeOperandsAndGateThread.join();
    testThreeOperandsOrGateThread.join();
    testThreeOperandsXorGateThread.join();

    return EXIT_SUCCESS;
}
