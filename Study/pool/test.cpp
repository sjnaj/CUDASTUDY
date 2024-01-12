#include "threadpool.hpp"
#include <iostream>
ThreadPool tp(10);
int main()
{
    auto func = [](int i)
    { std::cout << i; };
    for (int i = 0; i < 10; i++)
        tp.spawn(func, i);
    tp.wait_and_stop();
}