#include "/home/fengsc/CUDASTUDY/Study/pool/threadpool.hpp"
#include <iostream>
std::future<int> fs[10];

int main()
{
    ThreadPool tp(10);

    auto func = [](int i)
    {
        std::cout << i;
        return i;
    };
    for (int i = 0; i < 10; i++)
        fs[i] = tp.enqueue(func, i);
    for(int i=0;i<10;i++){
        std::cout<<fs[i].get();
    }
    tp.stop();
}