#pragma once

#include <cstdint>
#include <future>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <iostream>
class ThreadPool
{
private:
    std::vector<std::thread> threads;
    std::queue<std::function<void(void)>> tasks;

    std::mutex mutex;
    std::condition_variable cv, cv_wait;

    // bool stop_pool;
    std::atomic<bool> stop_pool;
    std::atomic<size_t> active_threads;
    const size_t capacity;

    template <typename Func, typename... Args, typename Rtrn = typename std::result_of<Func(Args...)>::type>
    auto make_task(Func &&func, Args &&...args) -> std::packaged_task<Rtrn(void)>
    {
        auto aux = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
        return std::packaged_task<Rtrn(void)>(aux);
    }

    void before_task_hook()
    {
        active_threads++;
    }
    void after_task_hook()
    {
        active_threads--;
        if (active_threads == 0 && tasks.empty())
        {
            stop_pool = true;
            cv_wait.notify_one();
        }
    }

public:
    ThreadPool(size_t capacity_) : stop_pool(false), active_threads(0), capacity(capacity_)
    {
        auto wait_loop = [this]() -> void
        {
            while (true)
            {
                std::function<void(void)> task;
                {
                    std::unique_lock<std::mutex> lk(mutex);

                    auto predicate = [this]() -> bool
                    {
                        return (stop_pool || !(tasks.empty()));
                    };
                    cv.wait(lk, predicate);

                    if (stop_pool && tasks.empty())
                    {
                        return;
                    }
                    task = std::move(tasks.front());
                    tasks.pop();
                    before_task_hook();
                }
                task();
                {
                    std::lock_guard<std::mutex> lk(mutex);
                    after_task_hook();
                }
            }
        };
        for (size_t id = 0; id < capacity; id++)
        {
            threads.emplace_back(wait_loop);
        }
    }

    ~ThreadPool()
    {
        stop_pool = true;
        cv.notify_all();
        for (auto &thread : threads)
        {
            thread.join();
        }
    }

    template <typename Func, typename... Args, typename Rtrn = typename std::result_of<Func(Args...)>::type>
    auto enqueue(Func &&func, Args &&...args) -> std::future<Rtrn>
    {
        auto task = make_task(func, args...);
        auto future = task.get_future();
        auto task_ptr = std::make_shared<decltype(task)>(std::move(task));
        {
            std::lock_guard<std::mutex> lk(mutex);
            if (stop_pool)
            {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            auto payload = [task_ptr]() -> void
            {
                (*task_ptr)();
                // task_ptr->operator()();
            };
            tasks.emplace(payload);
        }
        cv.notify_one();
        return future;
    }

    template <
        typename Func,
        typename... Args, typename Rtrn = typename std::result_of<Func(Args...)>::type>
    auto spawn(
        Func &&func,
        Args &&...args) -> std::future<Rtrn>
    {

        // enqueue if idling threads
        if (active_threads < capacity)
            return enqueue(func, args...);
        // else process sequential
        else
        // func(args...);
        {
            auto task = make_task(func, args...);
            func(args...);
            return task.get_future();
        }
    }

    void wait_and_stop()
    {

        // wait for pool being set to stop
        std::unique_lock<std::mutex>
            unique_lock(mutex);

        auto predicate = [&]() -> bool
        {
            return stop_pool;
        };

        cv_wait.wait(unique_lock, predicate);
    }
};
