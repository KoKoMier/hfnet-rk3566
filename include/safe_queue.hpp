#ifndef HFNET_SAFE_QUEUE_HPP
#define HFNET_SAFE_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>

// 线程安全有界队列
template <typename T>
class SafeQueue {
private:
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cv_push;
    std::condition_variable cv_pop;
    size_t max_size;
    bool finished = false;

public:
    explicit SafeQueue(size_t capacity = 4) : max_size(capacity) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_push.wait(lock, [this]() { return q.size() < max_size || finished; });
        if (finished) return;
        q.push(std::move(item));
        cv_pop.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_pop.wait(lock, [this]() { return !q.empty() || finished; });
        if (q.empty() && finished) {
            return false;
        }
        item = std::move(q.front());
        q.pop();
        cv_push.notify_one();
        return true;
    }

    void shutdown() {
        std::unique_lock<std::mutex> lock(mtx);
        finished = true;
        cv_push.notify_all();
        cv_pop.notify_all();
    }
};

#endif // HFNET_SAFE_QUEUE_HPP
