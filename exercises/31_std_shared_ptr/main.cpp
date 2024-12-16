#include "../exercise.h"
#include <memory>

// READ: `std::shared_ptr` <https://zh.cppreference.com/w/cpp/memory/shared_ptr>
// READ: `std::weak_ptr` <https://zh.cppreference.com/w/cpp/memory/weak_ptr>

// TODO: 将下列 `?` 替换为正确的值
int main(int argc, char **argv) {
    auto shared = std::make_shared<int>(10);
    std::shared_ptr<int> ptrs[]{shared, shared, shared};

    std::weak_ptr<int> observer = shared;
    ASSERT(observer.use_count() == 4, "");

    ptrs[0].reset();
    ASSERT(observer.use_count() == 3, "");

    ptrs[1] = nullptr;
    ASSERT(observer.use_count() == 2, "");

    ptrs[2] = std::make_shared<int>(*shared);
    // std::cout << observer.use_count() << std::endl;
    ASSERT(observer.use_count() == 1, "");
    // 这个新的 shared_ptr 指向一个复制的 int 对象（*shared），而不是 shared 对象。

    ptrs[0] = shared;
    ptrs[1] = shared;
    ptrs[2] = std::move(shared);
    // std::cout << observer.use_count() << std::endl;
    ASSERT(observer.use_count() == 3, "");

    std::ignore = std::move(ptrs[0]);
    ptrs[1] = std::move(ptrs[1]);
    ptrs[1] = std::move(ptrs[2]);
    // std::cout << observer.use_count() << std::endl;
    ASSERT(observer.use_count() == 2, "");
    // ptrs[0] 和 ptrs[1] 都使用 std::move() 转移了它们的所有权。
    // ptrs[1] 被重新赋值为 ptrs[2] 的所有权，因此 ptrs[1] 和 ptrs[2] 都指向相同的对象。
    // observer.use_count() 变为 2，因为只有 ptrs[1] 和 observer 保持对 shared 的引用。

    shared = observer.lock();
    // std::cout << observer.use_count() << std::endl;
    ASSERT(observer.use_count() == 3, "");
    // 通过 observer.lock() 获取 shared（即将 shared 赋值为 observer 所观察的 shared_ptr），
    // 此时 shared 成为 shared_ptr<int>，引用计数增加至 3（ptrs[1]、observer 和 shared 都持有引用）

    shared = nullptr;
    for (auto &ptr : ptrs) ptr = nullptr;
    // std::cout << observer.use_count() << std::endl;
    ASSERT(observer.use_count() == 0, "");

    shared = observer.lock();
    // std::cout << observer.use_count() << std::endl;
    ASSERT(observer.use_count() == 0, "");
    // 由于 shared 已经被销毁，lock() 返回一个空指针。

    return 0;
}
