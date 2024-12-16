#include "../exercise.h"

// READ: 静态字段 <https://zh.cppreference.com/w/cpp/language/static>
// READ: 虚析构函数 <https://zh.cppreference.com/w/cpp/language/destructor>

struct A {
    // TODO: 正确初始化静态字段
    static int num_a;

    A() {
        ++num_a;
    }
    // 当一个基类的析构函数被声明为 virtual 时，它允许派生类覆盖（override）这个函数。
    // 这是实现运行时多态（runtime polymorphism）的关键，
    // 即在程序运行时根据对象的实际类型调用相应的析构函数。
    // 如果析构函数不是 virtual，派生类的析构函数将不会被调用，
    // 这可能导致资源泄漏或其他清理问题，因为派生类特有的资源不会被释放。
    virtual ~A() {
        --num_a;
    }

    virtual char name() const {
        return 'A';
    }
};
struct B final : public A {
    // TODO: 正确初始化静态字段
    static int num_b;

    B() {
        ++num_b;
    }
    ~B() {
        --num_b;
    }

    char name() const final {
        return 'B';
    }
};

// 在类外部定义和初始化静态成员变量
int A::num_a = 0; // 初始化 A::num_a
int B::num_b = 0; // 初始化 B::num_b

int main(int argc, char **argv) {
    auto a = new A;
    auto b = new B;
    ASSERT(A::num_a == 2, "Fill in the correct value for A::num_a");
    ASSERT(B::num_b == 1, "Fill in the correct value for B::num_b");
    ASSERT(a->name() == 'A', "Fill in the correct value for a->name()");
    ASSERT(b->name() == 'B', "Fill in the correct value for b->name()");

    delete a;
    delete b;
    ASSERT(A::num_a == 0, "Every A was destroyed");
    ASSERT(B::num_b == 0, "Every B was destroyed");

    A *ab = new B;// 派生类指针可以随意转换为基类指针
    ASSERT(A::num_a == 1, "Fill in the correct value for A::num_a");
    ASSERT(B::num_b == 1, "Fill in the correct value for B::num_b");
    ASSERT(ab->name() == 'B', "Fill in the correct value for ab->name()");

    // TODO: 基类指针无法随意转换为派生类指针，补全正确的转换语句
    B &bb = *dynamic_cast<B *>(ab);
    // printf("num_a %d, num_b %d \n", A::num_a, B::num_b);
    ASSERT(bb.name() == 'B', "Fill in the correct value for bb->name()");

    // TODO: ---- 以下代码不要修改，通过改正类定义解决编译问题 ----
    delete ab;// 通过指针可以删除指向的对象，即使是多态对象
    // printf("num_a %d, num_b %d \n", A::num_a, B::num_b);
    ASSERT(A::num_a == 0, "Every A was destroyed");
    ASSERT(B::num_b == 0, "Every B was destroyed");

    return 0;
}
