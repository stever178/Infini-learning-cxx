#include "../exercise.h"

// READ: 类模板 <https://zh.cppreference.com/w/cpp/language/class_template>

template<class T>
struct Tensor4D {
    unsigned int shape[4];
    T *data;

    Tensor4D(unsigned int const shape_[4], T const *data_) {
        unsigned int size = 1;
        // TODO: 填入正确的 shape 并计算 size
        for (int i = 0; i < 4; i++) {
            shape[i] = shape_[i];
            size *= shape[i];
        }
        data = new T[size];
        // memcpy(shape, shape_, 4 * sizeof(unsigned int));
        // std::memcpy(data, data_, size * sizeof(T));
        memcpy(data, data_, size * sizeof(T));
    }
    ~Tensor4D() {
        delete[] data;
    }

    // 为了保持简单，禁止复制和移动
    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D &&) noexcept = delete;

    // 这个加法需要支持“单向广播”。
    // 具体来说，`others` 可以具有与 `this` 不同的形状，形状不同的维度长度必须为 1。
    // `others` 长度为 1 但 `this` 长度不为 1 的维度将发生广播计算。
    // 例如，`this` 形状为 `[1, 2, 3, 4]`，`others` 形状为 `[1, 2, 1, 4]`，
    // 则 `this` 与 `others` 相加时，3 个形状为 `[1, 2, 1, 4]` 的子张量各自与 `others` 对应项相加。
    Tensor4D &operator+=(Tensor4D const &others) {
        // TODO: 实现单向广播的加法
        // in this example, no need to consider the illegal broadcast
        for (int i = 0; i < 4; i++) {
            // notice this is +=, not +
            ASSERT(shape[i] != others.shape[i] and others.shape[i] != 1, "illegal broadcast");
        }
        // in this case, dim[]=this.shape[]
#define MAX_(a, b) ((a) >= (b) ? (a) : (b))
        int dim0 = MAX_(shape[0], others.shape[0]);
        int dim1 = MAX_(shape[1], others.shape[1]);
        int dim2 = MAX_(shape[2], others.shape[2]);
        int dim3 = MAX_(shape[3], others.shape[3]);
#undef MAX_
        int left_dim23 = shape[2] * shape[3];
        int left_dim123 = shape[1] * left_dim23;
        int right_dim23 = others.shape[2] * others.shape[3]; 
        int right_dim123 = others.shape[1] * right_dim23;

        for (int i0 = 0; i0 < dim0; i0++) {
            int right0 = (others.shape[0] > 1 ? i0 : 0);
            for (int i1 = 0; i1 < dim1; i1++) {
                int right1 = (others.shape[1] > 1 ? i1 : 0);
                for (int i2 = 0; i2 < dim2; i2++) {
                    int right2 = (others.shape[2] > 1 ? i2 : 0);
                    for (int i3 = 0; i3 < dim3; i3++) {
                        int right3 = (others.shape[3] > 1 ? i3 : 0);
                        data[i0 * left_dim123 + i1 * left_dim23 + i2 * shape[3] + i3] +=
                            others.data[right0 * right_dim123 + right1 * right_dim23 + right2 * others.shape[3] + right3];
                    }
                }
            }
        }
        return *this;
    }
};

// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D(shape, data);
        auto t1 = Tensor4D(shape, data);
        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == 7.f, "Every element of t0 should be 7 after adding t1 to it.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        double d0[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        unsigned int s1[]{1, 1, 1, 1};
        double d1[]{1};

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == d0[i] + 1, "Every element of t0 should be incremented by 1 after adding t1 to it.");
        }
    }
}
