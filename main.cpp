#include <iostream>
#include <fstream>
#include <random>
#include <array>

template<size_t N>
static std::array<double, N> randmat() {
    static const auto rand01fn = [] {
        static std::default_random_engine randgen;
        static std::uniform_real_distribution<double> rand(-1.0, 1.0);
        return rand(randgen);
    };

    auto result = std::array<double, N>();
    std::generate(result.begin(), result.end(), rand01fn);
    return result;
}

template<size_t N, size_t M>
static std::array<double, N + M> matconcat(
        const std::array<double, N>& a,
        const std::array<double, M>& b
) {
    auto result = std::array<double, N + M>();
    std::copy(a.begin(), a.end(), result.begin());
    std::copy(b.begin(), b.end(), result.begin() + N);
    return result;
};

template<size_t N, size_t M>
static std::array<double, M> matmul(
        const std::array<double, N * M>& w,
        const std::array<double, N>& h
) {
    auto result = std::array<double, M>();
    for(size_t m = 0; m < M; ++m)
        for(size_t n = 0; n < N; ++n)
            result[m] += h[n] * w[n * M + m];
    return result;
};

template<size_t N>
static std::array<double, N> matadd(
        const std::array<double, N>& a,
        const std::array<double, N>& b
) {
    auto result = std::array<double, N>();
    for(size_t i = 0; i < N; ++i)
        result[i] = a[i] + b[i];
    return result;
};

static double sigmoid(const double x) {
    return 1.0 / (1.0 + exp(-x));
}

template<size_t N>
static std::array<double, N> matsig(const std::array<double, N>& a) {
    auto result = std::array<double, N>();
    for(size_t i = 0; i < N; ++i)
        result[i] = sigmoid(a[i]);
    return result;
};

template<size_t N>
static std::array<double, N> mattanh(const std::array<double, N>& a) {
    auto result = std::array<double, N>();
    for(size_t i = 0; i < N; ++i)
        result[i] = tanh(a[i]);
    return result;
};

template<size_t N>
static std::array<double, N> matpointmul(
        const std::array<double, N>& a,
        const std::array<double, N>& b
) {
    auto result = std::array<double, N>();
    for(size_t i = 0; i < N; ++i)
        result[i] = a[i] * b[i];
    return result;
};

template<size_t N>
static std::array<double, N> matpointmul(
        const std::array<double, N>& a,
        const std::array<double, N>& b,
        const std::array<double, N>& c
) {
    auto result = std::array<double, N>();
    for(size_t i = 0; i < N; ++i)
        result[i] = a[i] * b[i] * c[i];
    return result;
};

template<size_t N>
static std::array<double, N> oneminusmat(const std::array<double, N>& a) {
    auto result = std::array<double, N>();
    for(size_t i = 0; i < N; ++i)
        result[i] = 1.0 - a[i];
    return result;
};

template<size_t N>
static std::array<double, N> mattimesscalar(const std::array<double, N>& a, const double s) {
    auto result = std::array<double, N>();
    for(size_t i = 0; i < N; ++i)
        result[i] = a[i] * s;
    return result;
};

template<size_t N>
static size_t matargmax(const std::array<double, N>& a) {
    double biggest = std::numeric_limits<double>::min();
    size_t biggestIndex = 0;

    for(size_t i = 0; i < N; ++i) {
        if (a[i] > biggest) {
            biggest = a[i];
            biggestIndex = i;
        }
    }

    return biggestIndex;
};

struct LstmCell {
    typedef std::array<double, 256> SingleWidthMat;
    typedef std::array<double, 512> DoubleWidthMat;
    typedef std::array<double, 256 * 512> WeightMat;

    // weights
    static WeightMat Wf;
    static SingleWidthMat bf;
    static WeightMat error_Wf;
    static SingleWidthMat error_bf;

    static WeightMat Wi;
    static SingleWidthMat bi;
    static WeightMat error_Wi;
    static SingleWidthMat error_bi;

    static WeightMat WC;
    static SingleWidthMat bC;
    static WeightMat error_WC;
    static SingleWidthMat error_bC;

    static WeightMat Wo;
    static SingleWidthMat bo;
    static WeightMat error_Wo;
    static SingleWidthMat error_bo;

    static void zeroErrors() {
        error_Wf = WeightMat();
        error_bf = SingleWidthMat();

        error_Wi = WeightMat();
        error_bi = SingleWidthMat();

        error_WC = WeightMat();
        error_bC = SingleWidthMat();

        error_Wo = WeightMat();
        error_bo = SingleWidthMat();
    }

    template<size_t N>
    static void learn(std::array<double, N>& w, const std::array<double, N>& e, const double learningRate) {
        for(size_t i = 0; i < N; ++i)
            w[i] -= learningRate * e[i];
    }

    static void learn(const double learningRate) {
        learn(Wf, error_Wf, learningRate);
        learn(bf, error_bf, learningRate);

        learn(Wi, error_Wi, learningRate);
        learn(bi, error_bi, learningRate);

        learn(WC, error_WC, learningRate);
        learn(bC, error_bC, learningRate);

        learn(Wo, error_Wo, learningRate);
        learn(bo, error_bo, learningRate);
    }

    // states
    SingleWidthMat h = randmat<256>();
    SingleWidthMat C = randmat<256>();

    // temporary states
    SingleWidthMat x;
    SingleWidthMat h_before;
    SingleWidthMat C_before;

    DoubleWidthMat h_concat_x;
    SingleWidthMat f;
    SingleWidthMat i;
    SingleWidthMat Cbar;
    SingleWidthMat o;

    void forwardPass(
        const SingleWidthMat& new_x,
        const SingleWidthMat& new_h_before,
        const SingleWidthMat& new_C_before
    ) {
        x = new_x;
        h_before = new_h_before;
        C_before = new_C_before;

        h_concat_x = matconcat(h_before, x);

        // calculate the new C
        const auto f_before_bias = matmul<512, 256>(Wf, h_concat_x);
        const auto f_before_activation = matadd(f_before_bias, bf);
        f = matsig(f_before_activation);

        const auto i_before_bias = matmul<512, 256>(Wi, h_concat_x);
        const auto i_before_activation = matadd(i_before_bias, bi);
        i = matsig(i_before_activation);

        const auto Cbar_before_bias = matmul<512, 256>(WC, h_concat_x);
        const auto Cbar_before_activation = matadd(Cbar_before_bias, bC);
        Cbar = matsig(Cbar_before_activation);

        auto C_after_forget = matpointmul(C_before, f);
        C = matadd(C_after_forget, matpointmul(i, Cbar));

        // calculate the new o
        const auto o_before_bias = matmul<512, 256>(Wo, h_concat_x);
        const auto o_before_activation = matadd(o_before_bias, bo);
        o = matsig(o_before_activation);

        h = matpointmul(o, mattanh(C));
    }

    SingleWidthMat backwardPass(
        const SingleWidthMat& y,
        const SingleWidthMat& error_Cnext
    ) {
        /*
         * Between the cells, we're only propagating the error from C, not from h. I'm not 100% sure that's the right
         * thing to do.
         *
         * One the one hand, there is no ambiguity about what we want h to be, so any propagated error for h would just
         * add noise. On the other hand, if we were stacking this LSTM, or combining it with other stuff, we would need
         * to do it.
         */

        auto error_h = SingleWidthMat();
        for(size_t i = 0; i < y.size(); ++i)
            error_h[i] = pow(y[i] - h[i], 2.0);

        // back through h
        const auto tanhC = mattanh(C);
        auto error_o = matpointmul(error_h, tanhC);
        auto error_Chere = matpointmul(error_h, o, oneminusmat(matpointmul(tanhC, tanhC)));
        auto error_C = matadd(error_Chere, error_Cnext);

        // back through o
        auto error_o_before_activation = matpointmul(error_o, o, oneminusmat(o));
        error_bo = matadd(error_bo, error_o_before_activation);
        auto& error_o_before_bias = error_o_before_activation;

        for(size_t j = 0; j < error_o_before_bias.size(); ++j)
            for(size_t i = 0; i < error_Wo.size() / error_o_before_bias.size(); ++i)
                error_Wo[i * 256 + j] += error_o_before_bias[j] * h_concat_x[i];

        //auto error_hcx = LstmCell::DoubleWidthMat();
        //for(size_t j = 0; j < error_o_before_bias.size(); ++j)
        //    for(size_t i = 0; i < error_hcx.size(); ++i)
        //        error_hcx[i] += error_o_before_bias[i] * Wo[i * 256 + j];

        // back through C
        auto& error_C_after_forget = error_C;
        auto error_Cbar = matpointmul(error_C, i);
        auto error_i = matpointmul(error_C, Cbar);

        auto error_C_before = matpointmul(error_C_after_forget, f);
        auto error_f = matpointmul(error_C_after_forget, C_before);

        // back through Cbar
        auto error_Cbar_before_activation = matpointmul(error_Cbar, Cbar, oneminusmat(Cbar));
        error_bC = matadd(error_bC, error_Cbar_before_activation);
        auto& error_Cbar_before_bias = error_Cbar_before_activation;

        for(size_t j = 0; j < error_Cbar_before_bias.size(); ++j)
            for(size_t i = 0; i < error_WC.size() / error_Cbar_before_bias.size(); ++i)
                error_WC[i * 256 + j] += error_Cbar_before_bias[j] * h_concat_x[i];

        //for(size_t j = 0; j < error_Cbar_before_bias.size(); ++j)
        //    for(size_t i = 0; i < error_hcx.size(); ++i)
        //        error_hcx[i] += error_Cbar_before_bias[i] * WC[i * 256 + j];

        // back through i
        auto error_i_before_activation = matpointmul(error_i, i, oneminusmat(i));
        error_bi = matadd(error_bi, error_i_before_activation);
        auto& error_i_before_bias = error_i_before_activation;

        for(size_t j = 0; j < error_i_before_bias.size(); ++j)
            for(size_t i = 0; i < error_Wi.size() / error_i_before_bias.size(); ++i)
                error_Wi[i * 256 + j] += error_i_before_bias[j] * h_concat_x[i];

        //for(size_t j = 0; j < error_i_before_bias.size(); ++j)
        //    for(size_t i = 0; i < error_hcx.size(); ++i)
        //        error_hcx[i] += error_i_before_bias[i] * Wi[i * 256 + j];

        // back through f
        auto error_f_before_activation = matpointmul(error_f, f, oneminusmat(f));
        error_bf = matadd(error_bf, error_f_before_activation);
        auto& error_f_before_bias = error_f_before_activation;

        for(size_t j = 0; j < error_f_before_bias.size(); ++j)
            for(size_t i = 0; i < error_Wf.size() / error_f_before_bias.size(); ++i)
                error_Wf[i * 256 + j] += error_f_before_bias[j] * h_concat_x[i];

        //for(size_t j = 0; j < error_f_before_bias.size(); ++j)
        //    for(size_t i = 0; i < error_hcx.size(); ++i)
        //        error_hcx[i] += error_f_before_bias[i] * Wf[i * 256 + j];

        // back through h
        //auto error_h_before = LstmCell::SingleWidthMat();
        //std::copy(error_hcx.begin(), error_hcx.begin() + error_h_before.size(), error_h_before.begin());

        return error_C_before;
    }
};

LstmCell::WeightMat LstmCell::Wf = randmat<(256 + 256) * 256>();
LstmCell::SingleWidthMat LstmCell::bf = randmat<256>();
LstmCell::WeightMat LstmCell::error_Wf = LstmCell::WeightMat();
LstmCell::SingleWidthMat LstmCell::error_bf = LstmCell::SingleWidthMat();

LstmCell::WeightMat LstmCell::Wi = randmat<(256 + 256) * 256>();
LstmCell::SingleWidthMat LstmCell::bi = randmat<256>();
LstmCell::WeightMat LstmCell::error_Wi = LstmCell::WeightMat();
LstmCell::SingleWidthMat LstmCell::error_bi = LstmCell::SingleWidthMat();

LstmCell::WeightMat LstmCell::WC = randmat<(256 + 256) * 256>();
LstmCell::SingleWidthMat LstmCell::bC = randmat<256>();
LstmCell::WeightMat LstmCell::error_WC = LstmCell::WeightMat();
LstmCell::SingleWidthMat LstmCell::error_bC = LstmCell::SingleWidthMat();

LstmCell::WeightMat LstmCell::Wo = randmat<(256 + 256) * 256>();
LstmCell::SingleWidthMat LstmCell::bo = randmat<256>();
LstmCell::WeightMat LstmCell::error_Wo = LstmCell::WeightMat();
LstmCell::SingleWidthMat LstmCell::error_bo = LstmCell::SingleWidthMat();

int main() {
    std::ifstream in("cnus.txt");
    int charsRead = 0;

    std::vector<LstmCell> cells(1024);
    int cellIndex = 0;

    while(!in.eof()) {
        const int ch = in.get();
        charsRead += 1;
        // make one-hot encoding from the input
        auto x = std::array<double, 256>();
        x[ch] = 1.0;

        // backwards pass
        // We do this first because at this point we have the new input available.
        if(cellIndex == cells.size() - 1) {
            // run the backwards pass
            // calculate mse just for tracking progress
            double mse = 0.0;
            LstmCell::zeroErrors();

            auto error_C = LstmCell::SingleWidthMat();
            for(int errorCellIndex = cells.size() - 1; errorCellIndex > 0; --errorCellIndex) {
                const int targetCellIndex = errorCellIndex + 1;  // Every cell is trying to predict the input of the next cell.
                const LstmCell::SingleWidthMat& target =
                        targetCellIndex >= cells.size() ?
                        x : cells[targetCellIndex].x;

                error_C = cells[errorCellIndex].backwardPass(target, error_C);

                const LstmCell::SingleWidthMat& output = cells[errorCellIndex].h;
                for(size_t i = 0; i < target.size(); ++i)
                    mse += pow(target[i] - output[i], 2.0);
            }
            mse /= cells.size() * 256;
            std::cout << mse << std::endl;

            LstmCell::learn(0.1);
        }

        // forward pass
        {
            const LstmCell& prevCell = cells.at((cellIndex - 1) % cells.size());
            cells[cellIndex].forwardPass(x, prevCell.h, prevCell.C);
        }

        // After a bit of training, run it forward a bunch to see what it does.
        if(charsRead % 10000 == 0) {
            const LstmCell* prevCell = &cells[cellIndex];
            LstmCell cell;
            for(int i = 0; i < 1000; ++i) {
                // find the character to display
                const size_t ch = matargmax(prevCell->h);
                if(ch < 32)
                    std::cout << " " << ch << " ";
                else
                    std::cout << (char)ch;

                auto x = std::array<double, 256>();
                x[ch] = 1.0;

                cell.forwardPass(x, prevCell->h, prevCell->C);
                prevCell = &cell;
            }
            std::cout << std::endl;
        }

        cellIndex += 1;
        cellIndex %= cells.size();
    }

    return 0;
}