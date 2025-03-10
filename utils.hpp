// ***********************************************************************
//
//                              TRIC
//
// ***********************************************************************
//
//       Copyright (2019) Battelle Memorial Institute
//                      All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************ 


#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

#define PI                          (3.14159)

#ifndef MAX_PRINT_NEDGE
#define MAX_PRINT_NEDGE             (100000)
#endif

// Read https://en.wikipedia.org/wiki/Linear_congruential_generator#Period_length
// about choice of LCG parameters
// From numerical recipes
// TODO FIXME investigate larger periods
#define MLCG                        (2147483647)    // 2^31 - 1
#define ALCG                        (16807)         // 7^5
#define BLCG                        (0)

#define SR_UP_TAG                   100
#define SR_DOWN_TAG                 101
#define SR_SIZES_UP_TAG             102
#define SR_SIZES_DOWN_TAG           103
#define SR_X_UP_TAG                 104
#define SR_X_DOWN_TAG               105
#define SR_Y_UP_TAG                 106
#define SR_Y_DOWN_TAG               107
#define SR_LCG_TAG                  108

#define MAX(x,y)                    ((x > y) ? (x) : (y))
#define MIN(x,y)                    ((x < y) ? (x) : (y))

#define ONE                         (1)
#define TWO                         (2)
#define FIVE                        (5)
#define PERCOF(x,y)                 std::ceil((y) * ((x) / 100))
#define ONEPERCOF(y)                PERCOF(ONE, y)
#define TWOPERCOF(y)                PERCOF(TWO, y)
#define FIVEPERCOF(y)               PERCOF(FIVE, y)

#include <random>
#include <utility>
#include <cstring>
#include <cinttypes>

#ifdef USE_32_BIT_GRAPH
using GraphElem = int32_t;
using GraphWeight = float;
const MPI_Datatype MPI_GRAPH_TYPE = MPI_INT32_T;
const MPI_Datatype MPI_WEIGHT_TYPE = MPI_FLOAT;
#else
using GraphElem = int64_t;
using GraphWeight = double;
const MPI_Datatype MPI_GRAPH_TYPE = MPI_INT64_T;
const MPI_Datatype MPI_WEIGHT_TYPE = MPI_DOUBLE;
#endif

extern unsigned seed;

#if defined(AGGR_BUFR)|| defined(AGGR_BUFR_INRECV)  || defined(AGGR_BUFR_IRECV) || defined(AGGR_BUFR_RMA) || defined(AGGR_HEUR) || defined(AGGR_MAP) || defined(AGGR_HASH) || defined(AGGR_HASH2) || defined(AGGR_PUSH)
#ifndef DEFAULT_BUF_SIZE
#define DEFAULT_BUF_SIZE   (8000)
#endif
#endif

// Is nprocs a power-of-2?
int is_pwr2(int nprocs) 
{ return ((nprocs != 0) && !(nprocs & (nprocs - 1))); }

// round to multiple of mult
GraphElem roundUp(GraphElem num, GraphElem mult)
{
    if (mult == 0)
        return num;
    GraphElem rem = num % mult;
    if (rem == 0)
        return num;
    return num + mult - rem;
}

// return unint32_t seed
GraphElem reseeder(unsigned initseed)
{
    std::seed_seq seq({initseed});
    std::vector<std::uint32_t> seeds(1);
    seq.generate(seeds.begin(), seeds.end());

    return (GraphElem)seeds[0];
}

// Local random number generator 
template<typename T, typename G = std::default_random_engine>
T genRandom(T lo, T hi)
{
    thread_local static G gen(std::random_device{}());
    using Dist = typename std::conditional
        <
        std::is_integral<T>::value
        , std::uniform_int_distribution<T>
        , std::uniform_real_distribution<T>
        >::type;

    thread_local static Dist utd {};
    return utd(gen, typename Dist::param_type{lo, hi});
}

#if defined(USE_RAPID_FAM_ALLOC)
#include "rapid.hpp"

//Allocator Impl from https://en.cppreference.com/w/cpp/named_req/Allocator#Examples
template<class T>
struct RapidAllocator
{
    typedef T value_type;

    RapidAllocator() = default;

    template<class U>
    constexpr RapidAllocator(const RapidAllocator <U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)){
            throw std::bad_array_new_length();
        }

        if (auto p = static_cast<T*>(rapid.malloc(n * sizeof(T)))) {
            return p;
        }

        throw std::bad_alloc();
    }

    void deallocate(T* p, std::size_t n) noexcept {
        rapid.free(p);
    }

    //Helper function for demonstration purposes to show
    //the Vec's data is in the Rapid VA space.
    //Not needed in actual code
    static bool is_fam_pointer(T* addr) noexcept {
        auto base_va = rapid.get_base_va();
        auto last_va = rapid.get_last_va();
        return *base_va <= addr && addr < *last_va;
    }
private:
    static rapid::Fam rapid;
};

template<class T, class U>
bool operator==(const RapidAllocator <T>&, const RapidAllocator <U>&) { return true; }

template<class T, class U>
bool operator!=(const RapidAllocator <T>&, const RapidAllocator <U>&) { return false; }

template <typename T>
rapid::Fam RapidAllocator<T>::rapid = rapid::Fam{};
#endif

// Parallel Linear Congruential Generator
// x[i] = (a*x[i-1] + b)%M
class LCG
{
    public:
        LCG(unsigned seed, GraphWeight* drand, 
            GraphElem n, MPI_Comm comm = MPI_COMM_WORLD): 
        seed_(seed), drand_(drand), n_(n)
        {
            comm_ = comm;
            MPI_Comm_size(comm_, &nprocs_);
            MPI_Comm_rank(comm_, &rank_);

            // allocate long random numbers
            rnums_.resize(n_);

            // init x0
            if (rank_ == 0)
                x0_ = reseeder(seed_);

            // step #1: bcast x0 from root
            MPI_Bcast(&x0_, 1, MPI_GRAPH_TYPE, 0, comm_);
            
            // step #2: parallel prefix to generate first random value per process
            parallel_prefix_op();
        }
        
        ~LCG() { rnums_.clear(); }

        // matrix-matrix multiplication for 2x2 matrices
        void matmat_2x2(GraphElem c[], GraphElem a[], GraphElem b[])
        {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    GraphElem sum = 0;
                    for (int k = 0; k < 2; k++) {
                        sum += a[i*2+k]*b[k*2+j];
                    }
                    c[i*2+j] = sum;
                }
            }
        }

        // x *= y
        void matop_2x2(GraphElem x[], GraphElem y[])
        {
            GraphElem tmp[4];
            matmat_2x2(tmp, x, y);
            memcpy(x, tmp, sizeof(GraphElem[4]));
        }

        // find kth power of a 2x2 matrix
        void mat_power(GraphElem mat[], GraphElem k)
        {
            GraphElem tmp[4];
            memcpy(tmp, mat, sizeof(GraphElem[4]));

            // mat-mat multiply k times
            for (GraphElem p = 0; p < k-1; p++)
                matop_2x2(mat, tmp);
        }

        // parallel prefix for matrix-matrix operation
        // `x0 is the very first random number in the series
        // `ab is a 2-length array which stores a and b
        // `n_ is (n/p)
        // `rnums is n_ length array which stores the random nums for a process
        void parallel_prefix_op()
        {
            GraphElem global_op[4]; 
            global_op[0] = ALCG;
            global_op[1] = 0;
            global_op[2] = BLCG;
            global_op[3] = 1;

            mat_power(global_op, n_);        // M^(n/p)
            GraphElem prefix_op[4] = {1,0,0,1};  // I in row-major

            GraphElem global_op_recv[4];

            int steps = (int)(log2((double)nprocs_));

            for (int s = 0; s < steps; s++) {
                
                int mate = rank_^(1 << s); // toggle the sth LSB to find my neighbor
                
                // send/recv global to/from mate
                MPI_Sendrecv(global_op, 4, MPI_GRAPH_TYPE, mate, SR_LCG_TAG, 
                        global_op_recv, 4, MPI_GRAPH_TYPE, mate, SR_LCG_TAG, 
                        comm_, MPI_STATUS_IGNORE);

                matop_2x2(global_op, global_op_recv);   
                
                if (mate < rank_) 
                    matop_2x2(prefix_op, global_op_recv);

                MPI_Barrier(comm_);
            }

            // populate the first random number entry for each process
            // (x0*a + b)%P
            if (rank_ == 0)
                rnums_[0] = x0_;
            else
                rnums_[0] = (x0_*prefix_op[0] + prefix_op[2])%MLCG;
        }

        // generate random number based on the first 
        // random number on a process
        // TODO check the 'quick'n dirty generators to
        // see if we can avoid the mod
        void generate()
        {
#if defined(PRINT_LCG_LONG_RANDOM_NUMBERS)
            for (int k = 0; k < nprocs_; k++) {
                if (k == rank_) {
                    std::cout << "------------" << std::endl;
                    std::cout << "Process#" << rank_ << " :" << std::endl;
                    std::cout << "------------" << std::endl;
                    std::cout << rnums_[0] << std::endl;
                    for (GraphElem i = 1; i < n_; i++) {
                        rnums_[i] = (rnums_[i-1]*ALCG + BLCG)%MLCG;
                        std::cout << rnums_[i] << std::endl;
                    }
                }
                MPI_Barrier(comm_);
            }
#else
            for (GraphElem i = 1; i < n_; i++) {
                rnums_[i] = (rnums_[i-1]*ALCG + BLCG)%MLCG;
            }
#endif
            GraphWeight mult = 1.0 / (GraphWeight)(1.0 + (GraphWeight)(MLCG-1));

#if defined(PRINT_LCG_DOUBLE_RANDOM_NUMBERS)
            for (int k = 0; k < nprocs_; k++) {
                if (k == rank_) {
                    std::cout << "------------" << std::endl;
                    std::cout << "Process#" << rank_ << " :" << std::endl;
                    std::cout << "------------" << std::endl;

                    for (GraphElem i = 0; i < n_; i++) {
                        drand_[i] = (GraphWeight)((GraphWeight)fabs(rnums_[i]) * mult ); // 0-1
                        std::cout << drand_[i] << std::endl;
                    }
                }
                MPI_Barrier(comm_);
            }
#else
            for (GraphElem i = 0; i < n_; i++)
                drand_[i] = (GraphWeight)((GraphWeight)fabs(rnums_[i]) * mult); // 0-1
#endif
        }
         
        // copy from drand_[idx_start] to new_drand, 
        // rescale the random numbers between lo and hi
        void rescale(GraphWeight* new_drand, GraphElem idx_start, GraphWeight const& lo)
        {
            GraphWeight range = (1.0 / (GraphWeight)nprocs_);

#if defined(PRINT_LCG_DOUBLE_LOHI_RANDOM_NUMBERS)
            for (int k = 0; k < nprocs_; k++) {
                if (k == rank_) {
                    std::cout << "------------" << std::endl;
                    std::cout << "Process#" << rank_ << " :" << std::endl;
                    std::cout << "------------" << std::endl;

                    for (GraphElem i = idx_start, j = 0; i < n_; i++, j++) {
                        new_drand[j] = lo + (GraphWeight)(range * drand_[i]);
                        std::cout << new_drand[j] << std::endl;
                    }
                }
                MPI_Barrier(comm_);
            }
#else
            for (GraphElem i = idx_start, j = 0; i < n_; i++, j++)
                new_drand[j] = lo + (GraphWeight)(range * drand_[i]); // lo-hi
#endif
        }

    private:
        MPI_Comm comm_;
        int nprocs_, rank_;
        unsigned seed_;
        GraphElem n_, x0_;
        GraphWeight* drand_;
        std::vector<GraphElem> rnums_;
};

// locks
#ifdef USE_OPENMP_LOCK
#else
#ifdef USE_SPINLOCK 
#include <atomic>
std::atomic_flag lkd_ = ATOMIC_FLAG_INIT;
#else
#include <mutex>
std::mutex mtx_;
#endif
void lock() {
#ifdef USE_SPINLOCK 
    while (lkd_.test_and_set(std::memory_order_acquire)) { ; } 
#else
    mtx_.lock();
#endif
}
void unlock() { 
#ifdef USE_SPINLOCK 
    lkd_.clear(std::memory_order_release); 
#else
    mtx_.unlock();
#endif
}
#endif

#endif // UTILS
