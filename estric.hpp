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
#ifndef EST_TRIC_HPP
#define EST_TRIC_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>

#ifndef PTOL
//#define PTOL 1E-06
#define PTOL 0.0
#endif

class TriangulateEstimate
{
    public:

        TriangulateEstimate(Graph* g): 
            g_(g), tail_freq_(nullptr), tail_freq_remote_(nullptr),
            remote_triangles_(nullptr), ntriangles_(0), 
            win_(MPI_WIN_NULL), twin_(MPI_WIN_NULL)
        {
            comm_ = g_->get_comm();
            lnv_ = g_->get_lnv();
            nv_ = g_->get_nv();
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);
            tail_freq_ = new GraphElem[nv_];
            tail_freq_remote_ = new GraphElem[nv_];
            remote_triangles_ = new GraphElem[size_];
            std::fill(tail_freq_, tail_freq_ + nv_, 0);
            std::fill(tail_freq_remote_, tail_freq_remote_ + nv_, 0);
            std::fill(remote_triangles_, remote_triangles_ + size_, 0);
            for (GraphElem i = 0; i < lnv_; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                GraphElem const i_g = g_->local_to_global(i);
                for (GraphElem m = e0; m < e1; m++)
                {
                    Edge const& edge = g_->get_edge(m);
                    tail_freq_[edge.tail_] += 1;
                }
            }
            MPI_Win_create(tail_freq_, nv_, sizeof(GraphElem), 
                    MPI_INFO_NULL, comm_, &win_);
            MPI_Win_create(&ntriangles_, 1, sizeof(GraphElem), 
                    MPI_INFO_NULL, comm_, &twin_);
            MPI_Win_lock_all(MPI_MODE_NOCHECK, win_);
            MPI_Win_lock_all(MPI_MODE_NOCHECK, twin_);
        }

        ~TriangulateEstimate() {}

        void clear()
        {
            MPI_Win_unlock_all(win_);
            MPI_Win_unlock_all(twin_);
            delete []tail_freq_;
            delete []tail_freq_remote_;
            delete []remote_triangles_;
        }

        // TODO
        inline void check()
        {
        }
        
        inline void lookup_edges(const GraphWeight d = 0.05)
        {
            GraphElem tup[2];
            
            for (GraphElem i = 0; i < lnv_; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                if ((e0 + 1) == e1)
                    continue;
                for (GraphElem m = e0; m < e1-1; m++)
                {
                    Edge const& edge_m = g_->get_edge(m);
                    const int owner = g_->get_owner(edge_m.tail_);
                    tup[0] = edge_m.tail_;
                    if (owner == rank_)
                    {
                        for (GraphElem n = m + 1; n < e1; n++)
                        {
                            Edge const& edge_n = g_->get_edge(n);
                            tup[1] = edge_n.tail_;
                            if (does_edge_exist(tup))
                                ntriangles_ += 1;
                        }
                    }
                    else
                    {
                        for (GraphElem n = m + 1; n < e1; n++)
                        {
                            Edge const& edge_n = g_->get_edge(n);
                            MPI_Get(&tail_freq_remote_[edge_n.tail_], 1, MPI_GRAPH_TYPE, 
                                    owner, edge_n.tail_, 1, MPI_GRAPH_TYPE, win_);
                        }
                        MPI_Win_flush_all(win_);
                        calc_remote_triangles(owner, d);
                        std::fill(tail_freq_remote_, tail_freq_remote_ + nv_, 0);
                    }
                }
            }
            ntriangles_ += std::accumulate(remote_triangles_, remote_triangles_ + size_, 0);
            for (int p = 0; p < size_; p++)
            {
                MPI_Accumulate(&remote_triangles_[p], 1, MPI_GRAPH_TYPE, 
                                    p, 0, 1, MPI_GRAPH_TYPE, MPI_SUM, twin_);
            }
            MPI_Win_flush_all(twin_);
        }

        // Chung-Lu probability calculation
        inline void lookup_edges_cl(const GraphWeight d = 0.05)
        {
            MPI_Allreduce(tail_freq_, tail_freq_remote_, nv_, 
                    MPI_GRAPH_TYPE, MPI_SUM, comm_);

            GraphElem tup[2];
            GraphElem ne = g_->get_ne();

            for (GraphElem i = 0; i < lnv_; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                if ((e0 + 1) == e1)
                    continue;
                for (GraphElem m = e0; m < e1-1; m++)
                {
                    Edge const& edge_m = g_->get_edge(m);
                    const int owner = g_->get_owner(edge_m.tail_);
                    tup[0] = edge_m.tail_;
                    if (owner == rank_)
                    {
                        for (GraphElem n = m + 1; n < e1; n++)
                        {
                            Edge const& edge_n = g_->get_edge(n);
                            tup[1] = edge_n.tail_;
                            if (does_edge_exist(tup))
                                ntriangles_ += 1;
                        }
                    }
                    else
                    {
                        const GraphWeight ki = (GraphWeight)tail_freq_remote_[edge_m.tail_];
                        for (GraphElem n = m + 1; n < e1; n++)
                        {
                            Edge const& edge_n = g_->get_edge(n);
                            const GraphWeight kj = (GraphWeight)tail_freq_remote_[edge_n.tail_];
                            const GraphWeight prob = (ki * kj) / (GraphWeight)(2.0*ne);
                            if (genRandom<GraphWeight>(PTOL, d) <= prob)
                                remote_triangles_[owner] += 1;
                        }
                    }
                }
            }
            ntriangles_ += std::accumulate(remote_triangles_, remote_triangles_ + size_, 0);
            for (int p = 0; p < size_; p++)
            {
                MPI_Accumulate(&remote_triangles_[p], 1, MPI_GRAPH_TYPE, 
                                    p, 0, 1, MPI_GRAPH_TYPE, MPI_SUM, twin_);
            }
            MPI_Win_flush_all(twin_);
        }

        inline GraphWeight lookup_edges_phases()
        {
            GraphElem tup[2];
            GraphWeight tpos = 0.0, tneg = 0.0, fneg = 0.0;
            
            // local
            for (GraphElem i = 0; i < lnv_; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                if ((e0 + 1) == e1)
                    continue;
                for (GraphElem m = e0; m < e1-1; m++)
                {
                    Edge const& edge_m = g_->get_edge(m);
                    const int owner = g_->get_owner(edge_m.tail_);
                    tup[0] = edge_m.tail_;
                    if (owner == rank_)
                    {
                        for (GraphElem n = m + 1; n < e1; n++)
                        {
                            Edge const& edge_n = g_->get_edge(n);
                            const GraphWeight prob = (GraphWeight)(tail_freq_[edge_n.tail_] / (GraphWeight)lnv_);
                            tup[1] = edge_n.tail_;
                            if (does_edge_exist(tup))
                            {
                                ntriangles_ += 1;
                                if (genRandom<GraphWeight>(PTOL, 1.0) <= prob)
                                    tpos += 1.0;
                                else
                                    fneg += 1.0;
                            }
                            else
                            {
                                if (genRandom<GraphWeight>(PTOL, 1.0) > prob)
                                    tneg += 1.0;
                            }
                        }
                    }
                }
            }
            
            GraphWeight acc[3] = {tpos, tneg, fneg}, 
                        acc_tot_sum[3] = {0.0, 0.0, 0.0}, 
                        acc_tot_min[3] = {0.0, 0.0, 0.0},
                        acc_tot_max[3] = {0.0, 0.0, 0.0};
            MPI_Allreduce(acc, acc_tot_sum, 3, MPI_WEIGHT_TYPE, MPI_SUM, comm_);
            MPI_Allreduce(acc, acc_tot_max, 3, MPI_WEIGHT_TYPE, MPI_MAX, comm_);
            MPI_Allreduce(acc, acc_tot_min, 3, MPI_WEIGHT_TYPE, MPI_MIN, comm_);
            acc[0] = (acc[0] - acc_tot_min[0]) / (acc_tot_max[0] - acc_tot_min[0]);
            acc[1] = (acc[1] - acc_tot_min[1]) / (acc_tot_max[1] - acc_tot_min[1]);
            acc[2] = (acc[2] - acc_tot_min[2]) / (acc_tot_max[2] - acc_tot_min[2]);
            GraphWeight se = acc[0] / (acc[0] + acc[1]);
            GraphWeight sp = acc[0] / (acc[0] + acc[2]);
            GraphWeight t1 = (se - 1.0);
            GraphWeight t2 = (sp - 1.0);
            GraphWeight d = std::sqrt(t1*t1 + t2*t2);

            // remote
            for (GraphElem i = 0; i < lnv_; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                if ((e0 + 1) == e1)
                    continue;

                for (GraphElem m = e0; m < e1-1; m++)
                {
                    Edge const& edge_m = g_->get_edge(m);
                    const int owner = g_->get_owner(edge_m.tail_);
                    tup[0] = edge_m.tail_;
                    if (owner != rank_)
                    {
                        for (GraphElem n = m + 1; n < e1; n++)
                        {
                            Edge const& edge_n = g_->get_edge(n);
                            MPI_Get(&tail_freq_remote_[edge_n.tail_], 1, MPI_GRAPH_TYPE, 
                                    owner, edge_n.tail_, 1, MPI_GRAPH_TYPE, win_);
                        }
                        MPI_Win_flush_all(win_);
                        calc_remote_triangles(owner, d);
                        std::fill(tail_freq_remote_, tail_freq_remote_ + nv_, 0);
                    }
                }
            }
            ntriangles_ += std::accumulate(remote_triangles_, remote_triangles_ + size_, 0);
            for (int p = 0; p < size_; p++)
            {
                MPI_Accumulate(&remote_triangles_[p], 1, MPI_GRAPH_TYPE, 
                        p, 0, 1, MPI_GRAPH_TYPE, MPI_SUM, twin_);
            }
            MPI_Win_flush_all(twin_);
            
            return d;
        }

        inline GraphWeight lookup_edges_cl_phases()
        {
            MPI_Allreduce(tail_freq_, tail_freq_remote_, nv_, 
                    MPI_GRAPH_TYPE, MPI_SUM, comm_);

            GraphElem tup[2];
            GraphElem ne = g_->get_ne();
            GraphWeight tpos = 0.0, tneg = 0.0, fneg = 0.0;

            // local
            for (GraphElem i = 0; i < lnv_; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                if ((e0 + 1) == e1)
                    continue;

                for (GraphElem m = e0; m < e1-1; m++)
                {
                    Edge const& edge_m = g_->get_edge(m);
                    const int owner = g_->get_owner(edge_m.tail_);
                    tup[0] = edge_m.tail_;
                    
                    if (owner == rank_)
                    {
                        for (GraphElem n = m + 1; n < e1; n++)
                        {
                            Edge const& edge_n = g_->get_edge(n);
                            const GraphWeight prob = (GraphWeight)(tail_freq_[edge_n.tail_] / (GraphWeight)lnv_);
                            tup[1] = edge_n.tail_;
                            if (does_edge_exist(tup))
                            {
                                ntriangles_ += 1;
                                if (genRandom<GraphWeight>(PTOL, 1.0) <= prob)
                                    tpos += 1.0;
                                else
                                    fneg += 1.0;
                            }
                            else
                            {
                                if (genRandom<GraphWeight>(PTOL, 1.0) > prob)
                                    tneg += 1.0;
                            }
                        }
                    }
                }
            }
             
            GraphWeight acc[3] = {tpos, tneg, fneg}, 
                        acc_tot_sum[3] = {0.0, 0.0, 0.0}, 
                        acc_tot_min[3] = {0.0, 0.0, 0.0},
                        acc_tot_max[3] = {0.0, 0.0, 0.0};
            MPI_Allreduce(acc, acc_tot_sum, 3, MPI_WEIGHT_TYPE, MPI_SUM, comm_);
            MPI_Allreduce(acc, acc_tot_max, 3, MPI_WEIGHT_TYPE, MPI_MAX, comm_);
            MPI_Allreduce(acc, acc_tot_min, 3, MPI_WEIGHT_TYPE, MPI_MIN, comm_);
            acc[0] = (acc[0] - acc_tot_min[0]) / (acc_tot_max[0] - acc_tot_min[0]);
            acc[1] = (acc[1] - acc_tot_min[1]) / (acc_tot_max[1] - acc_tot_min[1]);
            acc[2] = (acc[2] - acc_tot_min[2]) / (acc_tot_max[2] - acc_tot_min[2]);
            GraphWeight se = acc[0] / (acc[0] + acc[1]);
            GraphWeight sp = acc[0] / (acc[0] + acc[2]);
            GraphWeight t1 = (se - 1.0);
            GraphWeight t2 = (sp - 1.0);
            GraphWeight d = std::sqrt(t1*t1 + t2*t2);

            // remote
            for (GraphElem i = 0; i < lnv_; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                if ((e0 + 1) == e1)
                    continue;
                for (GraphElem m = e0; m < e1-1; m++)
                {
                    Edge const& edge_m = g_->get_edge(m);
                    const int owner = g_->get_owner(edge_m.tail_);
                    if (owner != rank_)
                    {
                        tup[0] = edge_m.tail_;
                        const GraphWeight ki = (GraphWeight)tail_freq_remote_[edge_m.tail_];
                        for (GraphElem n = m + 1; n < e1; n++)
                        {
                            Edge const& edge_n = g_->get_edge(n);
                            const GraphWeight kj = (GraphWeight)tail_freq_remote_[edge_n.tail_];
                            const GraphWeight prob = (ki * kj) / (GraphWeight)(2.0*ne);
                            if (genRandom<GraphWeight>(PTOL, d) <= prob)
                                remote_triangles_[owner] += 1;
                        }
                    }
                }
            }
            ntriangles_ += std::accumulate(remote_triangles_, remote_triangles_ + size_, 0);
            for (int p = 0; p < size_; p++)
            {
                MPI_Accumulate(&remote_triangles_[p], 1, MPI_GRAPH_TYPE, 
                                    p, 0, 1, MPI_GRAPH_TYPE, MPI_SUM, twin_);
            }
            MPI_Win_flush_all(twin_);

            return d;
        }


        inline bool does_edge_exist(GraphElem tup[2])
        {
            GraphElem e0, e1;
            const GraphElem lv = g_->global_to_local(tup[0]);
            g_->edge_range(lv, e0, e1);
            for (GraphElem e = e0; e < e1; e++)
            {
                Edge const& edge = g_->get_edge(e);
                if (tup[1] == edge.tail_)
                    return true;
                if (edge.tail_ > tup[1]) 
                    break;
            }
            return false;
        }

        void calc_remote_triangles(const int pe, const GraphWeight d = 0.05)
        {
            const GraphElem lnv_pe = g_->get_range(pe);
            for (GraphElem i = 0; i < nv_; i++)
            {
                if (tail_freq_remote_[i] > 0)
                {
                    const GraphWeight prob = (GraphWeight)(tail_freq_remote_[i] / (GraphWeight)lnv_pe);
                    if (genRandom<GraphWeight>(PTOL, d) <= prob)
                        remote_triangles_[pe] += 1;
                }
            }
        }

        inline GraphElem count()
        {
#if defined(USE_EUCLIDEAN_CLASSIFIER)
            GraphWeight d;
#if defined(USE_CL_MODEL)
            d = lookup_edges_cl_phases();
#else
            d = lookup_edges_phases();
#endif
            if (rank_ == 0)
                std::cout << "Calculated probability threshold: " << d << std::endl;
#else
#if defined(USE_CL_MODEL)
            lookup_edges_cl();
#else
            lookup_edges();
#endif
#endif
            GraphElem ttc = 0, ltc = ntriangles_;
            MPI_Barrier(comm_);
            MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
            return (ttc/3);
        }
    private:
        Graph* g_;
        GraphElem ntriangles_, lnv_, nv_;
	GraphElem *tail_freq_, *tail_freq_remote_, *remote_triangles_;
	
        MPI_Win win_, twin_;
        int rank_, size_;
        MPI_Comm comm_;
};
#endif
