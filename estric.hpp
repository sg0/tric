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
                for (GraphElem m = e0; m < e1; m++)
                {
                    Edge const& edge = g_->get_edge(m);
                    tail_freq_[edge.tail_] += 1;
                }
            }
            GraphElem lne = g_->get_lne();
            MPI_Win_create(tail_freq_, nv_, sizeof(GraphElem), MPI_INFO_NULL, comm_, &win_);
            MPI_Win_create(&ntriangles_, 1, sizeof(GraphElem), MPI_INFO_NULL, comm_, &twin_);
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
        
        inline void lookup_edges()
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
                        calc_remote_triangles(owner);
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
        inline void lookup_edges_cl()
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
                            if (genRandom<GraphWeight>(PTOL, 1.0) <= prob)
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
            GraphElem tot = 0, tpos = 0, tneg = 0, fneg = 0;
            
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
                            const GraphWeight prob = (GraphWeight)(tail_freq_[edge_n.tail_] / (GraphWeight)(nv_));
                            GraphWeight pcut = genRandom<GraphWeight>(PTOL, 1.0); 
                            tup[1] = edge_n.tail_;
                            if (does_edge_exist(tup))
                            {
                                ntriangles_ += 1;
                                if (pcut <= prob)
                                    tpos += 1;
                                else
                                    fneg += 1;
                            }
                            else
                            {
                               if (pcut > prob)
                                   tneg += 1;
                            }

                            tot += 1;
                        }
                    }
                }
            }

            
            GraphElem acc[4] = {tot, tpos, tneg, fneg}, 
                        acc_tot_sum[4] = {0, 0, 0, 0};
            MPI_Allreduce(acc, acc_tot_sum, 4, MPI_GRAPH_TYPE, MPI_SUM, comm_);
            GraphWeight gtpos = (GraphWeight)(acc_tot_sum[1] / (GraphWeight)acc_tot_sum[0]);
            GraphWeight gtneg = (GraphWeight)(acc_tot_sum[2] / (GraphWeight)acc_tot_sum[0]);
            GraphWeight gfneg = (GraphWeight)(acc_tot_sum[3] / (GraphWeight)acc_tot_sum[0]);
            GraphWeight se = gtpos / (gtpos + gtneg);
            GraphWeight sp = gtpos / (gtpos + gfneg);
            GraphWeight d = 2*se*sp / (se + sp);

            if (std::isnan(d))
              d = PTOL;
            
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
            GraphElem tot = 0, tpos = 0, tneg = 0, fneg = 0;
            GraphElem ne = g_->get_ne();

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
                            const GraphWeight prob = (GraphWeight)(tail_freq_[edge_n.tail_] / (GraphWeight)(nv_));
                            GraphWeight pcut = genRandom<GraphWeight>(PTOL, 1.0); 
                            tup[1] = edge_n.tail_;
                            if (does_edge_exist(tup))
                            {
                                ntriangles_ += 1;
                                if (pcut <= prob)
                                    tpos += 1;
                                else
                                    fneg += 1;
                            }
                            else
                            {
                                if (pcut > prob)
                                    tneg += 1;
                            }

                            tot += 1;
                        }
                    }
                }
            }
             
            GraphElem acc[4] = {tot, tpos, tneg, fneg}, 
                      acc_tot_sum[4] = {0, 0, 0, 0};
            MPI_Allreduce(acc, acc_tot_sum, 4, MPI_GRAPH_TYPE, MPI_SUM, comm_);
            GraphWeight gtpos = (GraphWeight)(acc_tot_sum[1] / (GraphWeight)acc_tot_sum[0]);
            GraphWeight gtneg = (GraphWeight)(acc_tot_sum[2] / (GraphWeight)acc_tot_sum[0]);
            GraphWeight gfneg = (GraphWeight)(acc_tot_sum[3] / (GraphWeight)acc_tot_sum[0]);
            GraphWeight se = gtpos / (gtpos + gtneg);
            GraphWeight sp = gtpos / (gtpos + gfneg);
            GraphWeight d = 2*se*sp / (se + sp);

            if (std::isnan(d))
                d = PTOL;
            
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

        void calc_remote_triangles(const int pe, GraphWeight d = 1.0)
        {
            for (GraphElem i = 0; i < nv_; i++)
            {
                if (tail_freq_remote_[i] > 0)
                {
                    const GraphWeight prob = (GraphWeight)(tail_freq_remote_[i] / (GraphWeight)nv_);
                    if (genRandom<GraphWeight>(PTOL, d) <= prob)
                        remote_triangles_[pe] += 1;
                }
            }
        }

        inline GraphElem count()
        {
#if defined(USE_CUTOFF_PROBABILITY)
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
