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
#ifndef AGGR_DFATRIC_HPP
#define AGGR_DFATRIC_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>

#define PAD_SIZE (1000)

class TriangulateAggrFatDtype
{
    public:

        TriangulateAggrFatDtype(Graph* g): 
            g_(g), sbuf_ctr_(nullptr), sbuf_disp_(nullptr), 
            out_ghosts_(0), in_ghosts_(0), nghosts_(0), sbuf_(nullptr), rbuf_(nullptr), 
            send_counts_(nullptr), recv_counts_(nullptr), ntriangles_(0),
            sdispls_(nullptr), rdispls_(nullptr), scnts_(nullptr), rcnts_(nullptr),
            send_t_(nullptr), recv_t_(nullptr), rptr_(nullptr)
        {
            comm_ = g_->get_comm();
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);
            send_counts_ = new GraphElem[size_];
            recv_counts_ = new GraphElem[size_];
            sbuf_ctr_ = new GraphElem[size_];
            sdispls_ = new int[size_];
            rdispls_ = new int[size_];
            scnts_ = new int[size_];
            rcnts_ = new int[size_];
            send_t_ = new MPI_Datatype[size_];
            recv_t_ = new MPI_Datatype[size_];
            sbuf_disp_ = new GraphElem[size_];
            rptr_ = new GraphElem[size_+1];
            std::memset(send_counts_, 0, sizeof(GraphElem)*size_);
            std::memset(recv_counts_, 0, sizeof(GraphElem)*size_);
            std::memset(sbuf_ctr_, 0, sizeof(GraphElem)*size_);
            std::memset(sbuf_disp_, 0, sizeof(GraphElem)*(size_));
            const GraphElem lnv = g_->get_lnv();
            for (GraphElem i = 0; i < lnv; i++)
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
                        for (GraphElem n = m + 1; n < e1; n++)
                            send_counts_[owner] += 1;
                    }
                }
            }
            MPI_Alltoall(send_counts_, 1, MPI_GRAPH_TYPE, recv_counts_, 1, MPI_GRAPH_TYPE, comm_);
            for (int p = 0; p < size_; p++)
            {
                out_ghosts_ += send_counts_[p];
                in_ghosts_ += recv_counts_[p];
            }
            nghosts_ = out_ghosts_ + in_ghosts_;
            GraphElem spos = 0, rpos = 0;
            for (int p = 0; p < size_; p++)
            {
                sbuf_disp_[p] = spos;
                sdispls_[p] = (int)(spos*sizeof(GraphElem));
                if ((send_counts_[p]*2) < std::numeric_limits<int>::max())
                {
                    MPI_Type_contiguous(1, MPI_GRAPH_TYPE, &send_t_[p]);
                    send_counts_[p] *= 2;
                    scnts_[p] = send_counts_[p];
                }
                else 
                {
                    send_counts_[p] = roundUp(send_counts_[p]*2, PAD_SIZE);
                    MPI_Type_contiguous(PAD_SIZE, MPI_GRAPH_TYPE, &send_t_[p]);
                    scnts_[p] = (int)((GraphElem)(send_counts_[p] / (GraphElem)PAD_SIZE));
                }
                spos += send_counts_[p];
                MPI_Type_commit(&send_t_[p]);
            }
            sbuf_ = new GraphElem[spos];
            std::memset(sbuf_, 0, spos*sizeof(GraphElem));
            MPI_Alltoall(send_counts_, 1, MPI_GRAPH_TYPE, recv_counts_, 1, MPI_GRAPH_TYPE, comm_);
            for (int p = 0; p < size_; p++) 
            {
                rptr_[p] = rpos;
                rdispls_[p] = (int)(rpos*sizeof(GraphElem));
                if ((recv_counts_[p]*2) < std::numeric_limits<int>::max())
                {
                    MPI_Type_contiguous(1, MPI_GRAPH_TYPE, &recv_t_[p]);
                    recv_counts_[p] *= 2;
                    rcnts_[p] = recv_counts_[p];
                }
                else 
                {
                    recv_counts_[p] = roundUp(recv_counts_[p]*2, PAD_SIZE);
                    MPI_Type_contiguous(PAD_SIZE, MPI_GRAPH_TYPE, &recv_t_[p]);
                    rcnts_[p] = (int)((GraphElem)(recv_counts_[p] / (GraphElem)PAD_SIZE));
                }
                rpos += recv_counts_[p];
                MPI_Type_commit(&recv_t_[p]);
            }
            rptr_[size_] = rpos;
            rbuf_ = new GraphElem[rpos];
        }

        ~TriangulateAggrFatDtype() {}

        void clear()
        {
            delete []sbuf_;
            delete []rbuf_;
            delete []sbuf_ctr_;
            delete []sbuf_disp_;
            delete []rptr_;
            delete []send_counts_;
            delete []recv_counts_;
            delete []sdispls_;
            delete []rdispls_;
            delete []scnts_;
            delete []rcnts_;
            for (int p = 0; p < size_; p++)
            {
                MPI_Type_free(&send_t_[p]);
                MPI_Type_free(&recv_t_[p]);
            }
            delete []send_t_;
            delete []recv_t_;
        }

        // TODO
        inline void check()
        {
        }
        
        inline void lookup_edges()
        {
            GraphElem tup[2];
            const GraphElem lnv = g_->get_lnv();
            for (GraphElem i = 0; i < lnv; i++)
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
                            if (check_edgelist(tup))
                                ntriangles_ += 1;
                        }
                    }
                    else
                    {
                        for (GraphElem n = m + 1; n < e1; n++)
                        {
                            Edge const& edge_n = g_->get_edge(n);
                            tup[1] = edge_n.tail_;
                            sbuf_[sbuf_disp_[owner]+sbuf_ctr_[owner]]   = tup[0];
                            sbuf_[sbuf_disp_[owner]+sbuf_ctr_[owner]+1] = tup[1];
                            sbuf_ctr_[owner] += 2;
                        }
                    }
                }
            }
        }
        
        inline bool check_edgelist(GraphElem tup[2])
        {
            GraphElem e0, e1;
            const GraphElem lv = g_->global_to_local(tup[0]);
            g_->edge_range(lv, e0, e1);
            for (GraphElem e = e0; e < e1; e++)
            {
                Edge const& edge = g_->get_edge(e);
                if (tup[1] == edge.tail_)
                    return true;
                if (edge.tail_ > tup[1]) // edge-list is sorted
                    break;
            }
            return false;
        }

        inline GraphElem count()
        {
            // local computation
            lookup_edges();
            MPI_Barrier(comm_);
            // communication step 1
            GraphElem *rinfo = new GraphElem[size_*2];
            GraphElem *srinfo = new GraphElem[size_*2];
            std::memset(srinfo, 0, sizeof(GraphElem)*size_*2);
            std::memset(rinfo, 0, sizeof(GraphElem)*size_*2);
            MPI_Alltoallw(sbuf_, scnts_, sdispls_, send_t_, 
                    rbuf_, rcnts_, rdispls_, recv_t_, comm_);
            // EDGE_SEARCH_TAG
            GraphElem tup[2];
            for (int p = 0; p < size_; p++)
            {
                for (GraphElem k = rptr_[p]; k < rptr_[p+1]; k+=2)
                {
                    tup[0] = rbuf_[k];
                    tup[1] = rbuf_[k+1];
                    if (check_edgelist(tup))
                        rinfo[p*2] += 1;   // 0 == EDGE_VALID_TAG 
                    else 
                        rinfo[p*2+1] += 1; // 1 == EDGE_INVALID_TAG 
                    nghosts_ -= 1;
                }
            }
            MPI_Barrier(comm_);
            // communication step 2
            MPI_Alltoall(rinfo, 2, MPI_GRAPH_TYPE, srinfo, 2, MPI_GRAPH_TYPE, comm_);
            for (int p = 0; p < size_; p++)
            {
                ntriangles_ += srinfo[p*2];
                nghosts_ -= srinfo[p*2+1];
            }
            delete []rinfo;
            delete []srinfo;
            GraphElem ttc = 0, ltc = ntriangles_;
            MPI_Barrier(comm_);
            MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
            return (ttc/3);
        }
    private:
        Graph* g_;
        GraphElem ntriangles_;
        GraphElem out_ghosts_, in_ghosts_, nghosts_;
	GraphElem *sbuf_, *rbuf_, *sbuf_ctr_, *sbuf_disp_, *rptr_;
        GraphElem *send_counts_, *recv_counts_;
	int rank_, size_;
        MPI_Comm comm_;
        int *sdispls_, *rdispls_, *scnts_, *rcnts_;
        MPI_Datatype *send_t_, *recv_t_;
};
#endif
