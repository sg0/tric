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

#define PACK_SIZE (64)

class TriangulateAggrFatDtype
{
    public:

        TriangulateAggrFatDtype(Graph* g): 
            g_(g), sbuf_ctr_(nullptr), sbuf_disp_(nullptr), 
            out_ghosts_(0), in_ghosts_(0), nghosts_(0), sbuf_(nullptr), 
            send_counts_(0), recv_counts_(0), ntriangles_(0)  
        {
            comm_ = g_->get_comm();
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);
            send_counts_ = new GraphElem[size_];
            recv_counts_ = new GraphElem[size_];
            sbuf_ctr_ = new GraphElem[size_];
            sbuf_disp_ = new GraphElem[size_];
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
            GraphElem pos = 0;
            for (int p = 0; p < size_; p++)
            {
                sbuf_disp_[p] = pos;
                in_ghosts_ += recv_counts_[p];
                out_ghosts_ += send_counts_[p];
                send_counts_[p] = roundUp(send_counts_[p]*2, PACK_SIZE); 
                recv_counts_[p] = roundUp(recv_counts_[p]*2, PACK_SIZE);
                pos += send_counts_[p];
            }
            nghosts_ = out_ghosts_ + in_ghosts_;
            sbuf_ = new GraphElem[pos];
            std::memset(sbuf_, 0, pos*sizeof(GraphElem));
        }

        ~TriangulateAggrFatDtype() {}

        void clear()
        {
            delete []sbuf_;
            delete []sbuf_ctr_;
            delete []sbuf_disp_;
            delete []send_counts_;
            delete []recv_counts_;
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
            GraphElem spos=0, rpos=0, dpos=0;
            int *sdispls = new int[size_];
            int *rdispls = new int[size_];
            GraphElem *rptr = new GraphElem[size_+1];
            GraphElem *rinfo = new GraphElem[size_*2];
            GraphElem *srinfo = new GraphElem[size_*2];
            std::memset(srinfo, 0, sizeof(GraphElem)*size_*2);
            std::memset(rinfo, 0, sizeof(GraphElem)*size_*2);
            GraphElem *rbuf = new GraphElem[in_ghosts_*2];
            int *scnts = new int[size_];
            int *rcnts = new int[size_];
            MPI_Datatype pack_t;
            MPI_Type_contiguous(PACK_SIZE, MPI_GRAPH_TYPE, &pack_t);
            MPI_Type_commit(&pack_t);
            for (int p = 0; p < size_; p++) 
            {
                rptr[p] = dpos;
                sdispls[p] = spos;
                rdispls[p] = rpos;
                scnts[p] = (int)(send_counts_[p] / PACK_SIZE);
                rcnts[p] = (int)(recv_counts_[p] / PACK_SIZE);
                spos += scnts[p];
                rpos += rcnts[p];
                dpos += recv_counts_[p];
            }
            rptr[size_] = dpos;
            MPI_Alltoallv(sbuf_, scnts, sdispls, pack_t, 
                    rbuf, rcnts, rdispls, pack_t, comm_);
            // EDGE_SEARCH_TAG
            GraphElem tup[2];
            for (int p = 0; p < size_; p++)
            {
                for (GraphElem k = rptr[p]; k < rptr[p+1]; k+=2)
                {
                    tup[0] = rbuf[k];
                    tup[1] = rbuf[k+1];
                    if (check_edgelist(tup))
                        rinfo[p*2] += 1;   // 0 == EDGE_VALID_TAG 
                    else 
                        rinfo[p*2+1] += 1; // 1 == EDGE_INVALID_TAG 
                    nghosts_ -= 1;
                }
            }
            // communication step 2
            MPI_Alltoall(rinfo, 2, MPI_GRAPH_TYPE, srinfo, 2, MPI_GRAPH_TYPE, comm_);
            for (int p = 0; p < size_; p++)
            {
                ntriangles_ += srinfo[p*2];
                nghosts_ -= srinfo[p*2+1];
            }
            MPI_Barrier(comm_);
            MPI_Type_free(&pack_t);
            delete []sdispls;
            delete []rdispls;
            delete []rinfo;
            delete []srinfo;
            delete []rbuf;
            delete []scnts;
            delete []rcnts;
            delete []rptr;
            GraphElem ttc = 0, ltc = ntriangles_;
            MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
            return (ttc/3);
        }
    private:
        Graph* g_;
        GraphElem ntriangles_;
        GraphElem out_ghosts_, in_ghosts_, nghosts_;
	GraphElem *sbuf_, *sbuf_ctr_, *sbuf_disp_;
        GraphElem *send_counts_, *recv_counts_;
	int rank_, size_;
        MPI_Comm comm_;
};
#endif
