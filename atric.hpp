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
#ifndef AGGR_TRIC_HPP
#define AGGR_TRIC_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>

#define AGGR_EDGE_SEARCH_TAG  101 
#define AGGR_EDGE_INVALID_TAG 201
#define AGGR_EDGE_VALID_TAG   301

class TriangulateAggr
{
    public:

        TriangulateAggr(Graph* g): 
            g_(g), sbuf_ctr_(nullptr), sreq_ctr_(0), 
            out_ghosts_(0), in_ghosts_(0), nghosts_(0), sbuf_(nullptr), 
            rbuf_(nullptr), sreq_(nullptr), send_counts_(0), recv_counts_(0), 
            ntriangles_(0)
        {
            comm_ = g_->get_comm();
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);
            send_counts_ = new GraphElem[size_];
            recv_counts_ = new GraphElem[size_];
            sbuf_ctr_ = new GraphElem[size_];
            memset(send_counts_, 0, sizeof(GraphElem)*size_);
            memset(sbuf_ctr_, 0, sizeof(GraphElem)*size_);
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
                if (p != rank_)
                    in_ghosts_ += recv_counts_[p];
            }
            nghosts_ = out_ghosts_ + in_ghosts_;
            sbuf_ = new GraphElem*[size_];
            for (int p = 0; p < size_; p++)
                sbuf_[p] = new GraphElem[send_counts_[p]*2];
            rbuf_ = new GraphElem[in_ghosts_*2];
            sreq_ = new MPI_Request[size_ + nghosts_ - 1];
        }

        ~TriangulateAggr() {}

        void clear()
        {
            for (int p = 0; p < size_; p++)
                delete []sbuf_[p];
            delete []sbuf_;
            delete []sreq_;
            delete []sbuf_ctr_;
            delete []send_counts_;
            delete []recv_counts_;
        }

        // TODO
        inline void check()
        {
        }
        
        inline void isend_nodata(int tag, int target)
        {
            MPI_Isend(&sbuf_[sbuf_ctr_[target]], 0, MPI_GRAPH_TYPE, 
                    target, tag, comm_, &sreq_[sreq_ctr_]);
            MPI_Request_free(&sreq_[sreq_ctr_]);
	    sreq_ctr_ += 1;
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
                    for (GraphElem n = m + 1; n < e1; n++)
                    {
                        Edge const& edge_n = g_->get_edge(n);
                        tup[1] = edge_n.tail_;
                        if (owner == rank_)
                        {
                            if (check_edgelist(tup))
                                ntriangles_ += 1;
                        }
                        else
                        {
                            sbuf_[owner][sbuf_ctr_[owner]]   = tup[0];
                            sbuf_[owner][sbuf_ctr_[owner]+1] = tup[1];
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

        inline void process_edges()
        {
            MPI_Status status;
            int flag = -1;
            int count = 0;
            GraphElem tup[2];
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_, 
                    &flag, &status);
            if (flag)
            {
                MPI_Get_count(&status, MPI_GRAPH_TYPE, &count);
                MPI_Recv(rbuf_, count, MPI_GRAPH_TYPE, status.MPI_SOURCE, 
                        status.MPI_TAG, comm_, MPI_STATUS_IGNORE);   
                if (status.MPI_TAG == AGGR_EDGE_SEARCH_TAG) 
                {
                    for (GraphElem k = 0; k < count; k+=2)
                    {
                        tup[0] = rbuf_[k];
                        tup[1] = rbuf_[k+1];
                        if (check_edgelist(tup))
                            isend_nodata(AGGR_EDGE_VALID_TAG, status.MPI_SOURCE);
                        else 
                            isend_nodata(AGGR_EDGE_INVALID_TAG, status.MPI_SOURCE);
                        nghosts_ -= 1;
                    }
                }
                else if (status.MPI_TAG == AGGR_EDGE_VALID_TAG)
                {
                    ntriangles_ += 1;
                    nghosts_ -= 1;
                }
                else
                    nghosts_ -= 1;
            }
        }

        inline GraphElem count()
        {
            lookup_edges();
            for (int p = 0; p < size_; p++)
            {
                if (p != rank_)
                {
                    MPI_Isend(sbuf_[p], send_counts_[p]*2, MPI_GRAPH_TYPE, 
                            p, AGGR_EDGE_SEARCH_TAG, comm_, &sreq_[sreq_ctr_]);
                    MPI_Request_free(&sreq_[sreq_ctr_]);
                    sreq_ctr_ += 1;
                }
            }
            while(1)
            {
                process_edges();
                if (nghosts_ == 0)
                    break;
            }
            MPI_Barrier(comm_);
            GraphElem ttc = 0, ltc = ntriangles_;
            MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
            return (ttc/3);
        }
       
    private:
        Graph* g_;
        GraphElem lnv_;
        GraphElem ntriangles_;
        GraphElem out_ghosts_, in_ghosts_, nghosts_;
	GraphElem **sbuf_, *rbuf_, *sbuf_ctr_;
        GraphElem *send_counts_, *recv_counts_;
        GraphElem sreq_ctr_;
        MPI_Request *sreq_;
	int rank_, size_;
        MPI_Comm comm_;
};
#endif
