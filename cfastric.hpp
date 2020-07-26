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
#ifndef AGGR_CFASTRIC_HPP
#define AGGR_CFASTRIC_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>

#ifdef SET_BATCH_SIZE
#ifndef DEFAULT_BATCH_SIZE
#define DEFAULT_BATCH_SIZE   (1073741824)
#endif
#endif

class TriangulateAggrFatCompressed
{
    public:

        TriangulateAggrFatCompressed(Graph* g): 
            g_(g), sbuf_ctr_(nullptr), sbuf_disp_(nullptr), 
            out_ghosts_(0), in_ghosts_(0), nghosts_(0), sbuf_(nullptr), 
            send_counts_(nullptr), recv_counts_(nullptr), ntriangles_(0) 
        {
            comm_ = g_->get_comm();
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);
            send_counts_ = new GraphElem[size_];
            recv_counts_ = new GraphElem[size_];
            sbuf_ctr_ = new GraphElem[size_];
            sbuf_disp_ = new GraphElem[size_];
            std::memset(send_counts_, 0, sizeof(GraphElem)*size_);
            std::memset(sbuf_ctr_, 0, sizeof(GraphElem)*size_);
            std::memset(sbuf_disp_, 0, sizeof(GraphElem)*(size_));
            GraphElem *pad_counts = new GraphElem[size_];
            std::memset(pad_counts, 0, sizeof(GraphElem)*(size_));
            const GraphElem lnv = g_->get_lnv();
            for (GraphElem i = 0; i < lnv; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                if ((e0 + 1) == e1)
                    continue;
                // calculate ghosts
                for (GraphElem m = e0; m < e1-1; m++)
                {
                    Edge const& edge_m = g_->get_edge(m);
                    const int owner = g_->get_owner(edge_m.tail_);
                    if (owner != rank_)
                    {
                        for (GraphElem n = m + 1; n < e1; n++)
                            send_counts_[owner] += 1; 
                        pad_counts[owner] += 2;  // extras for head vertex and demarcating vertices
                    }
                }
            }
            MPI_Alltoall(send_counts_, 1, MPI_GRAPH_TYPE, recv_counts_, 1, MPI_GRAPH_TYPE, comm_);
            GraphElem spos = 0, rpos = 0;
            for (int p = 0; p < size_; p++)
            {
                sbuf_disp_[p] = spos;
                out_ghosts_ += send_counts_[p];
                in_ghosts_ += recv_counts_[p];
                send_counts_[p] += pad_counts[p];
                spos += send_counts_[p];
            }
            sbuf_ = new GraphElem[spos];
            nghosts_ = out_ghosts_ + in_ghosts_;
            MPI_Alltoall(send_counts_, 1, MPI_GRAPH_TYPE, recv_counts_, 1, MPI_GRAPH_TYPE, comm_);
            for (int p = 0; p < size_; p++)
            {
                rpos += recv_counts_[p];
            }
            rbuf_ = new GraphElem[rpos];
            delete []pad_counts;
        }

        ~TriangulateAggrFatCompressed() {}

        void clear()
        {
            delete []sbuf_;
            delete []rbuf_;
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
                        sbuf_[sbuf_disp_[owner]+sbuf_ctr_[owner]] = edge_m.tail_;
                        sbuf_ctr_[owner] += 1;
                        for (GraphElem n = m + 1; n < e1; n++)
                        {
                            Edge const& edge_n = g_->get_edge(n);
                            sbuf_[sbuf_disp_[owner]+sbuf_ctr_[owner]] = edge_n.tail_;
                            sbuf_ctr_[owner] += 1;
                        }
                        sbuf_[sbuf_disp_[owner]+sbuf_ctr_[owner]] = -1; // demarcate vertex boundary
                        sbuf_ctr_[owner] += 1;
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
            GraphElem *rinfo = new GraphElem[size_*2];
            GraphElem *srinfo = new GraphElem[size_*2];
            GraphElem *rptr = new GraphElem[size_+1];
            int *sdispls = new int[size_];
            int *rdispls = new int[size_];
            int *scnts = new int[size_];
            int *rcnts = new int[size_];
            std::memset(rptr, 0, sizeof(GraphElem)*(size_+1));
            std::memset(rinfo, 0, sizeof(GraphElem)*size_*2);
            // batch configuration
            int nbatches = 1;
            GraphElem max_send_count = *std::max_element(send_counts_, send_counts_+size_);
            MPI_Allreduce(MPI_IN_PLACE, &max_send_count, 1, MPI_GRAPH_TYPE, MPI_MAX, comm_);
#if defined(SET_BATCH_SIZE)
            GraphElem batch_size = DEFAULT_BATCH_SIZE;
#else 
            GraphElem batch_size = (GraphElem)std::numeric_limits<int>::max();
#endif            
            while(batch_size < max_send_count)
            {
                nbatches += 1;
#if defined(SET_BATCH_SIZE)
                batch_size += (GraphElem)DEFAULT_BATCH_SIZE;
#else 
                batch_size += (GraphElem)std::numeric_limits<int>::max();
#endif            
            }
            int *batch_send_counts = new int[size_*nbatches];
            int *batch_recv_counts = new int[size_*nbatches];
            std::memset(batch_send_counts, 0, size_*nbatches*sizeof(int));
            for (int p = 0; p < size_; p++)
            {
                if (send_counts_[p] > 0)
                {
                    for (int i = 0; i < nbatches; i++)
                    {
#if defined(SET_BATCH_SIZE)
                        batch_send_counts[p*nbatches+i] = MIN(DEFAULT_BATCH_SIZE, send_counts_[p]);
#else 
                        batch_send_counts[p*nbatches+i] = MIN(std::numeric_limits<int>::max(), send_counts_[p]);
#endif            
                        send_counts_[p] -= batch_send_counts[p*nbatches+i];
                    }
                }
            }
            MPI_Alltoall(batch_send_counts, nbatches, MPI_INT, batch_recv_counts, nbatches, MPI_INT, comm_);
            GraphElem spos = 0, rpos = 0;
            // batched communication
            for (int n = 0; n < nbatches; n++)
            {
#ifdef USE_ALLTOALLV
                for (int p = 0; p < size_; p++)
                {
                    sdispls[p] = (int)spos;
                    rdispls[p] = (int)rpos;
                    scnts[p] = batch_send_counts[p*nbatches+n];
                    rcnts[p] = batch_recv_counts[p*nbatches+n];
                    spos += scnts[p];
                    rpos += rcnts[p];
                }
                MPI_Alltoallv(sbuf_, scnts, sdispls, MPI_GRAPH_TYPE, 
                        rbuf_, rcnts, rdispls, MPI_GRAPH_TYPE, comm_);
#else
                std::vector<MPI_Request> reqs(size_*2, MPI_REQUEST_NULL);
                for (int p = 0; p < size_; p++)
                {
                    rcnts[p] = batch_recv_counts[p*nbatches+n];
                    if (p != rank_)
                        MPI_Irecv(rbuf_ + rpos, rcnts[p], MPI_GRAPH_TYPE, p, 101, comm_, &reqs[p]);
                    else
                        reqs[p] = MPI_REQUEST_NULL;
                    rpos += rcnts[p];
                }
                for (int p = 0; p < size_; p++)
                {
                    scnts[p] = batch_send_counts[p*nbatches+n];
                    if (p != rank_)
                        MPI_Isend(sbuf_ + spos, scnts[p], MPI_GRAPH_TYPE, p, 101, comm_, &reqs[p+size_]);
                    else
                        reqs[p+size_] = MPI_REQUEST_NULL;
                    spos += scnts[p];
                }
                MPI_Waitall(size_*2, reqs.data(), MPI_STATUSES_IGNORE);
#endif
            } // end of batches
            rpos = 0;
            for (int p = 0; p < size_; p++)
            {
                rptr[p] = rpos;
                rpos += recv_counts_[p];
            }
            rptr[size_] = rpos;
            MPI_Barrier(comm_);
            // EDGE_SEARCH_TAG
            GraphElem tup[2], prev = 0;
            for (int p = 0; p < size_; p++)
            {
                if (rptr[p] != rptr[p+1])
                {
                    for (GraphElem k = rptr[p]; k < rptr[p+1]-1;)
                    {
                        if (rbuf_[k] == -1)
                            continue;
                        tup[0] = rbuf_[k];
                        GraphElem count = 0;
                        for (GraphElem m = k + 1; m < rptr[p+1]; m++)
                        {
                            if (rbuf_[m] == -1)
                            {
                                count = m + 1;
                                break;
                            }
                            tup[1] = rbuf_[m];
                            if (check_edgelist(tup))
                                rinfo[p*2] += 1;   // 0 == EDGE_VALID_TAG 
                            else 
                                rinfo[p*2+1] += 1; // 1 == EDGE_INVALID_TAG 
                            nghosts_ -= 1;
                        }
                        k += (count - prev);
                        prev = k;
                    }
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
            delete []rptr;
            delete []sdispls;
            delete []rdispls;
            delete []scnts;
            delete []rcnts;
            delete []batch_send_counts;
            delete []batch_recv_counts;
            GraphElem ttc = 0, ltc = ntriangles_;
            MPI_Barrier(comm_);
            MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
            if (rank_ == 0)
                std::cout << "Number of communication batches: " << nbatches << std::endl;
            return (ttc/3);
        }
    private:
        Graph* g_;
        GraphElem ntriangles_;
        GraphElem out_ghosts_, in_ghosts_, nghosts_;
	GraphElem *sbuf_, *rbuf_, *sbuf_ctr_, *sbuf_disp_;
        GraphElem *send_counts_, *recv_counts_;
	int rank_, size_;
        MPI_Comm comm_;
};
#endif
