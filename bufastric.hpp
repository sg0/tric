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
#ifndef BUF_CFASTRIC_HPP
#define BUF_CFASTRIC_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>

#ifndef TAG_DATA
#define TAG_DATA 100
#endif

#ifndef TAG_STAT
#define TAG_STAT 200
#endif

class TriangulateAggrBuffered
{
    public:

        TriangulateAggrBuffered(Graph* g, const GraphElem bufsize=DEFAULT_BUF_SIZE): 
            g_(g), sbuf_ctr_(nullptr), sbuf_(nullptr), rbuf_(nullptr),
            sreq_(nullptr), rinfo_(nullptr), srinfo_(nullptr), 
            ntriangles_(0), nghosts_(0), out_nghosts_(0), in_nghosts_(0), prev_n_(-1), 
            prev_m_(-1), prev_k_(-1), bufsize_(bufsize)
        {
            comm_ = g_->get_comm();
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);
            
            sbuf_ctr_ = new GraphElem[size_];
            rinfo_    = new GraphElem[size_];
            srinfo_   = new GraphElem[size_];
            sreq_     = new MPI_Request[size_-1];
            sbuf_     = new GraphElem[(size_-1)*bufsize_];
            rbuf_     = new GraphElem[bufsize_];

            std::fill(sreq_, sreq_ + (size_-1), MPI_REQUEST_NULL);
            std::memset(rinfo_, 0, sizeof(GraphElem)*size_);
            std::memset(sbuf_ctr_, 0, sizeof(GraphElem)*size_);

            GraphElem *send_count = new GraphElem[size_];
            GraphElem *recv_count = new GraphElem[size_];
            std::memset(send_count, 0, sizeof(GraphElem)*size_);
            std::memset(recv_count, 0, sizeof(GraphElem)*size_);
            
            const GraphElem lnv = g_->get_lnv();

            #pragma omp parallel for reduction(+:ntriangles_) default(shared)
            for (GraphElem i = 0; i < lnv; i++)
            {
                GraphElem e0, e1, tup[2];
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
                          #pragma omp atomic update
                          send_count[owner] += 1;
                        }
                    }
                }
            }

            MPI_Barrier(comm_);

            MPI_Alltoall(send_count, 1, MPI_GRAPH_TYPE, recv_count, 1, MPI_GRAPH_TYPE, comm_);

            for (GraphElem p = 0; p < size_; p++)
            {
              out_nghosts_ += send_count[p];
              in_nghosts_ += recv_count[p];
            }

            nghosts_ = out_nghosts_ + in_nghosts_;

            free(send_count);
            free(recv_count);
        }

        ~TriangulateAggrBuffered() {}

        void clear()
        {
            delete []sbuf_;
            delete []rbuf_;
            delete []sbuf_ctr_;
            delete []srinfo_;
            delete []rinfo_;
            delete []sreq_;
        }

        // TODO
        inline void check()
        {
        }

        void post_messages_reset(GraphElem target)
        {
            if (sbuf_ctr_[target] > 0)
            {
                const GraphElem idx = (target > rank_) ? (target-1) : target;
                MPI_Issend(&sbuf_[idx*bufsize_], sbuf_ctr_[target], 
                        MPI_GRAPH_TYPE, target, TAG_DATA, comm_, &sreq_[idx]);
                sbuf_ctr_[target] = 0;
            }
        }
         
        inline void post_messages_reset()
        {
            for (GraphElem p = 0; p < size_; p++)
            {
                if (p != rank_)
                    post_messages_reset(p);
            }
            
            prev_n_ = -2;
            prev_m_ = -2;
            prev_k_ = -2;
        }

        inline void lookup_edges()
        {
            if ((prev_n_ == -2) && (prev_m_ == -2) && (prev_k_ == -2))
                return;

            const GraphElem lnv = g_->get_lnv();
            for (GraphElem i = ((prev_n_ == -1) ? 0 : prev_n_); i < lnv; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                if ((e0 + 1) == e1)
                    continue;
                for (GraphElem m = ((prev_m_ == -1) ? e0 : prev_m_); m < e1-1; m++)
                {
                    Edge const& edge_m = g_->get_edge(m);
                    const int owner = g_->get_owner(edge_m.tail_);
                    if (owner != rank_)
                    {
                        if (sbuf_ctr_[owner] == bufsize_)
                        {
                            prev_n_ = i;
                            prev_m_ = m;
                            prev_k_ = -1;
                                
                            post_messages_reset(owner);

                            return;
                        }

                        const GraphElem disp = (owner > rank_) ? (owner-1)*bufsize_ : owner*bufsize_;
                        sbuf_[disp+sbuf_ctr_[owner]] = edge_m.tail_;
                        sbuf_ctr_[owner] += 1;

                        for (GraphElem n = ((prev_k_ == -1) ? (m + 1) : prev_k_); n < e1; n++)
                        {
                            if ((sbuf_ctr_[owner] + 1) == bufsize_)
                            {
                                prev_n_ = i;
                                prev_m_ = m;
                                prev_k_ = n;
                            
                                sbuf_[disp+sbuf_ctr_[owner]] = -1; // demarcate vertex boundary
                                sbuf_ctr_[owner] += 1;                               
                                
                                post_messages_reset(owner);

                                return;
                            }

                            Edge const& edge_n = g_->get_edge(n);
                            sbuf_[disp+sbuf_ctr_[owner]] = edge_n.tail_;
                            sbuf_ctr_[owner] += 1;
                        }
                        
                        sbuf_[disp+sbuf_ctr_[owner]] = -1; // demarcate vertex boundary
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
                if (edge.tail_ > tup[1]) 
                    break;
            }
            return false;
        }

        inline void process_messages()
        {
            MPI_Status status;
            int flag = -1;
            GraphElem tup[2] = {-1,-1}, source = -1, prev = 0, outg_counts = 0;
            int count = 0;
                           
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_, &flag, &status);

            if (flag)
            { 
                source = status.MPI_SOURCE;
                MPI_Get_count(&status, MPI_GRAPH_TYPE, &count);
                MPI_Recv(rbuf_, count, MPI_GRAPH_TYPE, source, 
                        status.MPI_TAG, comm_, MPI_STATUS_IGNORE);            
            }
            else
                return;

            for (GraphElem k = 0; k < count;)
            {
                if (rbuf_[k] == -1)
                    continue;

                tup[0] = rbuf_[k];
                GraphElem seg_count = 0;
               
                #pragma omp parallel for schedule(dynamic) \
                reduction(-:in_nghosts_) reduction(+:outg_counts) \
                default(shared)
                for (GraphElem m = k + 1; m < count; m++)
                {
                    if (rbuf_[m] == -1)
                    {
                      seg_count = m + 1;
                      break;
                    }

                    tup[1] = rbuf_[m];
                    
                    if (check_edgelist(tup))
                        outg_counts += 1;

                    in_nghosts_ -= 1;    
                }

                k += (seg_count - prev);
                prev = k;
                rinfo_[source] = outg_counts;
            }
        }

        inline GraphElem count()
        {
            bool done = false, nbar_active = false;
            MPI_Request nbar_req;

            while(!done)
            {
                lookup_edges();
                process_messages();

                if (nbar_active)
                {
                    int flag = -1;
                    MPI_Test(&nbar_req, &flag, MPI_STATUS_IGNORE);
                    done = flag ? true : false;
                }
                else
                {
                    int flag = -1;
                    MPI_Testall(size_-1, sreq_, &flag, MPI_STATUSES_IGNORE);
                    
                    if ((in_nghosts_ == 0) && flag)
                    {
                        MPI_Ibarrier(comm_, &nbar_req);
                        nbar_active = true;
                    }
                }
                
                // last chunk
                GraphElem *nchunk = std::max_element(sbuf_ctr_, sbuf_ctr_ + size_);
                if (*nchunk < bufsize_) 
                    post_messages_reset();
            }

            MPI_Alltoall(rinfo_, 1, MPI_GRAPH_TYPE, srinfo_, 1, MPI_GRAPH_TYPE, comm_);
            for (int p = 0; p < size_; p++)
                ntriangles_ += srinfo_[p];
            
            GraphElem ttc = 0, ltc = ntriangles_;
            MPI_Barrier(comm_);
            MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
            
            return (ttc/3);
        }

    private:
        Graph* g_;
        
        GraphElem ntriangles_;
        GraphElem prev_n_, prev_m_, prev_k_, bufsize_, nghosts_, out_nghosts_, in_nghosts_;
        GraphElem *sbuf_, *rbuf_, *sbuf_ctr_, *rinfo_, *srinfo_;
        MPI_Request *sreq_;
	
        int rank_, size_;
        MPI_Comm comm_;
};
#endif
