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
#ifndef RMABUF_CFASTRIC_HPP
#define RMABUF_CFASTRIC_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>

class TriangulateAggrBufferedRMA
{
    public:

        TriangulateAggrBufferedRMA(Graph* g, const GraphElem bufsize): 
            g_(g), sbuf_ctr_(nullptr), rbuf_(nullptr), sbuf_(nullptr), win_(MPI_WIN_NULL), sreq_(nullptr),
            pdegree_(-1), displs_(nullptr), scounts_(nullptr), rcounts_(nullptr), rinfo_(nullptr), 
            srinfo_(nullptr), vcount_(nullptr), gcomm_(MPI_COMM_NULL), ntriangles_(0), nghosts_(0), 
            out_nghosts_(0), in_nghosts_(0), pindex_(0), prev_m_(nullptr), prev_k_(nullptr), stat_(nullptr), 
            targets_(0), bufsize_(bufsize)
        {
            comm_ = g_->get_comm();
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);

            GraphElem *send_count = new GraphElem[size_];
            GraphElem *recv_count = new GraphElem[size_];
            std::memset(send_count, 0, sizeof(GraphElem)*size_);
            std::memset(recv_count, 0, sizeof(GraphElem)*size_);
            
            const GraphElem lnv = g_->get_lnv();
            vcount_ = new GraphElem[lnv];
            std::fill(vcount_, vcount_ + lnv, 0);

            double t0 = MPI_Wtime();

#if defined(USE_OPENMP)
            GraphElem *vcount = new GraphElem[lnv];
            std::memset(vcount, 0, sizeof(GraphElem)*lnv);
            #pragma omp declare reduction(merge : std::vector<GraphElem> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
            #pragma omp parallel for schedule(dynamic) reduction(merge: targets_) default(shared)
#endif            
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
                            {
#if defined(USE_OPENMP)
                                vcount[i] += 1;
#else
                                ntriangles_ += 1;
#endif
                            }
                        }
                    }
                    else
                    {
                        if (std::find(targets_.begin(), targets_.end(), owner) 
                                == targets_.end())
                            targets_.push_back(owner);

                        for (GraphElem n = m + 1; n < e1; n++)
                        {
#if defined(USE_OPENMP)
                          #pragma omp atomic update
#endif
                          send_count[owner] += 1;
                          vcount_[i] += 1;
                        }
                    }
                }
            }

#if defined(USE_OPENMP)
            ntriangles_ = std::accumulate(vcount, vcount + lnv, 0);
            free(vcount);
#endif

            MPI_Barrier(comm_);

            double t1 = MPI_Wtime();
            double p_tot = t1 - t0, t_tot = 0.0;
    
            MPI_Reduce(&p_tot, &t_tot, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
    
            if (rank_ == 0) 
            {   
                std::cout << "Average time for local counting during instantiation (secs.): " 
                    << ((double)(t_tot / (double)size_)) << std::endl;
            }

            MPI_Alltoall(send_count, 1, MPI_GRAPH_TYPE, recv_count, 1, MPI_GRAPH_TYPE, comm_);

            for (GraphElem p = 0; p < size_; p++)
            {
              out_nghosts_ += send_count[p];
              in_nghosts_ += recv_count[p];
            }

            nghosts_ = out_nghosts_ + in_nghosts_;

            free(send_count);
            free(recv_count);

            for (int i = 0; i < targets_.size(); i++)
                pindex_.insert({targets_[i], (GraphElem)i});

            pdegree_ = pindex_.size();
             
            sbuf_ctr_ = new GraphElem[pdegree_];
            scounts_  = new GraphElem[pdegree_];
            rcounts_  = new GraphElem[pdegree_];
            srinfo_   = new GraphElem[pdegree_];
            rinfo_    = new GraphElem[pdegree_];
            prev_k_   = new GraphElem[pdegree_];
            prev_m_   = new GraphElem[pdegree_];
            stat_     = new char[pdegree_];
            sbuf_     = new GraphElem[pdegree_*bufsize_];
            displs_   = new GraphElem[pdegree_];
            sreq_     = new MPI_Request[pdegree_];

            std::fill(sreq_, sreq_ + pdegree_, MPI_REQUEST_NULL);
            std::fill(prev_k_, prev_k_ + pdegree_, -1);
            std::fill(prev_m_, prev_m_ + pdegree_, -1);
            std::fill(scounts_, scounts_ + pdegree_, 0);
            std::fill(rcounts_, rcounts_ + pdegree_, 0);
            std::fill(rinfo_, rinfo_ + pdegree_, 0);
            std::fill(stat_, stat_ + pdegree_, '0');
            std::fill(sbuf_ctr_, sbuf_ctr_ + pdegree_, 0);
            std::fill(displs_, displs_ + pdegree_, 0);
                       
            MPI_Dist_graph_create_adjacent(comm_, pdegree_, (const int*)targets_.data(), 
                    MPI_UNWEIGHTED, pdegree_, (const int*)targets_.data(), MPI_UNWEIGHTED, 
                    MPI_INFO_NULL, 0 /*reorder ranks?*/, &gcomm_);

            GraphElem disp = 0;
            std::vector<GraphElem> displs(pdegree_);

            for (int t = 0; t < pdegree_; t++)
            {
                displs[t] = disp;
                disp += bufsize_;
            }
               
            MPI_Neighbor_alltoall(displs.data(), 1, MPI_GRAPH_TYPE, 
                    displs_, 1, MPI_GRAPH_TYPE, gcomm_);

            MPI_Win_allocate(pdegree_*bufsize_*sizeof(GraphElem), 
                    sizeof(GraphElem), MPI_INFO_NULL, comm_, &rbuf_, &win_);             
 
            MPI_Win_lock_all(MPI_MODE_NOCHECK, win_);
            
            displs.clear();
        }

        ~TriangulateAggrBufferedRMA() {}

        void clear()
        {
            delete []sbuf_;
            delete []sreq_;
            delete []sbuf_ctr_;
            delete []scounts_;
            delete []rcounts_;
            delete []srinfo_;
            delete []rinfo_;
            delete []prev_k_;
            delete []prev_m_;
            delete []stat_;
            delete []vcount_;
            delete []displs_;

            targets_.clear();
            pindex_.clear();
             
            MPI_Win_unlock_all(win_);
            MPI_Win_free(&win_);
            MPI_Comm_free(&gcomm_);
        }

        // TODO
        inline void check()
        {
        }

        void nbput(GraphElem owner)
        {
            if (sbuf_ctr_[pindex_[owner]] > 0)
            {
#if defined(USE_MPI_ACCUMULATE)
                MPI_Raccumulate(&sbuf_[pindex_[owner]*bufsize_], sbuf_ctr_[pindex_[owner]], MPI_GRAPH_TYPE, owner, 
                        (MPI_Aint)(displs_[pindex_[owner]] + sbuf_ctr_[pindex_[owner]]), 
                        sbuf_ctr_[pindex_[owner]], MPI_GRAPH_TYPE, MPI_REPLACE, win_, &sreq_[pindex_[owner]]);
#else
                MPI_Rput(&sbuf_[pindex_[owner]*bufsize_], sbuf_ctr_[pindex_[owner]], MPI_GRAPH_TYPE, owner, 
                        (MPI_Aint)(displs_[pindex_[owner]] + sbuf_ctr_[pindex_[owner]]), 
                        sbuf_ctr_[pindex_[owner]], MPI_GRAPH_TYPE, win_, &sreq_[pindex_[owner]]);
#endif
            }
        }
        
        void nbput()
        {
          for (GraphElem p : targets_)
              nbput(p);
        }
        
        inline void lookup_edges()
        {
          const GraphElem lnv = g_->get_lnv();
          for (GraphElem i = 0; i < lnv; i++)
          {
            if (vcount_[i] == 0) // all edges processed, move on
                continue;

            GraphElem e0, e1;
            g_->edge_range(i, e0, e1);

            if ((e0 + 1) == e1)
              continue;

            for (GraphElem m = e0; m < e1-1; m++)
            {
              EdgeStat& edge = g_->get_edge_stat(m);
              const int owner = g_->get_owner(edge.edge_->tail_);
              const GraphElem pidx = pindex_[owner];

              if (owner != rank_ && edge.active_)
              {               
                if (stat_[pidx] == '1') 
                  continue;

                if (m >= prev_m_[pidx])
                {
                  const GraphElem disp = pidx*bufsize_;
                  
                  if (sbuf_ctr_[pidx] == (bufsize_-1))
                  {
                    prev_m_[pidx] = m;
                    prev_k_[pidx] = -1;
                    stat_[pidx]   = '1'; // messages in-flight

                    nbput(owner);

                    continue;
                  }

                  sbuf_[disp+sbuf_ctr_[pidx]] = edge.edge_->tail_;
                  sbuf_ctr_[pidx] += 1;

                  for (GraphElem n = ((prev_k_[pidx] == -1) ? (m + 1) : prev_k_[pidx]); n < e1; n++)
                  {
                    if (sbuf_ctr_[pidx] == (bufsize_-1))
                    {
                      prev_m_[pidx] = m;
                      prev_k_[pidx] = n;

                      sbuf_[disp+sbuf_ctr_[pidx]] = -1; // demarcate vertex boundary
                      sbuf_ctr_[pidx] += 1;
                      stat_[pidx] = '1'; 

                      nbput(owner);

                      break;
                    }

                    Edge const& edge_n = g_->get_edge(n);
                    sbuf_[disp+sbuf_ctr_[pidx]] = edge_n.tail_;
                    sbuf_ctr_[pidx] += 1;
                    out_nghosts_ -= 1;
                    vcount_[i] -= 1;
                  }

                  if (stat_[pidx] == '0') 
                  {
                    prev_m_[pidx] = m;
                    prev_k_[pidx] = -1;

                    edge.active_ = false;

                    if (sbuf_ctr_[pidx] == (bufsize_-1))
                    {
                      sbuf_[disp+sbuf_ctr_[pidx]] = -1; 
                      sbuf_ctr_[pidx] += 1;
                      stat_[pidx] = '1';

                      nbput(owner);
                    }
                    else
                    {
                      sbuf_[disp+sbuf_ctr_[pidx]] = -1; 
                      sbuf_ctr_[pidx] += 1;
                    }
                  }
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
                if (edge.tail_ > tup[1]) 
                    break;
            }
            return false;
        }

        inline void process_messages()
        {     
            MPI_Win_flush_all(win_);
            
            MPI_Neighbor_alltoall(scounts_, 1, MPI_GRAPH_TYPE, 
                    rcounts_, 1, MPI_GRAPH_TYPE, gcomm_);

            GraphElem count = std::accumulate(rcounts_, rcounts_ + pdegree_, 0);
            
            if (count == 0)
                return;

            for (GraphElem p = 0; p < pdegree_; p++)
            {
                GraphElem tup[2] = {-1,-1}, prev = 0;
                
                for (GraphElem k = 0; k < rcounts_[p];)
                {
                    if (rbuf_[p*bufsize_+k] == -1)
                        continue;

                    tup[0] = rbuf_[p*bufsize_+k];
                    GraphElem curr_count = 0;

                    for (GraphElem m = k + 1; m < rcounts_[p]; m++)
                    {
                        if (rbuf_[p*bufsize_+m] == -1)
                        {
                            curr_count = m + 1;
                            break;
                        }

                        tup[1] = rbuf_[p*bufsize_+m];

                        if (check_edgelist(tup))
                            rinfo_[p] += 1; // valid edge 

                        in_nghosts_ -= 1;
                    }

                    k += (curr_count - prev);
                    prev = k;
                }
            }
        }

        inline GraphElem count()
        {
            bool done = false, nbar_active = false, sends_done = false;
            MPI_Request nbar_req = MPI_REQUEST_NULL;
            int *inds = new int[pdegree_];
            int over = -1;

            while(!done)
            {  
              if (nbar_active)
              {
                int test_nbar = -1;
                MPI_Test(&nbar_req, &test_nbar, MPI_STATUS_IGNORE);
                done = !test_nbar ? false : true;
              }
              else
              {
                if (in_nghosts_ == 0)
                {
                  MPI_Ibarrier(comm_, &nbar_req);
                  nbar_active = true;
                }
              }

              if (out_nghosts_ == 0)
              {
                  if (!sends_done)
                  {
                      nbput();
                      sends_done = true;
                  }
              }
              else
                lookup_edges();
              
              std::fill(scounts_, scounts_ + pdegree_, 0);
              MPI_Testsome(pdegree_, sreq_, &over, inds, MPI_STATUSES_IGNORE);

              if (over != MPI_UNDEFINED)
              { 
                  for (int i = 0; i < over; i++)
                  {
                      scounts_[inds[i]] = sbuf_ctr_[inds[i]];
                      sbuf_ctr_[inds[i]] = 0;
                      stat_[inds[i]] = '0';
                  } 
              }

              process_messages();

#if defined(DEBUG_PRINTF)
              std::cout << "in/out: " << in_nghosts_ << ", " << out_nghosts_ << std::endl;
#endif            
            }
            
            MPI_Neighbor_alltoall(rinfo_, 1, MPI_GRAPH_TYPE, srinfo_, 
                    1, MPI_GRAPH_TYPE, gcomm_);
            
            for (int p = 0; p < pdegree_; p++)
                ntriangles_ += srinfo_[p];
            
            GraphElem ttc = 0, ltc = ntriangles_;
            MPI_Barrier(comm_);
            MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
            
            return (ttc/3);
        }

    private:
        Graph* g_;
        
        GraphElem ntriangles_, pdegree_, bufsize_, nghosts_, out_nghosts_, in_nghosts_;
        GraphElem *sbuf_, *rbuf_, *prev_k_, *prev_m_, *displs_, *sbuf_ctr_, *scounts_, *rcounts_, *rinfo_, *srinfo_, *vcount_;
        char *stat_;
        
        MPI_Request *sreq_;
        
        int rank_, size_;
        std::unordered_map<GraphElem, GraphElem> pindex_; 
        std::vector<GraphElem> targets_;
        MPI_Comm comm_, gcomm_;
        MPI_Win win_;
};
#endif
