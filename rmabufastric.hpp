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
            g_(g), sbuf_ctr_(nullptr), wbuf_(nullptr), sbuf_(nullptr), win_(MPI_WIN_NULL), sreq_(nullptr), 
            pdegree_(-1), displs_(nullptr), scounts_(nullptr), rcounts_(nullptr),  vcount_(nullptr), 
            gcomm_(MPI_COMM_NULL), ntriangles_(0), nghosts_(0), out_nghosts_(0), in_nghosts_(0), pindex_(0), 
            prev_m_(nullptr), prev_k_(nullptr), stat_(nullptr), targets_(0), bufsize_(bufsize)
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
            GraphElem *tcount = new GraphElem[lnv];
            std::memset(tcount, 0, sizeof(GraphElem)*lnv);
            #pragma omp declare reduction(merge : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
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
                                tcount[i] += 1;
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
            ntriangles_ = std::accumulate(tcount, tcount + lnv, 0);
            free(tcount);
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
            
            MPI_Barrier(comm_);

            free(send_count);
            free(recv_count);

            MPI_Dist_graph_create_adjacent(comm_, targets_.size(), (const int*)targets_.data(), 
                    MPI_UNWEIGHTED, targets_.size(), (const int*)targets_.data(), MPI_UNWEIGHTED, 
                    MPI_INFO_NULL, 0 /*reorder ranks?*/, &gcomm_);
            
            MPI_Barrier(comm_);
            
            // double-checking indegree/outdegree
            int weighted, indegree, outdegree;
            MPI_Dist_graph_neighbors_count(gcomm_, &indegree, &outdegree, &weighted);
            assert(indegree == targets_.size());
            assert(outdegree == targets_.size());
            assert(indegree == outdegree);
            
            pdegree_ = indegree; // for undirected graph, indegree == outdegree
            
            for (int i = 0; i < pdegree_; i++)
                pindex_.insert({targets_[i], (GraphElem)i});

            sbuf_ctr_ = new GraphElem[pdegree_]();
            scounts_  = new GraphElem[pdegree_]();
            rcounts_  = new GraphElem[pdegree_]();
            prev_k_   = new GraphElem[pdegree_];
            prev_m_   = new GraphElem[pdegree_];
            stat_     = new char[pdegree_];
            sbuf_     = new GraphElem[pdegree_*bufsize_];
            displs_   = new GraphElem[pdegree_]();
            sreq_     = new MPI_Request[pdegree_];
            
            GraphElem disp = 0;

            for (GraphElem p = 0; p < pdegree_; p++)
            {
              prev_m_[p] = -1;
              prev_k_[p] = -1;
              stat_[p] = '0';
              sreq_[p] = MPI_REQUEST_NULL;
              scounts_[p] = disp;
              disp += bufsize_;
            }
            
            MPI_Barrier(comm_);

            MPI_Neighbor_alltoall(scounts_, 1, MPI_GRAPH_TYPE, 
                    displs_, 1, MPI_GRAPH_TYPE, gcomm_);
            
            MPI_Info info = MPI_INFO_NULL;
#if defined(USE_RMA_ACCUMULATE)
            MPI_Info_create(&info);
            MPI_Info_set(info, "accumulate_ordering", "none");
            MPI_Info_set(info, "accumulate_ops", "same_op");
#endif

            MPI_Win_allocate(pdegree_*bufsize_*sizeof(GraphElem), 
                    sizeof(GraphElem), info, comm_, &wbuf_, &win_);             
            MPI_Win_lock_all(MPI_MODE_NOCHECK, win_);
        }

        ~TriangulateAggrBufferedRMA() {}

        void clear()
        {
            MPI_Win_unlock_all(win_);
            MPI_Win_free(&win_);

            MPI_Comm_free(&gcomm_);

            delete []sbuf_;
            delete []sreq_;
            delete []sbuf_ctr_;
            delete []scounts_;
            delete []rcounts_;
            delete []prev_k_;
            delete []prev_m_;
            delete []stat_;
            delete []vcount_;
            delete []displs_;

            targets_.clear();
            pindex_.clear();
        }

        // TODO
        inline void check()
        {
        }

        inline void rput(GraphElem owner)
        {
            if (sbuf_ctr_[pindex_[owner]] > 0)
            {
#if defined(USE_RMA_ACCUMULATE)
                MPI_Raccumulate(&sbuf_[pindex_[owner]*bufsize_], sbuf_ctr_[pindex_[owner]], MPI_GRAPH_TYPE, owner, 
                        (MPI_Aint)(displs_[pindex_[owner]]), sbuf_ctr_[pindex_[owner]], MPI_GRAPH_TYPE, MPI_REPLACE, 
                        win_, &sreq_[pindex_[owner]]);
#else
                MPI_Rput(&sbuf_[pindex_[owner]*bufsize_], sbuf_ctr_[pindex_[owner]], MPI_GRAPH_TYPE, owner, 
                        (MPI_Aint)(displs_[pindex_[owner]]), sbuf_ctr_[pindex_[owner]], MPI_GRAPH_TYPE, 
                        win_, &sreq_[pindex_[owner]]);
#endif
            }
        }
        
        inline void rput()
        {
          for (int const& p : targets_)
              rput(p);
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
                  if (sbuf_ctr_[pidx] == (bufsize_-1))
                  {
                    prev_m_[pidx] = m;
                    prev_k_[pidx] = -1;
                    stat_[pidx]   = '1'; // messages in-flight

                    rput(owner);

                    continue;
                  }

                  const GraphElem disp = pidx*bufsize_;
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

                      rput(owner);

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
                      
                      rput(owner);
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

        inline void process_data()
        {
            MPI_Neighbor_alltoall(scounts_, 1, MPI_GRAPH_TYPE, 
                    rcounts_, 1, MPI_GRAPH_TYPE, gcomm_);

            for (GraphElem p = 0; p < pdegree_; p++)
            {
                if (rcounts_[p] > 0)
                {
                    GraphElem tup[2] = {-1,-1}, prev = 0;
                    for (GraphElem k = 0; k < rcounts_[p];)
                    {
                        if (wbuf_[p*bufsize_+k] == -1)
                            continue;

                        tup[0] = wbuf_[p*bufsize_+k];
                        GraphElem curr_count = 0;

                        for (GraphElem m = k + 1; m < rcounts_[p]; m++)
                        {
                            if (wbuf_[p*bufsize_+m] == -1)
                            {
                                curr_count = m + 1;
                                break;
                            }

                            tup[1] = wbuf_[p*bufsize_+m];

                            if (check_edgelist(tup))
                                ntriangles_ += 1; // valid edge 

                            in_nghosts_ -= 1;
                        }

                        k += (curr_count - prev);
                        prev = k;
                    }
                }
            }
        }

        inline GraphElem count()
        {
            bool sends_done = false;
            GraphElem count = 0;
            int *inds = new int[pdegree_];
            int over = -1;

            while(1)
            {  
              if (out_nghosts_ == 0)
              {
                  if (!sends_done)
                  {
                      rput();
                      sends_done = true;
                  }
              }
              else
                lookup_edges();
 
              MPI_Testsome(pdegree_, sreq_, &over, inds, MPI_STATUSES_IGNORE);
              std::fill(scounts_, scounts_ + pdegree_, 0);

              if (over != MPI_UNDEFINED)
              { 
                  for (int i = 0; i < over; i++)
                  {
                      scounts_[inds[i]] = sbuf_ctr_[inds[i]];
                      sbuf_ctr_[inds[i]] = 0;
                      stat_[inds[i]] = '0';
#if defined(USE_WIN_FLUSH)
                      MPI_Win_flush(targets_[inds[i]], win_);
#endif
                  }
#if defined(USE_WIN_FLUSH)
#else
                  MPI_Win_flush_all(win_);
#endif
              }
              
              process_data();
              
#if defined(USE_ALLREDUCE_FOR_EXIT)
              count = in_nghosts_;
              MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
#else
              std::fill(scounts_, scounts_ + pdegree_, in_nghosts_);
              MPI_Neighbor_alltoall(scounts_, 1, MPI_GRAPH_TYPE, 
                  rcounts_, 1, MPI_GRAPH_TYPE, gcomm_);
              count = std::accumulate(rcounts_, rcounts_ + pdegree_, 0);
#endif
              if (count == 0)
                  break;
#if defined(DEBUG_PRINTF)
              std::cout << "#incoming/outgoing count: " << in_nghosts_ << ", " << out_nghosts_ << std::endl;
#endif            
            }
            
            GraphElem ttc = 0, ltc = ntriangles_;
            MPI_Barrier(comm_);
            MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
            
            return (ttc/3);
        }

    private:
        Graph* g_;
        
        GraphElem ntriangles_, pdegree_, bufsize_, nghosts_, out_nghosts_, in_nghosts_;
        GraphElem *sbuf_, *wbuf_, *prev_k_, *prev_m_, *displs_, *sbuf_ctr_, 
                  *scounts_, *rcounts_, *vcount_;
        char *stat_;
        
        MPI_Request *sreq_;
        
        int rank_, size_;
        std::unordered_map<GraphElem, GraphElem> pindex_; 
        std::vector<GraphElem> targets_;
        MPI_Comm comm_, gcomm_;
        MPI_Win win_, cwin_;
};
#endif
