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
#ifndef BFSTRIC_HPP
#define BFSTRIC_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string> 

#ifndef DEF_BFS_BUFSIZE
#define DEF_BFS_BUFSIZE (256)
#endif

#ifndef DEF_BFS_SEED
#define DEF_BFS_SEED (2)
#endif

class BFS
{
    public:
        BFS(Graph* g): 
            g_(g), visited_(nullptr), pred_(nullptr), bufsize_(DEF_BFS_BUFSIZE),
            comm_(MPI_COMM_NULL), rank_(MPI_PROC_NULL), size_(0), 
            ract_(0), sact_(nullptr), sctr_(nullptr),
            sbuf_(nullptr), rbuf_(nullptr), sreq_(nullptr), rreq_(MPI_REQUEST_NULL), 
            oldq_(nullptr), newq_(nullptr), nranks_done_(0), newq_count_(0), 
            oldq_count_(0), seed_(DEF_BFS_SEED), edge_visit_count_(0)
        {
            const GraphElem lnv = g_->get_lnv();
            comm_ = g_->get_comm();

            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);

            visited_ = new GraphElem[lnv];
            pred_    = new GraphElem[lnv];
            oldq_    = new GraphElem[lnv];
            newq_    = new GraphElem[lnv];
            rbuf_    = new GraphElem[bufsize_*2];
            sbuf_    = new GraphElem[size_*bufsize_*2];
            sctr_    = new GraphElem[size_];
            sreq_    = new MPI_Request[size_];
            sact_    = new GraphElem[size_];

            std::fill(sreq_, sreq_ + size_, MPI_REQUEST_NULL);
            std::fill(sctr_, sctr_ + size_, 0);
            std::fill(sact_, sact_ + size_, 0);
            std::fill(oldq_, oldq_ + lnv, -1);
            std::fill(newq_, newq_ + lnv, -1);
            std::fill(pred_, pred_ + lnv, -1);
            std::fill(visited_, visited_ + lnv, 0);
        }

        ~BFS() 
        {
            delete []visited_;
            delete []pred_;
            delete []oldq_;
            delete []newq_;
            delete []rbuf_;
            delete []sbuf_;
            delete []sctr_;
            delete []sreq_;
            delete []sact_;
        }

        void set_visited(GraphElem v) { visited_[g_->global_to_local(v)] = 1; }
        GraphElem test_visited(GraphElem v) const { return visited_[g_->global_to_local(v)]; } 

        void process_msgs()
        {
            /* Check all MPI requests and handle any that have completed. */
            /* Test for incoming vertices to put onto the queue. */
            while (ract_) 
            {
                int flag;
                MPI_Status st;
                MPI_Test(&rreq_, &flag, &st);
                if (flag) 
                {
                    ract_ = 0;
                    int count;
                    MPI_Get_count(&st, MPI_GRAPH_TYPE, &count);

                    /* count == 0 is a signal from a rank that it is done sending to me
                     * (using MPI's non-overtaking rules to keep that signal after all
                     * "real" messages. */
                    if (count == 0) 
                    {
                        ++nranks_done_;
                    } 
                    else 
                    {
                        for (GraphElem j = 0; j < count; j += 2) 
                        {
                            GraphElem tgt = rbuf_[j];
                            GraphElem src = rbuf_[j + 1];

                            /* Process one incoming edge. */
                            assert (g_->get_owner(tgt) == rank_);
                            if (!test_visited(tgt)) 
                            {
                                set_visited(tgt);
                                pred_[g_->global_to_local(tgt)] = src;
                                newq_[newq_count_++] = tgt;
                                edge_visit_count_++;
                            }
                        }
                    }

                    /* Restart the receive if more messages will be coming. */
                    if (nranks_done_ < size_) 
                    {
                        MPI_Irecv(rbuf_, bufsize_ * 2, MPI_GRAPH_TYPE, MPI_ANY_SOURCE, 0, comm_, &rreq_);
                        ract_ = 1;
                    }
                } 
                else 
                    break;
            }

            /* Mark any sends that completed as inactive so their buffers can be reused. */
            for (int c = 0; c < size_; ++c) 
            {
                if (sact_[c]) 
                {
                    int flag;
                    MPI_Test(&sreq_[c], &flag, MPI_STATUS_IGNORE);
                    if (flag) 
                        sact_[c] = 0;
                }
            }
        }

        void nbsend(GraphElem owner)
        {
            MPI_Isend(&sbuf_[owner * bufsize_ * 2], bufsize_ * 2, MPI_GRAPH_TYPE, 
                    owner, 0, comm_, &sreq_[owner]);

            sact_[owner] = 1;
            sctr_[owner] = 0;
        }
        
        void nbsend_count(GraphElem owner)
        {
            MPI_Isend(&sbuf_[owner * bufsize_ * 2], sctr_[owner], MPI_GRAPH_TYPE, 
                    owner, 0, comm_, &sreq_[owner]);

            sact_[owner] = 1;
            sctr_[owner] = 0;
        }

        void nbsend_zero(GraphElem owner)
        {
            /* Base address is meaningless for 0-sends. */
            MPI_Isend(&sbuf_[0], 0, MPI_GRAPH_TYPE, owner, 0, comm_, &sreq_[owner]);

            sact_[owner] = 1;
        }

        // reimplementation of graph500 BFS
        void run_bfs(GraphElem root) 
        {
#if defined(USE_ALLREDUCE_FOR_EXIT)
            GraphElem global_newq_count;
#else      
            bool done = false, nbar_active = false; 
            MPI_Request nbar_req = MPI_REQUEST_NULL;
#endif
            /* Mark the root and put it into the queue. */
            if (g_->get_owner(root) == rank_) 
            {
                set_visited(root);
                pred_[g_->global_to_local(root)] = root;
                oldq_[oldq_count_++] = root;
                edge_visit_count_++;
            }

            process_msgs();

#if defined(USE_ALLREDUCE_FOR_EXIT)
                while(1)
#else
                while(!done)
#endif
                {
                    memset(sctr_, 0, size_ * sizeof(GraphElem));
                    nranks_done_ = 0;

                    /* Start the initial receive. */
                    if (nranks_done_ < size_) 
                    {
                        MPI_Irecv(rbuf_, bufsize_ * 2, MPI_GRAPH_TYPE, MPI_ANY_SOURCE, 0, comm_, &rreq_);
                        ract_ = 1;
                    }

                    /* Step through the current level's queue. */
                    for (GraphElem i = 0; i < oldq_count_; ++i) 
                    {
                        process_msgs();

                        assert (g_->get_owner(oldq_[i]) == rank_);
                        assert (pred_[g_->global_to_local(oldq_[i])] >= 0 && pred_[g_->global_to_local(oldq_[i])] < g_->get_nv());
                        GraphElem src = oldq_[i];

                        /* Iterate through its incident edges. */
                        GraphElem e0, e1;
                        g_->edge_range(g_->global_to_local(src), e0, e1);

                        if ((e0 + 1) == e1)
                          continue;

                        for (GraphElem m = e0; m < e1; m++)
                        {
                            Edge const& edge = g_->get_edge(m);
                            const int owner = g_->get_owner(edge.tail_);

                            if (owner == rank_)
                            {
                                if (!test_visited(edge.tail_)) 
                                {
                                    set_visited(edge.tail_);
                                    pred_[g_->global_to_local(edge.tail_)] = src;
                                    newq_[newq_count_++] = edge.tail_;
                                    edge_visit_count_++;
                                }
                            }
                            else
                            {
                                /* Wait for buffer to be available */
                                while (sact_[owner]) 
                                    process_msgs();

                                GraphElem c = sctr_[owner];
                                sbuf_[owner * bufsize_ * 2 + c]     = edge.tail_;
                                sbuf_[owner * bufsize_ * 2 + c + 1] = src;
                                sctr_[owner] += 2;

                                if (sctr_[owner] == (bufsize_ * 2))
                                    nbsend(owner);
                            }
                        }
                    }

                    /* Flush any coalescing buffers that still have messages. */
                    for (int p = 0; p < size_; p++)
                    {
                        if (sctr_[p] != 0) 
                        {
                            while (sact_[p]) 
                                process_msgs();

                            nbsend_count(p);
                        }

                        /* Wait until all sends to this destination are done. */
                        while (sact_[p]) 
                            process_msgs();

                        /* Tell the destination that we are done sending to them. */
                        /* Signal no more sends */
                        nbsend_zero(p);

                        while (sact_[p]) 
                            process_msgs();
                    }

                    /* Wait until everyone else is done (and thus couldn't send us any more
                     * messages). */
                    while (nranks_done_ < size_) 
                        process_msgs();

                    /* Test globally if all queues are empty. */
#if defined(USE_ALLREDUCE_FOR_EXIT)
                    MPI_Allreduce(&newq_count_, &global_newq_count, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);

                    /* Quit if they all are empty. */
                    if (global_newq_count == 0) 
                        break;
#else
                    if (nbar_active)
                    {
                        int test_nbar = -1;
                        MPI_Test(&nbar_req, &test_nbar, MPI_STATUS_IGNORE);
                        done = !test_nbar ? false : true;
                    }
                    else
                    {
                        if (newq_count_ == 0)
                        {
                            MPI_Ibarrier(comm_, &nbar_req);
                            nbar_active = true;
                        }
                    }
#endif

                    /* Swap old and new queues; clear new queue for next level. */
                    GraphElem *tmp = oldq_; 
                    oldq_ = newq_; 
                    newq_ = tmp;

                    oldq_count_ = newq_count_;
                    newq_count_ = 0;
                }
        }

        void run_test(GraphElem nbfs_roots=DEF_BFS_ROOTS)
        {
            std::seed_seq seed{seed_};
            std::mt19937 gen{seed};
            std::uniform_int_distribution<GraphElem> uid(0, g_->get_nv()-1); 

            std::vector<GraphElem> bfs_roots(nbfs_roots);
            std::vector<double> bfs_times(nbfs_roots);

            double t1 = MPI_Wtime();

            // calculate bfs roots
            GraphElem counter = 0, bfs_root_idx = 0, nv = g_->get_nv();
            
            for (bfs_root_idx = 0; bfs_root_idx < nbfs_roots; ++bfs_root_idx) 
            {
              GraphElem root;

              while (1) 
              {
                root = uid(gen);

                if (counter > nv) 
                  break;
                int is_duplicate = 0;

                for (GraphElem i = 0; i < bfs_root_idx; ++i) 
                {
                  if (root == bfs_roots[i]) 
                  {
                    is_duplicate = 1;
                    break;
                  }
                }

                if (is_duplicate) 
                  continue;

                int root_ok = 0;
                if (g_->get_owner(root) == rank_)
                {
                  GraphElem e0, e1;
                  g_->edge_range(g_->global_to_local(root), e0, e1);
                  if ((e0 + 1) != e1)
                    root_ok = 1;
                }
                
                MPI_Barrier(comm_);
                MPI_Allreduce(MPI_IN_PLACE, &root_ok, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
                
                if (root_ok) 
                  break;
              }

              bfs_roots[bfs_root_idx] = root;
            }

            nbfs_roots = bfs_root_idx;

            double t2 = MPI_Wtime() - t1;
            double root_t = 0.0;
            MPI_Reduce(&t2, &root_t, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
            if (rank_ == 0)
              fprintf(stderr, "Average time(s) taken to calculate %d BFS root vertices: %f\n", nbfs_roots, root_t);

            int test_ctr = 0;

            for (GraphElem const& r : bfs_roots)
            {
                if (rank_ == 0) 
                    fprintf(stderr, "Running BFS %d\n", test_ctr);
            
                /* Set all vertices to "not visited." */
                std::fill(pred_, pred_ + g_->get_lnv(), 0);

                /* Do the actual BFS. */
                double bfs_start = MPI_Wtime(), g_bfs_time = 0.0;
                run_bfs(r);
                double bfs_stop = MPI_Wtime();
                bfs_times[test_ctr] = bfs_stop - bfs_start;
                MPI_Allreduce(&bfs_times[test_ctr], &g_bfs_time, 1, MPI_DOUBLE, MPI_SUM, comm_);

                if (rank_ == 0)
                {
                    double avgt = ((double)(g_bfs_time / (double)size_));
                    bfs_times[test_ctr] = avgt;
                    fprintf(stderr, "Average time(s), TEPS for BFS %d, %d: %f\n", test_ctr, r, avgt);
                }
                    
                test_ctr++;
            }
            
            GraphElem ecg = 0; /* Total edge visitations. */
            MPI_Reduce(&edge_visit_count_, &ecg, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
                
            if (rank_ == 0)
            {
                double avgt_nroots = std::accumulate(bfs_times.begin(), bfs_times.end(), 0.0) / bfs_roots.size();
                fprintf(stderr, "-------------------------------------\n");
                fprintf(stderr, "Average time(s), TEPS across %d roots: %f, %g\n", bfs_roots.size(), avgt_nroots, (double)((double)ecg / avgt_nroots));
                fprintf(stderr, "-------------------------------------\n");
            }
        }

    private:
        Graph* g_;

        int rank_, size_;
        MPI_Comm comm_;

        GraphElem bufsize_, newq_count_, oldq_count_, nranks_done_, ract_, seed_, edge_visit_count_;
        GraphElem *sbuf_, *rbuf_, *pred_, *visited_, *oldq_, *newq_, *sctr_, *sact_;
        MPI_Request *sreq_, rreq_;
};



class TriangulateBFS
{
  public:

    TriangulateAggrBufferedInrecv(Graph* g): 
      g_(g), sbuf_ctr_(nullptr), sbuf_(nullptr), rbuf_(nullptr), 
      pdegree_(0), sreq_(nullptr), erange_(nullptr), vcount_(nullptr), ntriangles_(0), 
      nghosts_(0), out_nghosts_(0), in_nghosts_(0), pindex_(0), targets_(0), sources_(0)
  {
    comm_ = g_->get_comm();
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

    const GraphElem lnv = g_->get_lnv();
    const GraphElem nv = g_->get_nv();

    vcount_ = new GraphElem[lnv]();
    erange_ = new GraphElem[nv*2]();
    GraphElem *send_count  = new GraphElem[size_]();
    recv_count_  = new GraphElem[size_]();

    double t0 = MPI_Wtime();

    // store edge ranges
    GraphElem base = g_->get_base(rank_);
    for (GraphElem i = 0; i < lnv; i++)
    {
      GraphElem e0, e1;
      g_->edge_range(i, e0, e1);

      if ((e0 + 1) == e1)
        continue;

      Edge const& edge_s = g_->get_edge(e0);
      Edge const& edge_t = g_->get_edge(e1-1);

      erange_[(i + base)*2] = edge_s.tail_;
      erange_[(i + base)*2+1] = edge_t.tail_;
    }
    
    MPI_Barrier(comm_);
    
    MPI_Allreduce(MPI_IN_PLACE, erange_, nv*2, MPI_GRAPH_TYPE, 
        MPI_SUM, comm_);

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
            Edge const& edge_n = g_->get_edge(n);
             
            if (!edge_above_min(edge_m.tail_, edge_n.tail_))
              continue;

            if (!edge_within_max(edge_m.tail_, edge_n.tail_))
              break;

            send_count[owner] += 1;
            vcount_[i] += 1;
          }
        }
      }
    }

    MPI_Barrier(comm_);

    double t1 = MPI_Wtime();
    double p_tot = t1 - t0, t_tot = 0.0;

    MPI_Reduce(&p_tot, &t_tot, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

    if (rank_ == 0) 
    {   
      std::cout << "Average time for local counting during instantiation (secs.): " 
        << ((double)(t_tot / (double)size_)) << std::endl;
    }

    t0 = MPI_Wtime();

    // outgoing/incoming data and buffer size
    MPI_Alltoall(send_count, 1, MPI_GRAPH_TYPE, recv_count_, 1, MPI_GRAPH_TYPE, comm_);
    
    for (GraphElem p = 0; p < size_; p++)
    {
      out_nghosts_ += send_count[p];
      in_nghosts_ += recv_count_[p];

      if (send_count[p] > 0)
        targets_.push_back(p);

      if (recv_count_[p] > 0)
        sources_.push_back(p);
    }
      
    pdegree_ = targets_.size();
    rdegree_ = sources_.size();

    for (int i = 0; i < pdegree_; i++)
      pindex_.insert({targets_[i], i});

    for (int i = 0; i < rdegree_; i++)
      rindex_.insert({sources_[i], i});

    nghosts_ = out_nghosts_ + in_nghosts_;

    bufsize_ = ((nghosts_*2) < bufsize) ? (nghosts_*2) : bufsize;

    MPI_Allreduce(MPI_IN_PLACE, &bufsize_, 1, MPI_GRAPH_TYPE, MPI_MAX, comm_);

    if (rank_ == 0)
      std::cout << "Adjusted Per-PE buffer count: " << bufsize_ << std::endl;
 
    sbuf_     = new GraphElem[pdegree_*bufsize_];
    sbuf_ctr_ = new GraphElem[pdegree_]();
    prev_k_   = new GraphElem[pdegree_];
    prev_m_   = new GraphElem[pdegree_];
    sreq_     = new MPI_Request[pdegree_];
    stat_     = new char[pdegree_];
    
    rbuf_     = new GraphElem[rdegree_*bufsize_];
    rreq_     = new MPI_Request[rdegree_];
    ract_     = new char[rdegree_];

    std::fill(sreq_, sreq_ + pdegree_, MPI_REQUEST_NULL);
    std::fill(prev_k_, prev_k_ + pdegree_, -1);
    std::fill(prev_m_, prev_m_ + pdegree_, -1);
    std::fill(stat_, stat_ + pdegree_, '0');
    
    std::fill(rreq_, rreq_ + rdegree_, MPI_REQUEST_NULL);
    std::fill(ract_, ract_ + rdegree_, '0');

    MPI_Barrier(comm_);

#if defined(DEBUG_PRINTF)
    if (rank_ == 0)
    {
      std::cout << "Edge range per vertex (#ID: <range>): " << std::endl;
      for (int i = 0, j = 0; i < nv*2; i+=2, j++)
        std::cout << j << ": " << erange_[i] << ", " << erange_[i+1] << std::endl;
    }
#endif
    
    delete []send_count;
  }

    ~TriangulateAggrBufferedInrecv() {}

    void clear()
    {
      delete []sbuf_;
      delete []rbuf_;
      delete []recv_count_;
      delete []sbuf_ctr_;
      delete []sreq_;
      delete []rreq_;
      delete []prev_k_;
      delete []prev_m_;
      delete []stat_;
      delete []vcount_;
      delete []erange_;

#if defined(RECV_MAP)
      rmap_.clear();
#endif

      pindex_.clear();
      targets_.clear();
      rindex_.clear();
      sources_.clear();
    }

    void nbsend(GraphElem owner)
    {
      if (stat_[pindex_[owner]] == '0' && sbuf_ctr_[pindex_[owner]] > 0)
      {
        MPI_Isend(&sbuf_[pindex_[owner]*bufsize_], sbuf_ctr_[pindex_[owner]], 
            MPI_GRAPH_TYPE, owner, TAG_DATA, comm_, &sreq_[pindex_[owner]]);
        stat_[pindex_[owner]] = '1'; 
      }
    }

    void nbsend()
    {
#if defined(USE_OPENMP) && !defined(RECV_MAP)
#pragma omp parallel for default(shared)
#endif
      for (GraphElem p = 0; p < pdegree_; p++)
        nbsend(targets_[p]);
    }

    inline void nbrecv()
    {
#if defined(USE_OPENMP) && !defined(RECV_MAP) 
#pragma omp parallel for default(shared)
#endif
      for (int p = 0; p < rdegree_; p++)
      {
        if (ract_[p] == '0' && recv_count_[sources_[p]] > 0)
        {
          MPI_Irecv(&rbuf_[p*bufsize_], bufsize_, 
              MPI_GRAPH_TYPE, sources_[p], TAG_DATA, comm_, &rreq_[p]);
          ract_[p] = '1';
        }
      }
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
          const GraphElem disp = pidx*bufsize_;

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

                nbsend(owner);

                continue;
              }

              sbuf_[disp+sbuf_ctr_[pidx]] = edge.edge_->tail_;
              sbuf_ctr_[pidx] += 1;

              for (GraphElem n = ((prev_k_[pidx] == -1) ? (m + 1) : prev_k_[pidx]); n < e1; n++)
              {  
                Edge const& edge_n = g_->get_edge(n);                                
                                  
                if (!edge_above_min(edge.edge_->tail_, edge_n.tail_))
                  continue;

                if (!edge_within_max(edge.edge_->tail_, edge_n.tail_))
                  break;

                if (sbuf_ctr_[pidx] == (bufsize_-1))
                {
                  prev_m_[pidx] = m;
                  prev_k_[pidx] = n;

                  sbuf_[disp+sbuf_ctr_[pidx]] = -1; // demarcate vertex boundary
                  sbuf_ctr_[pidx] += 1;

                  nbsend(owner);

                  break;
                }
                              
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

                  nbsend(owner);
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

    inline bool check_edgelist(GraphElem (&tup)[2])
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
     
#if defined(USE_OPENMP_NESTED)
    inline bool check_edgelist_omp(GraphElem (&tup)[2])
    {
      GraphElem e0, e1;
      bool found = false;
      
      const GraphElem lv = g_->global_to_local(tup[0]);
      g_->edge_range(lv, e0, e1);

#pragma omp parallel for schedule(dynamic) firstprivate(tup) shared(g_, e0, e1) reduction(||:found) if (e1 - e0 >= 50)
      for (GraphElem e = e0; e < e1; e++)
      {
        Edge const& edge = g_->get_edge(e);
        
        if (tup[1] == edge.tail_)
          found = found || true;
      }

      return found;
    }
#endif

    inline bool edge_between_range(GraphElem x, GraphElem y) const
    {
      if ((y >= erange_[x*2]) && (y <= erange_[x*2+1]))
        return true;
      return false;
    }
         
    inline bool edge_above_min(GraphElem x, GraphElem y) const
    {
      if (y >= erange_[x*2])
        return true;
      return false;
    }

    inline bool edge_within_max(GraphElem x, GraphElem y) const
    {
      if (y <= erange_[x*2+1])
        return true;
      return false;
    }
   
#if defined(RECV_MAP)
    bool find_pair(GraphElem (&pair)[2])
    {
      if (rmap_.count(pair[0]))
      {
        typedef std::multimap<const GraphElem, GraphElem>::const_iterator mmap_t;
        std::pair<mmap_t, mmap_t> range = rmap_.equal_range(pair[0]);
        
        for (mmap_t it = range.first; it != range.second; ++it)
          if (it->second == pair[1])
            return true;
      }
      return false;
    }

    void check_rmap(GraphElem (&pair)[2])
    {
      if (find_pair(pair)) 
        ntriangles_ += 1;
      else
      {
        if (check_edgelist(pair))
        {
          ntriangles_ += 1;
          rmap_.insert(std::pair<const GraphElem, GraphElem>{pair[0], pair[1]});
        }
      }
    }
#endif

    inline void process_recvs()
    {
      if (in_nghosts_ == 0)
        return;

#if defined(USE_OPENMP) && !defined(RECV_MAP)
#pragma omp parallel for default(shared) reduction(+:ntriangles_) reduction(-:in_nghosts_)
#endif
      for (int p = 0; p < rdegree_; p++)
      {
        int over = -1;
        MPI_Status status;

        MPI_Test(&rreq_[p], &over, &status);

        if (over)
        { 
          GraphElem tup[2] = {-1,-1}, prev = 0;
          int count = 0;

          ract_[p] = '0';

          const int source = status.MPI_SOURCE;

          MPI_Get_count(&status, MPI_GRAPH_TYPE, &count);

          if (source != MPI_ANY_SOURCE)
          {
            for (GraphElem k = 0; k < count;)
            {
              if (rbuf_[p*bufsize_+k] == -1)
              {
                k += 1;
                prev = k;
                continue;
              }

              tup[0] = rbuf_[p*bufsize_+k];
              GraphElem curr_count = 0;

              for (GraphElem m = k + 1; m < count; m++)
              {
                if (rbuf_[p*bufsize_+m] == -1)
                {
                  curr_count = m + 1;
                  break;
                }

                tup[1] = rbuf_[p*bufsize_+m];

#if defined(RECV_MAP)
                check_rmap(tup);
#else
#if defined(USE_OPENMP_NESTED)
                if (check_edgelist_omp(tup))
                  ntriangles_ += 1;
#else
                if (check_edgelist(tup))
                  ntriangles_ += 1;
#endif
#endif
                in_nghosts_ -= 1;
                recv_count_[source] -= 1;
              }

              k += (curr_count - prev);
              prev = k;
            }

            if (recv_count_[source] > 0)
            {
              MPI_Irecv(&rbuf_[p*bufsize_], bufsize_, 
                  MPI_GRAPH_TYPE, source, TAG_DATA, comm_, &rreq_[p]);
              ract_[p] = '1';
            }
          }
        }
      }
    }

    inline GraphElem count()
    {
#if defined(USE_ALLREDUCE_FOR_EXIT)
      GraphElem count;
#else      
      bool done = false, nbar_active = false; 
      MPI_Request nbar_req = MPI_REQUEST_NULL;
#endif

      int* inds = new int[pdegree_]();
      int over = -1;
      
      nbrecv();

#if defined(USE_ALLREDUCE_FOR_EXIT)
      while(1)
#else
      while(!done)
#endif
      {
        process_recvs();

        if (out_nghosts_ == 0)
          nbsend();
        else
          lookup_edges();
        
        MPI_Testsome(pdegree_, sreq_, &over, inds, MPI_STATUSES_IGNORE);

        if (over > 0)
        {
          for (int i = 0; i < over; i++)
          {
            sbuf_ctr_[inds[i]] = 0;
            stat_[inds[i]] = '0';
          }
        }

#if defined(USE_ALLREDUCE_FOR_EXIT)
        count = in_nghosts_;
        MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
        if (count == 0)
          break;
#else       
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
#endif
#if defined(DEBUG_PRINTF)
        std::cout << "in/out: " << in_nghosts_ << ", " << out_nghosts_ << std::endl;
#endif
      }

      GraphElem ttc = 0, ltc = ntriangles_;
      MPI_Barrier(comm_);
      MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
      
      delete []inds;

      return (ttc/3);
    }

  private:
    Graph* g_;

    GraphElem ntriangles_, bufsize_, nghosts_, out_nghosts_, in_nghosts_, pdegree_, rdegree_;
    GraphElem *sbuf_, *rbuf_, *recv_count_, *prev_k_, *prev_m_, *sbuf_ctr_, *vcount_, *erange_;
    MPI_Request *sreq_, *rreq_;
    char *stat_, *ract_;
    
    std::vector<int> targets_, sources_;
    int rank_, size_;
    std::unordered_map<int, int> pindex_, rindex_; 
#if defined(RECV_MAP)
    std::multimap<const GraphElem, GraphElem> rmap_;
#endif
    MPI_Comm comm_;
};
#endif
