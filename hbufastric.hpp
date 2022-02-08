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
#ifndef BUF_HFASTRIC_HPP
#define BUF_HFASTRIC_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>

#ifndef TAG_DATA
#define TAG_DATA 100
#endif


class TriangulateAggrBufferedHeuristics
{
  public:

    TriangulateAggrBufferedHeuristics(Graph* g, const GraphElem bufsize): 
      g_(g), sbuf_ctr_(nullptr), sbuf_(nullptr), rbuf_(nullptr), pdegree_(0), 
      gcomm_(MPI_COMM_NULL), sreq_(nullptr), rinfo_(nullptr), srinfo_(nullptr), erange_(nullptr), 
      vcount_(nullptr), ntriangles_(0), nghosts_(0), out_nghosts_(0), in_nghosts_(0), pindex_(0), 
      prev_m_(nullptr), prev_k_(nullptr), stat_(nullptr), bufsize_(bufsize)
  {
    comm_ = g_->get_comm();
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

    GraphElem *send_count = new GraphElem[size_]();
    GraphElem *recv_count = new GraphElem[size_]();

    const GraphElem lnv = g_->get_lnv();
    const GraphElem nv = g_->get_nv();

    vcount_ = new GraphElem[lnv]();
    erange_ = new GraphElem[nv*2]();

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
        EdgeStat& edge_m = g_->get_edge_stat(m);
        const int owner = g_->get_owner(edge_m.edge_->tail_);
        tup[0] = edge_m.edge_->tail_;

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
          int check = 0;
          for (GraphElem n = m + 1; n < e1; n++)
          {
            Edge const& edge_n = g_->get_edge(n);
            
            // check validity of edge range
            if (!edge_within_range(edge_m.edge_->tail_, edge_n.tail_) || !edge_within_range(edge_n.tail_, edge_m.edge_->tail_))
            {
              check += 1;
              break;
            }

            check += 1;
            send_count[owner] += 1;
            vcount_[i] += 1;
          }

          if (check == 1)
            edge_m.active_ = false;

          if (check > 1)
          {
            if (std::find(targets_.begin(), targets_.end(), owner) 
                == targets_.end())
              targets_.push_back(owner);
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

    MPI_Alltoall(send_count, 1, MPI_GRAPH_TYPE, recv_count, 1, MPI_GRAPH_TYPE, comm_);

    for (GraphElem p = 0; p < size_; p++)
    {
      out_nghosts_ += send_count[p];
      in_nghosts_ += recv_count[p];
    }

    nghosts_ = out_nghosts_ + in_nghosts_;

    free(send_count);
    free(recv_count);

    MPI_Dist_graph_create_adjacent(comm_, targets_.size(), targets_.data(), 
        MPI_UNWEIGHTED, targets_.size(), targets_.data(), MPI_UNWEIGHTED, 
        MPI_INFO_NULL, 0 /*reorder ranks?*/, &gcomm_);

    // double-checking indegree/outdegree
    int weighted, indegree, outdegree;
    MPI_Dist_graph_neighbors_count(gcomm_, &indegree, &outdegree, &weighted);
    assert(indegree == targets_.size());
    assert(outdegree == targets_.size());
    assert(indegree == outdegree);

    pdegree_ = indegree; // for undirected graph, indegree == outdegree

    for (int i = 0; i < pdegree_; i++)
      pindex_.insert({targets_[i], (GraphElem)i});

    sbuf_     = new GraphElem[pdegree_*bufsize_];
    sbuf_ctr_ = new GraphElem[pdegree_]();
    rinfo_    = new GraphElem[pdegree_]();
    srinfo_   = new GraphElem[pdegree_]();
    prev_k_   = new GraphElem[pdegree_];
    prev_m_   = new GraphElem[pdegree_];
    stat_     = new char[pdegree_];
    sreq_     = new MPI_Request[pdegree_];
    rbuf_     = new GraphElem[bufsize_]();

    std::fill(sreq_, sreq_ + pdegree_, MPI_REQUEST_NULL);
    std::fill(prev_k_, prev_k_ + pdegree_, -1);
    std::fill(prev_m_, prev_m_ + pdegree_, -1);
    std::fill(stat_, stat_ + pdegree_, '0');

    MPI_Barrier(comm_);

#if defined(DEBUG_PRINTF)
    if (rank_ == 0)
    {
      std::cout << "Edge range per vertex (#ID: <range>): " << std::endl;
      for (int i = 0, j = 0; i < nv*2; i+=2, j++)
        std::cout << j << ": " << erange_[i] << ", " << erange_[i+1] << std::endl;
    }
#endif
  }

    ~TriangulateAggrBufferedHeuristics() {}

    void clear()
    {
      MPI_Comm_free(&gcomm_);

      delete []sbuf_;
      delete []rbuf_;
      delete []sbuf_ctr_;
      delete []srinfo_;
      delete []rinfo_;
      delete []sreq_;
      delete []prev_k_;
      delete []prev_m_;
      delete []stat_;
      delete []vcount_;
      delete []erange_;

      pindex_.clear();
      targets_.clear();
    }

    // TODO
    inline void check()
    {
    }

    void nbsend(GraphElem owner)
    {
      if (sbuf_ctr_[pindex_[owner]] > 0)
      {
        MPI_Isend(&sbuf_[pindex_[owner]*bufsize_], sbuf_ctr_[pindex_[owner]], 
            MPI_GRAPH_TYPE, owner, TAG_DATA, comm_, &sreq_[pindex_[owner]]);
      }
    }

    // create a stat to denote end
    void nbsend()
    {
      for (GraphElem p = 0; p < size_; p++)
      {
        if (p != rank_)
          nbsend(p);
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

          if ((owner != rank_) && edge.active_)
          {   
            if (stat_[pindex_[owner]] == '1') 
              continue;

            if (m >= prev_m_[pindex_[owner]])
            {
              const GraphElem disp = pindex_[owner]*bufsize_;

              if (sbuf_ctr_[pindex_[owner]] == (bufsize_-1))
              {
                prev_m_[pindex_[owner]] = m;
                prev_k_[pindex_[owner]] = -1;
                stat_[pindex_[owner]] = '1'; // messages in-flight

                nbsend(owner);

                continue;
              }

              sbuf_[disp+sbuf_ctr_[pindex_[owner]]] = edge.edge_->tail_;
              sbuf_ctr_[pindex_[owner]] += 1;
              int inrange = 0;

              for (GraphElem n = ((prev_k_[pindex_[owner]] == -1) ? (m + 1) : prev_k_[pindex_[owner]]); n < e1; n++)
              {               
                Edge const& edge_n = g_->get_edge(n);                                
                if (!edge_within_range(edge.edge_->tail_, edge_n.tail_) || !edge_within_range(edge_n.tail_, edge.edge_->tail_))
                {
                  inrange += 1;
                  break;
                }

                if (sbuf_ctr_[pindex_[owner]] == (bufsize_-1))
                {
                  prev_m_[pindex_[owner]] = m;
                  prev_k_[pindex_[owner]] = n;

                  sbuf_[disp+sbuf_ctr_[pindex_[owner]]] = -1; // demarcate vertex boundary
                  sbuf_ctr_[pindex_[owner]] += 1;
                  stat_[pindex_[owner]] = '1'; 

                  nbsend(owner);

                  break;
                }
                               
                inrange += 1;
                sbuf_[disp+sbuf_ctr_[pindex_[owner]]] = edge_n.tail_;
                sbuf_ctr_[pindex_[owner]] += 1;
                out_nghosts_ -= 1;
                vcount_[i] -= 1;
              }

              if (stat_[pindex_[owner]] == '0') 
              {               
                prev_m_[pindex_[owner]] = m;
                prev_k_[pindex_[owner]] = -1;

                edge.active_ = false;
  
                if (inrange == 1)
                  sbuf_ctr_[pindex_[owner]] -= 1;

                if (sbuf_ctr_[pindex_[owner]] == (bufsize_-1))
                {
                  sbuf_[disp+sbuf_ctr_[pindex_[owner]]] = -1; 
                  sbuf_ctr_[pindex_[owner]] += 1;
                  stat_[pindex_[owner]] = '1';

                  nbsend(owner);
                }
                else
                {
                  sbuf_[disp+sbuf_ctr_[pindex_[owner]]] = -1; 
                  sbuf_ctr_[pindex_[owner]] += 1;
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

    inline bool edge_within_range(GraphElem x, GraphElem y)
    {
      if ((y >= erange_[x*2]) && (y <= erange_[x*2+1]))
        return true;
      return false;
    }

    inline void process_messages()
    {
      MPI_Status status;
      int flag = -1;
      GraphElem tup[2] = {-1,-1}, source = -1, prev = 0;
      int count = 0;

      MPI_Iprobe(MPI_ANY_SOURCE, TAG_DATA, comm_, &flag, &status);

      if (flag)
      { 
        source = status.MPI_SOURCE;
        MPI_Get_count(&status, MPI_GRAPH_TYPE, &count);
        MPI_Recv(rbuf_, count, MPI_GRAPH_TYPE, source, 
            TAG_DATA, comm_, MPI_STATUS_IGNORE);            
      }
      else
        return;

      for (GraphElem k = 0; k < count;)
      {
        if (rbuf_[k] == -1)
          continue;

        tup[0] = rbuf_[k];
        GraphElem curr_count = 0;

        for (GraphElem m = k + 1; m < count; m++)
        {
          if (rbuf_[m] == -1)
          {
            curr_count = m + 1;
            break;
          }

          tup[1] = rbuf_[m];

          if (check_edgelist(tup))
            rinfo_[pindex_[source]] += 1; // EDGE_VALID_TAG 

          in_nghosts_ -= 1;
        }

        k += (curr_count - prev);
        prev = k;
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
        if (out_nghosts_ == 0)
        {
          if (!sends_done)
          {
            nbsend();
            sends_done = true;
          }
        }
        else
          lookup_edges();

        process_messages();

        MPI_Testsome(pdegree_, sreq_, &over, inds, MPI_STATUSES_IGNORE);

        if (over != MPI_UNDEFINED)
        {
          for (int i = 0; i < over; i++)
          {
            sbuf_ctr_[inds[i]] = 0;
            stat_[inds[i]] = '0';
          }
        }
        
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

      free(inds);

      return (ttc/3);
    }

  private:
    Graph* g_;

    GraphElem ntriangles_;
    GraphElem bufsize_, nghosts_, out_nghosts_, in_nghosts_, pdegree_;
    GraphElem *sbuf_, *rbuf_, *prev_k_, *prev_m_, *sbuf_ctr_, *rinfo_, *srinfo_, *vcount_, *erange_;
    MPI_Request *sreq_;
    char *stat_;

    std::vector<int> targets_;

    int rank_, size_;
    std::unordered_map<GraphElem, GraphElem> pindex_; 
    MPI_Comm comm_, gcomm_;
};
#endif
