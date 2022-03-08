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
#ifndef DEGCOMM_TFASTRIC_HPP
#define DEGCOMM_TFASTRIC_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>

#ifndef TAG_DATA
#define TAG_DATA 100
#endif


class TriangulateDegreeBased
{
  public:

    TriangulateDegreeBased(Graph* g): 
      g_(g), sbuf_(nullptr), rbuf_(nullptr),  sreq_(nullptr), erange_(nullptr), 
      nsdispls_(nullptr), nrdispls_(nullptr), stat_(nullptr), sbuf_ctr_(nullptr),
      send_count_(nullptr), recv_count_(nullptr), pdegree_(0), ntriangles_(0), 
      out_nghosts_(0), in_nghosts_(0), targets_(0) 
  {
    comm_ = g_->get_comm();
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

    send_count_  = new GraphElem[size_]();
    recv_count_  = new GraphElem[size_]();
    GraphElem *vsend_count = new GraphElem[size_]();
    GraphElem *vrecv_count = new GraphElem[size_]();

    const GraphElem lnv = g_->get_lnv();
    const GraphElem nv = g_->get_nv();

    erange_ = new GraphElem[nv*2]();

    double t0 = MPI_Wtime();

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

    GraphElem past_v = -1;

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
          if (std::find(targets_.begin(), targets_.end(), owner) 
              == targets_.end()) 
            targets_.push_back(static_cast<GraphElem>(owner));

          if (i != past_v)
          {
            vsend_count[owner] += 1;
            past_v = i;
          }

          send_count_[owner] += 1;
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

    MPI_Alltoall(send_count_, 1, MPI_GRAPH_TYPE, recv_count_, 1, MPI_GRAPH_TYPE, comm_);
    MPI_Alltoall(vsend_count, 1, MPI_GRAPH_TYPE, vrecv_count, 1, MPI_GRAPH_TYPE, comm_);

    pdegree_ = targets_.size(); 

    for (int i = 0; i < pdegree_; i++)
      pindex_.insert({targets_[i], static_cast<GraphElem>(i)});

    stat_ = new char[pdegree_];
    sreq_ = new MPI_Request[pdegree_];
    nsdispls_ = new GraphElem[pdegree_]();
    nrdispls_ = new GraphElem[pdegree_]();
    sbuf_ctr_ = new GraphElem[pdegree_]();

    std::fill(sreq_, sreq_ + pdegree_, MPI_REQUEST_NULL);
    std::fill(stat_, stat_ + pdegree_, '0');

    GraphElem rcount = 0, scount = 0, incr = 0; 

    for (GraphElem p = 0; p < size_; p++)
    {
      if (send_count_[p] > 0)
      {
        if ((incr % 2) == 0)
        {
          out_nghosts_ += send_count_[p];
          stat_[pindex_[targets_[p]]] = '0'; // send to targets_[p]
          nsdispls_[pindex_[targets_[p]]] = scount;
          scount += (send_count_[p] + 2*vsend_count[p]);
        }
        else
        {
          in_nghosts_ += recv_count_[p];
          stat_[pindex_[targets_[p]]] = '1'; // recv from targets_[p]
          nrdispls_[pindex_[targets_[p]]] = rcount;
          rcount += (recv_count_[p] + 2*vrecv_count[p]);
        }

        incr += 1;
      }
    }

    sbuf_ = new GraphElem[scount];
    rbuf_ = new GraphElem[rcount];

#if defined(DEBUG_PRINTF)
    if (rank_ == 0)
    {
      std::cout << "Edge range per vertex (#ID: <range>): " << std::endl;
      for (int i = 0, j = 0; i < nv*2; i+=2, j++)
        std::cout << j << ": " << erange_[i] << ", " << erange_[i+1] << std::endl;
    }
#endif
      
    delete []vsend_count;
    delete []vrecv_count;
  }

    ~TriangulateDegreeBased() {}

    void clear()
    {
      delete []sbuf_;
      delete []rbuf_;
      delete []sreq_;
      delete []nsdispls_;
      delete []nrdispls_;
      delete []stat_;
      delete []erange_;
      delete []send_count_;
      delete []recv_count_;

      pindex_.clear();
      targets_.clear();
    }

    void nbsend(GraphElem owner)
    {
      if (sbuf_ctr_[pindex_[owner]] > 0)
      {
        MPI_Issend(&sbuf_[nsdispls_[pindex_[owner]]], sbuf_ctr_[pindex_[owner]], 
            MPI_GRAPH_TYPE, owner, TAG_DATA, comm_, &sreq_[pindex_[owner]]);
      }
    }

    void nbsend()
    {
      for (int const& p : targets_)
        nbsend(p);
    }

    void outgoing_edges()
    {
      const GraphElem lnv = g_->get_lnv();
      
      for (GraphElem i = 0; i < lnv; i++)
      {
        GraphElem e0, e1, c = 0, past_owner = -1;
        g_->edge_range(i, e0, e1);

        if ((e0 + 1) == e1)
          continue;

        for (GraphElem m = e0; m < e1-1; m++)
        {
          Edge const& edge_m = g_->get_edge(m);
          const int owner = g_->get_owner(edge_m.tail_);
          const GraphElem pidx = pindex_[owner];

          if (owner != rank_)
          {   
            if (stat_[pidx] == '1') // I am a `receiver` 
              continue;

            if ((past_owner > -1) && (owner != past_owner))
            {
              sbuf_[nsdispls_[pindex_[past_owner]]+sbuf_ctr_[pindex_[past_owner]]] = -1;
              sbuf_ctr_[pindex_[past_owner]] += 1;
              past_owner = owner;
            }

            if (c == 0)
            {
              sbuf_[nsdispls_[pidx]+sbuf_ctr_[pidx]] = g_->local_to_global(i);
              sbuf_ctr_[pidx] += 1;
            }

            sbuf_[nsdispls_[pidx]+sbuf_ctr_[pidx]] = edge_m.tail_;
            sbuf_ctr_[pidx] += 1;
            c += 1;
          }
        }
      }

      nbsend();
    }

    // receivers count - ensuring edges are counted once
    void incoming_edges()
    {
      MPI_Status status;
      int flag = -1;
      GraphElem tup[2] = {-1,-1}, k = 0, prev = 0;
      int count = 0, source = -1;

      MPI_Iprobe(MPI_ANY_SOURCE, TAG_DATA, comm_, &flag, &status);

      if (flag)
      { 
        source = status.MPI_SOURCE;
        MPI_Get_count(&status, MPI_GRAPH_TYPE, &count);
        MPI_Recv(&rbuf_[nrdispls_[source]], count, MPI_GRAPH_TYPE, source, 
            TAG_DATA, comm_, MPI_STATUS_IGNORE);      
      }
      else
        return;

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
          const GraphElem pidx = pindex_[owner];

          if (owner == source)
          {   
            if (stat_[pidx] == '0') // sender 
              continue;

            for (GraphElem n = m + 1; n < e1; n++)
            { 
              Edge const& edge_n = g_->get_edge(n);                                

              if (!edge_within_max(edge_m.tail_, edge_n.tail_))
                break;
              if (!edge_above_min(edge_m.tail_, edge_n.tail_) || !edge_above_min(edge_n.tail_, edge_m.tail_))
                continue;

              check_remote_edgelist(&rbuf_[nrdispls_[source]], count, edge_m.tail_, edge_n.tail_);
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

    inline void check_remote_edgelist(GraphElem *rbuf, GraphElem count, GraphElem x, GraphElem y)
    {
      GraphElem tup[2] = {-1,-1}, k = 0, prev = 0;
      
      while(1)
      {
        if (k == count)
          break;

        if (rbuf_[k] == -1)
        {
          k += 1;
          prev = k;
          continue;
        }

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

          if ((rbuf_[k] == x) && (rbuf_[m] == y))
          {
            ntriangles_ += 1;
            break;
          }
        }

        k += (curr_count - prev);
        prev = k;
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
      
      int *inds = new int[pdegree_];
      int over = -1;

      // send phase
      outgoing_edges();

      if (rank_ == 0)
        std::cout << "Initial remote edges send phase completed (" << 2*out_nghosts_ << " edges sent)." << std::endl;

      // recv phase
#if defined(USE_ALLREDUCE_FOR_EXIT)
      while(1)
#else
      while(!done)
#endif
        {
          MPI_Testsome(pdegree_, sreq_, &over, inds, MPI_STATUSES_IGNORE);

          if (over != MPI_UNDEFINED)
          {
            for (int i = 0; i < over; i++)
            {
              GraphElem idx = static_cast<GraphElem>(inds[i]);
              sbuf_ctr_[idx] = 0;
              out_nghosts_ -= send_count_[targets_[idx]];
            }
          }
          
          incoming_edges();

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
            if (out_nghosts_ == 0)
            {
              MPI_Ibarrier(comm_, &nbar_req);
              nbar_active = true;
            }
          }
#endif
        }

      GraphElem ttc = 0, ltc = ntriangles_;
      MPI_Barrier(comm_);
      MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);

      delete []inds;

      return (ttc / 3);
    }

  private:
    Graph* g_;

    GraphElem ntriangles_, out_nghosts_, in_nghosts_, pdegree_;
    GraphElem *sbuf_, *rbuf_, *erange_, *nsdispls_, *nrdispls_, *sbuf_ctr_, *send_count_, *recv_count_;
    MPI_Request *sreq_;
    char *stat_;

    std::vector<GraphElem> targets_;

    int rank_, size_;
    std::unordered_map<GraphElem, GraphElem> pindex_; 
    MPI_Comm comm_;
};
#endif
