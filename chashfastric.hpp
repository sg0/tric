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
#ifndef CHASH_TFASTRIC_HPP
#define CHASH_TFASTRIC_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>

#ifndef TAG_DATA
#define TAG_DATA 100
#endif

#ifndef BLOOMFILTER_TOL
#define BLOOMFILTER_TOL 1E-09
#endif

#include "murmurhash/MurmurHash3.h"

class Bloomfilter
{
  public:
    Bloomfilter(GraphElem n, GraphWeight p=BLOOMFILTER_TOL) 
      : n_(pow(2, std::ceil(log(n)/log(2)))), p_(p), bits_(nullptr)
    {
      m_ = std::ceil((n_ * log(p_)) / log(1 / pow(2, log(2))));
      k_ = std::round((m_ / n_) * log(2));

      hashes_.resize(k_); 

      if (k_ == 0)
        throw std::invalid_argument("Bloomfilter could not be initialized: k must be larger than 0");
    }
        
    Bloomfilter(GraphElem n, GraphElem k, GraphWeight p) 
      : n_(pow(2, std::ceil(log(n)/log(2)))), k_(k), p_(p), bits_(nullptr)
    {
      m_ = std::ceil((n_ * log(p_)) / log(1 / pow(2, log(2))));

      if (k_%2 != 0)
        k_ += 1;

      hashes_.resize(k_); 

      if (k_ == 0)
        throw std::invalid_argument("Bloomfilter could not be initialized: k must be larger than 0");
    }


    void insert(GraphElem const& i, GraphElem const& j)
    {
      hash(i, j);
      for (GraphElem k = 0; k < k_; k++)
        bits_[hashes_[k]] = '1';
    }

    void print() const
    {
      std::cout << "-------------Bloom filter statistics-------------" << std::endl;
      std::cout << "Number of Items (n): " << n_ << std::endl;
      std::cout << "Probability of False Positives (p): " << p_ << std::endl;
      std::cout << "Number of bits in filter (m): " << m_ << std::endl;
      std::cout << "Number of hash functions (k): " << k_ << std::endl;
      std::cout << "-------------------------------------------------" << std::endl;
    }

    void clear() { hashes_.clear(); }

    bool contains(GraphElem i, GraphElem j) 
    {
      hash(i, j);
      for (GraphElem k = 0; k < k_; k++)
      {
        if (bits_[hashes_[k]] == '0') 
          return false;
      }
      return true;
    }

    // objects must eventually call `set for
    // avoiding undefined behavior 
    void set(char *ptr) { bits_ = new (ptr) char(m_); }

    GraphElem nbits() const { return m_; }
    
    // "nucular" options, use iff 
    // you know what you're doing
    void copy_from(char* dest)
    { std::memcpy(dest, bits_, m_); }
      
    void copy_to(char* source)
    { std::memcpy(bits_, source, m_); }
     
    void copy_from(char* dest, ptrdiff_t offset)
    { std::memcpy(dest, &bits_[offset], m_); }
      
    void copy_to(char* source, ptrdiff_t offset)
    { std::memcpy(&bits_[offset], source, m_); }   
    
  private:
    GraphElem n_, m_, k_;
    GraphWeight p_;

    void hash( uint64_t lhs, uint64_t rhs ) 
    {
      uint64_t key[2] = {lhs, rhs};
      for (uint64_t n = 0; n < k_; n+=2)
      {
        MurmurHash3_x64_128 ( &key, 2*sizeof(uint64_t), 0, &hashes_[n] );
        hashes_[n] = hashes_[n] % m_; 
        hashes_[n+1] = hashes_[n+1] % m_;
      }
    }
    
    char *bits_;
    std::vector<uint64_t> hashes_;
};

class TriangulateHashRemote
{
  public:

    TriangulateHashRemote(Graph* g, const GraphElem combufsize=-1): 
      g_(g), pdegree_(0), erange_(nullptr), ntriangles_(0), pindex_(0), 
      sebf_(nullptr), rebf_(nullptr), sbuf_(nullptr), rbuf_(nullptr), 
      targets_(0), combufsize_(combufsize)
  {
    comm_ = g_->get_comm();
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

    const GraphElem lnv = g_->get_lnv();
    const GraphElem nv = g_->get_nv();
    
    erange_ = new GraphElem[nv*2]();
    std::vector<int> vtargets; 
    std::vector<std::vector<int>> vcount(lnv);

    // store edge ranges
    GraphElem base = g_->get_base(rank_);
    for (GraphElem i = 0; i < lnv; i++)
    {
      GraphElem e0, e1;
      g_->edge_range(i, e0, e1);

      if ((e0 + 1) == e1)
        continue;
      
      for (GraphElem m = e0; m < e1; m++)
      {
        Edge const& edge_m = g_->get_edge(m);
        const int owner = g_->get_owner(edge_m.tail_);
        if (owner != rank_)
        {
          if (std::find(vtargets.begin(), vtargets.end(), owner) 
              == vtargets.end())
            vtargets.push_back(owner);
        }
      }

      vcount[i].insert(vcount[i].end(), vtargets.begin(), vtargets.end());      
      vtargets.clear();

      Edge const& edge_s = g_->get_edge(e0);
      Edge const& edge_t = g_->get_edge(e1-1);

      erange_[(i + base)*2] = edge_s.tail_;
      erange_[(i + base)*2+1] = edge_t.tail_;
    }
    
    MPI_Barrier(comm_);
    
    MPI_Allreduce(MPI_IN_PLACE, erange_, nv*2, MPI_GRAPH_TYPE, 
        MPI_SUM, comm_);

    GraphElem *send_count  = new GraphElem[size_]();
    GraphElem *recv_count  = new GraphElem[size_]();

    double t0 = MPI_Wtime();

    GraphElem nedges = 0;

    // perform local counting, identify edges and store targets 
    for (GraphElem i = 0; i < lnv; i++)
    {
      GraphElem e0, e1, tup[2];
      g_->edge_range(i, e0, e1);

      if ((e0 + 1) == e1)
        continue;

      for (GraphElem m = e0; m < e1; m++)
      {
        Edge const& edge_m = g_->get_edge(m);
        const int owner = g_->get_owner(edge_m.tail_);

        if (owner != rank_)
        {  
          nedges += vcount[i].size();
          for (int p : vcount[i])
            send_count[p] += 1;
        }
        else
        {
          if (m < (e1 - 1))
          {
            tup[0] = edge_m.tail_;
            for (GraphElem n = m + 1; n < e1; n++)
            {
              Edge const& edge_n = g_->get_edge(n);
              tup[1] = edge_n.tail_;

              if (check_edgelist(tup))
                ntriangles_ += 1;
            }
          }
          
          int past_target = -1;
          GraphElem l0, l1;
          const GraphElem lv = g_->global_to_local(edge_m.tail_);
          g_->edge_range(lv, l0, l1);
          
          for (GraphElem l = l0; l < l1; l++)
          {
            Edge const& edge = g_->get_edge(l);
            const int target = g_->get_owner(edge.tail_);
            if (target != rank_)
            {
              if (target != past_target)
              {
                send_count[target] += 1;
                nedges += 1;
                past_target = target;
              }
            }
          }
        }
      }
    }

    assert(nedges == std::accumulate(send_count, send_count + size_, 0));
    for (int p = 0; p < size_; p++)
    {
      if (send_count[p])
        targets_.push_back(p);
    }

    if (combufsize_ != -1)
    { 
      combufsize_ = (combufsize_ % 2 == 0) ? combufsize_ : combufsize_ + 1;
      for (int p : targets_)
        send_count[p] = MIN(send_count[p]*2, combufsize_);
    }
    else
    {
      for (int p : targets_)
        send_count[p] *= 2; 
    }

    // outgoing/incoming data and buffer size
    MPI_Alltoall(send_count, 1, MPI_GRAPH_TYPE, recv_count, 1, MPI_GRAPH_TYPE, comm_);

    MPI_Barrier(comm_);

    double t1 = MPI_Wtime();
    double p_tot = t1 - t0, t_tot = 0.0;

    MPI_Reduce(&p_tot, &t_tot, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

    if (rank_ == 0) 
    {   
      std::cout << "Average time for local counting during instantiation (secs.): " 
        << ((double)(t_tot / (double)size_)) << std::endl;
      if (combufsize_ != -1)
        std::cout << "Adjusted maximum per-PE communication buffer count: " << combufsize_ << std::endl;
    }

    // neighbor topology
    MPI_Dist_graph_create_adjacent(comm_, targets_.size(), targets_.data(), 
       MPI_UNWEIGHTED, targets_.size(), targets_.data(), MPI_UNWEIGHTED, 
       MPI_INFO_NULL, 0 /*reorder ranks?*/, &gcomm_);

    pdegree_ = targets_.size();

    for (int i = 0; i < pdegree_; i++)
      pindex_.insert({targets_[i], i});
  
    sebf_ = new Bloomfilter*[pdegree_]; 
    rebf_ = new Bloomfilter*[pdegree_]; 
    
    MPI_Barrier(comm_);

    GraphElem sdisp = 0, rdisp = 0;
    std::vector<int> scounts(pdegree_,0), rcounts(pdegree_,0), sdispl(pdegree_,0), rdispl(pdegree_,0);

    for (GraphElem p = 0; p < pdegree_; p++)
    {
        sebf_[p] = new Bloomfilter(send_count[targets_[p]]);
        scounts[p] = sebf_[p]->nbits();
        sdispl[p] = sdisp;
        sdisp += scounts[p];
              
        rebf_[p] = new Bloomfilter(recv_count[targets_[p]]);
        rcounts[p] = rebf_[p]->nbits();
        rdispl[p] = rdisp;
        rdisp += rcounts[p];
    }
    
    t0 = MPI_Wtime();

    if (rank_ == 0)
    {
      std::cout << "Bloom Filter details for PE #0:" << std::endl;
      sebf_[0]->print();   
    }

    sbuf_ = new char[sdisp];
    rbuf_ = new char[rdisp];
    std::memset(sbuf_, '0', sdisp);
    std::memset(rbuf_, '0', rdisp);
    
    for (GraphElem p = 0; p < pdegree_; p++)
    {
      sebf_[p]->set(sbuf_ + sdispl[p]);
      rebf_[p]->set(rbuf_ + rdispl[p]);
    }

    MPI_Barrier(comm_);

    // store edges in bloomfilter
    for (GraphElem i = 0; i < lnv; i++)
    {
      GraphElem e0, e1;
      g_->edge_range(i, e0, e1);

      if ((e0 + 1) == e1)
        continue;

      for (GraphElem m = e0; m < e1; m++)
      {
        Edge const& edge_m = g_->get_edge(m);
        const int owner = g_->get_owner(edge_m.tail_);

        if (owner != rank_)
        {
          for (int p : vcount[i])
            sebf_[pindex_[p]]->insert(g_->local_to_global(i), edge_m.tail_);
        }
        else
        {
          int past_target = -1;
          GraphElem l0, l1;
          const GraphElem lv = g_->global_to_local(edge_m.tail_);
          g_->edge_range(lv, l0, l1);
          for (GraphElem l = l0; l < l1; l++)
          {
            Edge const& edge = g_->get_edge(l);
            const int target = g_->get_owner(edge.tail_);
            if (target != rank_)
            {
              if (target != past_target)
              {
                sebf_[pindex_[target]]->insert(g_->local_to_global(i), edge_m.tail_);
                past_target = target;
              }
            }
          }
        }
      }
    }

    MPI_Barrier(comm_);

#if defined(USE_NBR_ALLTOALLV)
    MPI_Neighbor_alltoallv(sbuf_, scounts.data(), sdispl.data(), MPI_CHAR, 
        rbuf_, rcounts.data(), rdispl.data(), MPI_CHAR, gcomm_);
#else
    // extra neighbor function calls for demo purposes
    int indegree, outdegree, weighted;
    
    MPI_Dist_graph_neighbors_count(gcomm_, &indegree, &outdegree, &weighted);
    
    int *srcs = (int*)malloc(indegree*sizeof(int));
    int *dsts = (int*)malloc(outdegree*sizeof(int));
    
    MPI_Dist_graph_neighbors(gcomm_, indegree, srcs, MPI_UNWEIGHTED,
        outdegree, dsts, MPI_UNWEIGHTED);

    MPI_Request *sreqs = (MPI_Request*)malloc(outdegree*sizeof(MPI_Request));
    MPI_Request *rreqs = (MPI_Request*)malloc(indegree*sizeof(MPI_Request));

    for (int p = 0; p < outdegree; p++)
      MPI_Isend(sbuf_ + sdispl[p], scounts[p], MPI_CHAR, dsts[p], TAG_DATA, comm_, &sreqs[p]);

    for (int p = 0; p < indegree; p++)
      MPI_Irecv(rbuf_ + rdispl[p], rcounts[p], MPI_CHAR, srcs[p], TAG_DATA, comm_, &rreqs[p]);

    MPI_Waitall(outdegree, sreqs, MPI_STATUSES_IGNORE);
    MPI_Waitall(indegree, rreqs, MPI_STATUSES_IGNORE);

    free(srcs);
    free(dsts);
    free(sreqs);
    free(rreqs);
#endif
    MPI_Barrier(comm_);

    t1 = MPI_Wtime();
    double it_tot = t1 - t0, tt_tot = 0.0;

    MPI_Reduce(&it_tot, &tt_tot, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

    if (rank_ == 0) 
    {   
      std::cout << "Average time for local bloomfilter insertions and subsequent exchange (secs.): " 
        << ((double)(tt_tot / (double)size_)) << std::endl;
    }

#if defined(DEBUG_PRINTF)
    if (rank_ == 0)
    {
      std::cout << "Edge range per vertex (#ID: <range>): " << std::endl;
      for (int i = 0, j = 0; i < nv*2; i+=2, j++)
        std::cout << j << ": " << erange_[i] << ", " << erange_[i+1] << std::endl;
    }
#endif
    
    delete []send_count;
    delete []recv_count;

    for (int i = 0; i < lnv; i++)
      vcount[i].clear();
    vcount.clear();
 
    sdispl.clear();
    rdispl.clear();
    scounts.clear();
    rcounts.clear();
  }

    ~TriangulateHashRemote() {}

    void clear()
    {
      MPI_Comm_free(&gcomm_);
      
      if (rebf_ && sebf_)
      {
        for (int p = 0; p < pdegree_; p++)
        {
          rebf_[p]->clear();
          sebf_[p]->clear();
        }

        delete []rebf_;
        delete []sebf_;
      }
    
      delete []sbuf_;
      delete []rbuf_;

      pindex_.clear();
      targets_.clear();
      delete []erange_;
    }

    inline GraphElem count()
    {
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

          if (owner != rank_)
          {
            for (GraphElem n = m + 1; n < e1; n++)
            {
              Edge const& edge_n = g_->get_edge(n);
#if defined(DISABLE_EDGE_RANGE_CHECKS)
#else
              if (!edge_within_max(edge_m.tail_, edge_n.tail_))
                break;
              if (!edge_above_min(edge_m.tail_, edge_n.tail_) || !edge_above_min(edge_n.tail_, edge_m.tail_))
                continue;
#endif
              if (rebf_[pidx]->contains(edge_m.tail_, edge_n.tail_))
              {
                ntriangles_ += 1;
              }
            }
          }
        }
      }
      
      GraphElem ttc = 0, ltc = ntriangles_;
      MPI_Barrier(comm_);
      MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);

      return (ttc / 3);
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

  private:
    Graph* g_;

    GraphElem ntriangles_, pdegree_, combufsize_;
    GraphElem *erange_;
    Bloomfilter **sebf_, **rebf_;
    char *sbuf_, *rbuf_;

    std::vector<int> targets_;

    int rank_, size_;
    std::unordered_map<int, int> pindex_; 
    MPI_Comm comm_, gcomm_;
};
#endif
