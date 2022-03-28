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
#ifndef BHASH_TFASTRIC_HPP
#define BHASH_TFASTRIC_HPP

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
      : n_(pow(2, std::ceil(log(n)/log(2)))), p_(p)
    {
      m_ = std::ceil((n_ * log(p_)) / log(1 / pow(2, log(2))));
      k_ = std::round((m_ / n_) * log(2));

      hashes_.resize(k_); 
      bits_.resize(m_);
      std::fill(bits_.begin(), bits_.end(), '0');

      if (k_ == 0)
        throw std::invalid_argument("Bloomfilter could not be initialized: k must be larger than 0");
    }
        
    Bloomfilter(GraphElem n, GraphElem k, GraphWeight p) 
      : n_(pow(2, std::ceil(log(n)/log(2)))), k_(k), p_(p)
    {
      m_ = std::ceil((n_ * log(p_)) / log(1 / pow(2, log(2))));

      if (k_%2 != 0)
        k_ += 1;

      hashes_.resize(k_); 
      bits_.resize(m_);
      std::fill(bits_.begin(), bits_.end(), '0');

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

    void clear()
    {
        bits_.clear(); 
        hashes_.clear(); 
    }

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

    GraphElem nbits() const
    { return m_; }

    // "nucular" options, use iff 
    // you know what you're doing
    void copy_from(char* source)
    { std::memcpy(bits_.data(), source, m_); }
      
    void copy_to(char* dest)
    { std::memcpy(dest, bits_.data(), m_); }

    void zfill() 
    { std::fill(bits_.begin(), bits_.end(), '0'); }

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
    
    std::vector<char> bits_;
    std::vector<uint64_t> hashes_;
};

class TriangulateAggrBufferedHashPush
{
  public:

    TriangulateAggrBufferedHashPush(Graph* g, const GraphElem bufsize): 
      g_(g), sbuf_ctr_(nullptr), pdegree_(0), vcount_(0), erange_(nullptr), 
      ntriangles_(0), pindex_(0), prev_m_(nullptr), prev_k_(nullptr), prev_i_(-1), 
      targets_(0), bufsize_(bufsize), sebf_(nullptr), rebf_(nullptr), sbuf_(nullptr), 
      rbuf_(nullptr), send_count_(0), gcomm_(MPI_COMM_NULL)
  {
    comm_ = g_->get_comm();
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

    const GraphElem lnv = g_->get_lnv();
    const GraphElem nv = g_->get_nv();

    erange_ = new GraphElem[nv*2]();
    
    double t0 = MPI_Wtime();

    std::vector<int> vtargets; 
    vcount_.resize(lnv);

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

      vcount_[i].insert(vcount_[i].end(), vtargets.begin(), vtargets.end());      
      vtargets.clear();

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

      for (GraphElem m = e0; m < e1; m++)
      {
        Edge const& edge_m = g_->get_edge(m);
        const int owner = g_->get_owner(edge_m.tail_);

        if (owner != rank_)
        {  
          if (std::find(targets_.begin(), targets_.end(), owner) 
              == targets_.end())
            targets_.push_back(owner);

          send_count_ += vcount_[i].size();
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
            if ((target != rank_) && (target != past_target))
            {
              send_count_ += 1;
              past_target = target;
            }
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
      std::cout << "Average time for local counting and misc. during instantiation (secs.): " 
        << ((double)(t_tot / (double)size_)) << std::endl;
    }

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
      pindex_.insert({targets_[i], i});
    
    sebf_ = new Bloomfilter*[pdegree_]; 
    rebf_ = new Bloomfilter*[pdegree_]; 
   
    GraphElem count = 0;
    for (GraphElem p = 0; p < pdegree_; p++)
    {
        sebf_[p] = new Bloomfilter(bufsize_);
        rebf_[p] = new Bloomfilter(bufsize_);
    }

    if (sebf_)
      count = sebf_[0]->nbits();

    sbuf_ = new char[count*pdegree_];
    rbuf_ = new char[count*pdegree_];

    if (rank_ == 0)
      std::cout << "Per-PE buffer count: " << bufsize_ << std::endl;
    
    sbuf_ctr_ = new GraphElem[pdegree_]();
    prev_k_   = new GraphElem[pdegree_];
    prev_m_   = new GraphElem[pdegree_];
    
    std::fill(prev_k_, prev_k_ + pdegree_, -1);
    std::fill(prev_m_, prev_m_ + pdegree_, -1);

    MPI_Barrier(comm_);

#if defined(DEBUG_PRINTF)
    if (rank_ == 0)
    {
      std::cout << "Edge range per vertex (#ID: <range>): " << std::endl;
      for (int i = 0, j = 0; i < nv*2; i+=2, j++)
        std::cout << j << ": " << erange_[i] << ", " << erange_[i+1] << std::endl;
    }
#endif
    
    vtargets.clear();
  }

    ~TriangulateAggrBufferedHashPush() {}

    void clear()
    {
      MPI_Comm_free(&gcomm_);

      delete []sbuf_;
      delete []rbuf_;
      delete []sbuf_ctr_;
      delete []prev_k_;
      delete []prev_m_;
      delete []erange_;
      
      if (sebf_ && rebf_)
      {
        for (int i = 0; i < pdegree_; i++)
        {
            sebf_[i]->clear();
            rebf_[i]->clear();

            delete []sebf_[i];
            delete []rebf_[i];
        }

        delete[] sebf_;
        delete[] rebf_;
      }

      pindex_.clear();
      targets_.clear();
    }

    inline GraphElem count()
    {
      bool done = false, nbar_active = false; 
      MPI_Request nbar_req = MPI_REQUEST_NULL;
      
      std::vector<int> displs(size_, 0), counts(size_, 0);
      GraphElem disp = 0;

      for (GraphElem p = 0; p < size_; p++)
      {
        if (p == targets_[p])
          counts[p] = sebf_[p]->nbits(); 
        displs[p] = disp;
        disp += counts[p];
      }
    
      const GraphElem lnv = g_->get_lnv();

      while(!done)
      {
        for (GraphElem i = ((prev_i_ == -1) ? 0 : prev_i_); i < lnv; i++)
        {
          GraphElem e0, e1, tup[2];
          g_->edge_range(i, e0, e1);

          if ((e0 + 1) == e1)
            continue;

          for (GraphElem m = e0; m < e1; m++)
          {
            Edge const& edge_m = g_->get_edge(m);
            const int owner = g_->get_owner(edge_m.tail_);
            const GraphElem pidx = pindex_[owner];

            if (owner != rank_)
            {
              if (m >= prev_m_[pidx])
              {
                if (sbuf_ctr_[pidx] == bufsize_)
                {
                  prev_m_[pidx] = m;
                  prev_k_[pidx] = -1;
                  prev_i_ = i;

                  break;
                }

                for (int p : vcount_[i])
                {
                  sebf_[pindex_[p]]->insert(g_->local_to_global(i), edge_m.tail_);
                  send_count_ -= 1;
                  sbuf_ctr_[pidx] += 1;
                }
              }
            }
            else
            {
              int past_target = -1;
              GraphElem l0, l1;

              const GraphElem lv = g_->global_to_local(edge_m.tail_);
              g_->edge_range(lv, l0, l1);

              for (GraphElem l = ((prev_k_[pidx] == -1) ? l0 : prev_k_[pidx]); l < l1; l++)
              {
                Edge const& edge = g_->get_edge(l);
                const int target = g_->get_owner(edge.tail_);

                if ((target != rank_) && (target != past_target))
                {
                  if (sbuf_ctr_[pindex_[target]] == bufsize_)
                  {
                    prev_m_[pindex_[target]] = m;
                    prev_k_[pindex_[target]] = l;
                    prev_i_ = i;

                    break;
                  }

                  sebf_[pindex_[target]]->insert(g_->local_to_global(i), edge_m.tail_);
                  sbuf_ctr_[pindex_[target]] += 1;
                  send_count_ -= 1;
                  past_target = target;
                }
              }
            }
          }
        }

        MPI_Barrier(comm_);

        for(GraphElem p = 0; p < pdegree_; p++)
          sebf_[p]->copy_from(&sbuf_[displs[targets_[p]]]);

        MPI_Alltoallv(sbuf_, counts.data(), displs.data(), MPI_CHAR, rbuf_, 
            counts.data(), displs.data(), MPI_CHAR, comm_);   

        for(GraphElem p = 0; p < pdegree_; p++)
          rebf_[p]->copy_to(&rbuf_[displs[targets_[p]]]);

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

                if (!edge_within_max(edge_m.tail_, edge_n.tail_))
                  break;
                if (!edge_above_min(edge_m.tail_, edge_n.tail_) || !edge_above_min(edge_n.tail_, edge_m.tail_))
                  continue;

                if (rebf_[pidx]->contains(edge_m.tail_, edge_n.tail_))
                  ntriangles_ += 1;
              }
            }
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
          if (send_count_ == 0)
          {
            MPI_Ibarrier(comm_, &nbar_req);
            nbar_active = true;
          }
        }
      }

      counts.clear();
      displs.clear();

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

    GraphElem ntriangles_, bufsize_, pdegree_, send_count_, prev_i_;
    GraphElem *prev_k_, *prev_m_, *sbuf_ctr_, *erange_;
    
    Bloomfilter **sebf_, **rebf_;
    char *sbuf_, *rbuf_;

    std::vector<int> targets_;
    std::vector<std::vector<int>> vcount_;

    int rank_, size_;
    std::unordered_map<int, int> pindex_; 
    MPI_Comm comm_, gcomm_;
};
#endif
