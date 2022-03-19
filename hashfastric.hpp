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
        
    Bloomfilter(GraphElem n, GraphElem k, GraphWeight p=BLOOMFILTER_TOL) 
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

  private:
    GraphElem n_, m_, k_;
    GraphWeight p_;

    void hash( uint64_t lhs, uint64_t rhs ) 
    {
      uint64_t key[2] = {lhs, rhs};
      for (uint64_t n = 0; n < k_; n+=2)
      {
        MurmurHash3_x64_128 ( &key, 2*sizeof(uint64_t), n, &hashes_[n] );
        hashes_[n] = hashes_[n] % m_; 
        hashes_[n+1] = hashes_[n+1] % m_;
      }
    }
    
    std::vector<char> bits_;
    std::vector<uint64_t> hashes_;
};

class TriangulateHashBased
{
  public:

    TriangulateHashBased(Graph* g): 
      g_(g), sbf_(nullptr), rbf_(nullptr), pbf_(nullptr), pdegree_(0), ntriangles_(0), 
      targets_(0), gcomm_(MPI_COMM_NULL)
  {
    comm_ = g_->get_comm();
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

    double t0 = MPI_Wtime();
    
    const GraphElem lnv = g_->get_lnv();
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
            targets_.push_back(owner);
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

    if (size_ > 1)
    {
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
        pindex_.insert({targets_[i], i});

      sbf_ = new Bloomfilter*[pdegree_];
      rbf_ = new Bloomfilter*[pdegree_];

      t0 = MPI_Wtime();

      std::vector<int> rdispls(pdegree_, 0), sdispls(pdegree_, 0), scounts(pdegree_, 0), rcounts(pdegree_, 0);
      std::vector<int> source_counts(pdegree_, 0);
      GraphElem *send_count  = new GraphElem[pdegree_]();
      GraphElem *recv_count  = new GraphElem[pdegree_]();

      const int targets_size = targets_.size();
      MPI_Neighbor_allgather(&targets_size, 1, MPI_INT, source_counts.data(), 1, MPI_INT, gcomm_);

      int sdisp = 0, rdisp = 0;

      for (GraphElem p = 0; p < pdegree_; p++)
      {
        rdispls[p] = rdisp;
        rdisp += source_counts[p];
      }

      std::vector<int> source_data(rdisp, 0);

      MPI_Neighbor_allgatherv(targets_.data(), targets_size, MPI_INT, source_data.data(), 
          source_counts.data(), rdispls.data(), MPI_INT, gcomm_);
      
      // TODO FIXME overallocation & multiple
      // insertions (wasted cycles), unique 
      // neighbors + 1 is ideal
      pbf_ = new Bloomfilter(rdisp);

      for (int p = 0; p < pdegree_; p++)
      {
        for (int n = 0; n < rdisp; n+=source_counts[p])
        {
          if (rank_ != source_data[n])
            pbf_->insert(targets_[p], source_data[n]);
        }
      }

      // TODO FIXME this is wrong, send_count
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
            if (pbf_->contains(rank_, owner))
              send_count[pindex_[owner]] += 1;
          }
        }
      }

      MPI_Neighbor_alltoall(send_count, 1, MPI_GRAPH_TYPE, recv_count, 1, MPI_GRAPH_TYPE, gcomm_);

      sdisp = 0;
      rdisp = 0;

      for (GraphElem p = 0; p < pdegree_; p++)
      {
        sbf_[p] = new Bloomfilter(send_count[p]);
        sdispls[p] = sdisp;
        scounts[p] = sbf_[p]->nbits();
        sdisp += scounts[p];

        rbf_[p] = new Bloomfilter(recv_count[p]);
        rdispls[p] = rdisp;
        rcounts[p] = rbf_[p]->nbits();
        rdisp += rcounts[p];
      }

      // store remote edges in bloomfilter and communicate 
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
            if (pbf_->contains(rank_, owner))
              sbf_[pindex_[owner]]->insert(edge_m.tail_, g_->local_to_global(i));
          }
        }
      }

      MPI_Barrier(comm_);

      char *sbuf = new char[sdisp];
      char *rbuf = new char[rdisp];
      GraphElem c = 0;

      for(GraphElem p = 0; p < pdegree_; p++)
      {
        sbf_[p]->copy_to(&sbuf[c]);
        c += sbf_[p]->nbits();
      }

      MPI_Neighbor_alltoallv(sbuf, scounts.data(), sdispls.data(), 
          MPI_CHAR, rbuf, rcounts.data(), rdispls.data(), MPI_CHAR, gcomm_);   

      c = 0;
      for(GraphElem p = 0; p < pdegree_; p++)
      {
        rbf_[p]->copy_from(&rbuf[c]);
        c += rbf_[p]->nbits();
      }

      t1 = MPI_Wtime();
      p_tot = t1 - t0, t_tot = 0.0;

      MPI_Reduce(&p_tot, &t_tot, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

      if (rank_ == 0) 
      {   
        std::cout << "Average time for inserting and exchanging bloomfilter bits (secs.): " 
          << ((double)(t_tot / (double)size_)) << std::endl;
      }

      delete []send_count;
      delete []recv_count;
      delete []sbuf;
      delete []rbuf;

      sdispls.clear();
      rdispls.clear();
      scounts.clear();
      rcounts.clear();
      sdispls.clear();
      rdispls.clear();
      source_counts.clear();
      source_data.clear();
    }
  }

    ~TriangulateHashBased() {}

    void clear()
    {
      MPI_Comm_free(&gcomm_);
      
      pindex_.clear();
      targets_.clear();

      for(GraphElem p = 0; p < pdegree_; p++)
      {
        rbf_[p]->clear();
        sbf_[p]->clear();
      }

      if (pbf_)
        pbf_->clear();

      delete []rbf_;
      delete []sbf_;
    }

    inline GraphElem count()
    {
      const GraphElem lnv = g_->get_lnv();
      GraphElem rtriangles = 0;

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
            if (pbf_->contains(rank_, owner))
            {
              for (GraphElem n = m + 1; n < e1; n++)
              { 
                Edge const& edge_n = g_->get_edge(n);                                
                if (rbf_[pidx]->contains(edge_m.tail_, edge_n.tail_))
                  rtriangles += 1;
              }
            }
          }
        }
      }
      
      GraphElem ttc[2] = {0,0}, ltc[2] = {ntriangles_, rtriangles};
      MPI_Barrier(comm_);
      MPI_Reduce(ltc, ttc, 2, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);

      return (ttc[0] / 3) + (ttc[1] / 2);
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

  private:
    Graph* g_;
    Bloomfilter **rbf_, **sbf_, *pbf_;

    GraphElem ntriangles_, pdegree_;
    char *sbuf_, *rbuf_; 

    std::vector<int> targets_;

    int rank_, size_;
    std::unordered_map<int, int> pindex_; 
    MPI_Comm comm_, gcomm_;
};
#endif
