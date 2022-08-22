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

#ifndef COMM_BUFSIZE 
#define COMM_BUFSIZE (262144)
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
      bits_.resize(m_+sizeof(GraphElem)*8);
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
      bits_.resize(m_+sizeof(GraphElem)*8);
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

    const char* data() const
    { return bits_.data(); }
    
    char* data()
    { return bits_.data(); }
    
    // "nucular" options, use iff 
    // you know what you're doing
    void copy_from(char* dest)
    { std::memcpy(dest, bits_.data(), m_); }
      
    void copy_to(char* source)
    { std::memcpy(bits_.data(), source, m_); }

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

    TriangulateAggrBufferedHashPush(Graph* g, const GraphElem bufsize, const GraphElem combufsize=COMM_BUFSIZE): 
      g_(g), sbuf_ctr_(nullptr), sbuf_(nullptr), rbuf_(nullptr), pdegree_(0), 
      sreq_(nullptr), erange_(nullptr), vcount_(nullptr), ntriangles_(0), 
      nghosts_(0), out_nghosts_(0), in_nghosts_(0), pindex_(0), prev_m_(nullptr), 
      prev_k_(nullptr), stat_(nullptr), targets_(0), bufsize_(0), combufsize_(combufsize) 
  {
    comm_ = g_->get_comm();
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

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

    pdegree_ = targets_.size();

    for (int i = 0; i < pdegree_; i++)
      pindex_.insert({targets_[i], i});
    
    t0 = MPI_Wtime();

    GraphElem *send_count  = new GraphElem[size_]();
    GraphElem *recv_count  = new GraphElem[size_]();

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

        if (owner != rank_)
        {  
          for (GraphElem n = m + 1; n < e1; n++)
          {
            Edge const& edge_n = g_->get_edge(n);
                            
            if (!edge_within_max(edge_m.edge_->tail_, edge_n.tail_))
              break;
            if (!edge_above_min(edge_m.edge_->tail_, edge_n.tail_) || !edge_above_min(edge_n.tail_, edge_m.edge_->tail_))
              continue;
          
            send_count[owner] += 2;
            vcount_[i] += 1;
          }
        }
      }
    }

    // outgoing/incoming data and buffer size
    MPI_Alltoall(send_count, 1, MPI_GRAPH_TYPE, recv_count, 1, MPI_GRAPH_TYPE, comm_);
    
    for (GraphElem p = 0; p < size_; p++)
    {
      out_nghosts_ += send_count[p];
      in_nghosts_ += recv_count[p];
    }
    
    nghosts_ = out_nghosts_ + in_nghosts_;

    bufsize_ = ((MAX(out_nghosts_, in_nghosts_)) < bufsize) ? (MAX(out_nghosts_, in_nghosts_)) : bufsize;
    bufsize_ = (bufsize_ % 2 == 0) ? bufsize_ : bufsize_ + 1;
    MPI_Allreduce(MPI_IN_PLACE, &bufsize_, 1, MPI_GRAPH_TYPE, MPI_MAX, comm_);

    combufsize_ = ((MAX(out_nghosts_, in_nghosts_)) < combufsize) ? (MAX(out_nghosts_, in_nghosts_)) : combufsize;
    combufsize_ = (combufsize_ % 2 == 0) ? combufsize_ : combufsize_ + 1;
    MPI_Allreduce(MPI_IN_PLACE, &combufsize_, 1, MPI_GRAPH_TYPE, MPI_MAX, comm_);

    sbuf_ = new Bloomfilter*[pdegree_]; 
    rbuf_ = new Bloomfilter(combufsize_); 

    for (GraphElem p = 0; p < pdegree_; p++)
      sbuf_[p] = new Bloomfilter(combufsize_);

    if (rank_ == 0)
      sbuf_[0]->print();   

    if (rank_ == 0)
    {
      std::cout << "Adjusted Per-PE buffer count: " << bufsize_ << std::endl;
      std::cout << "Adjusted Per-PE communication buffer count: " << combufsize_ << std::endl;
    }

    sbuf_ctr_ = new GraphElem[pdegree_]();
    prev_k_   = new GraphElem[pdegree_];
    prev_m_   = new GraphElem[pdegree_];
    stat_     = new char[pdegree_];
    sreq_     = new MPI_Request[pdegree_];

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
    
    delete []send_count;
    delete []recv_count;

    MPI_Barrier(comm_);
  }

    ~TriangulateAggrBufferedHashPush(){}

    void clear()
    {
      for (int i = 0; i < pdegree_; i++)
        sbuf_[i]->clear();
      rbuf_->clear();

      delete sbuf_;
      delete rbuf_;

      delete []sbuf_ctr_;
      delete []sreq_;
      delete []prev_k_;
      delete []prev_m_;
      delete []stat_;
      delete []vcount_;
      delete []erange_;

      pindex_.clear();
      targets_.clear();
    }

    void nbsend(int owner)
    {
      if (sbuf_ctr_[pindex_[owner]] > 0)
      {
        std::string tmp = std::to_string(sbuf_ctr_[pindex_[owner]]);
        char const *num_char = tmp.c_str();
        std::memcpy(sbuf_[pindex_[owner]]->data() + sbuf_[pindex_[owner]]->nbits(), num_char, sizeof(GraphElem)*8);

        MPI_Isend(sbuf_[pindex_[owner]]->data(), sbuf_[pindex_[owner]]->nbits() + sizeof(GraphElem)*8, MPI_CHAR, owner, TAG_DATA, comm_, &sreq_[pindex_[owner]]);
      }
    }

    void nbsend()
    {
      for (int const& p : targets_)
        nbsend(p);
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
              if (sbuf_ctr_[pidx] == bufsize_)
              {
                prev_m_[pidx] = m;
                prev_k_[pidx] = -1;
                stat_[pidx] = '1'; // messages in-flight

                nbsend(owner);

                continue;
              }

              for (GraphElem n = ((prev_k_[pidx] == -1) ? (m + 1) : prev_k_[pidx]); n < e1; n++)
              {  
                Edge const& edge_n = g_->get_edge(n);                                
                                
                if (!edge_within_max(edge.edge_->tail_, edge_n.tail_))
                  break;
                if (!edge_above_min(edge.edge_->tail_, edge_n.tail_) || !edge_above_min(edge_n.tail_, edge.edge_->tail_))
                  continue;

                if (sbuf_ctr_[pidx] == bufsize_)
                {
                  prev_m_[pidx] = m;
                  prev_k_[pidx] = n;

                  stat_[pidx] = '1'; 

                  nbsend(owner);

                  break;
                }
                              
                sbuf_[pidx]->insert(edge.edge_->tail_, edge_n.tail_);
                
                sbuf_ctr_[pidx] += 2;
                out_nghosts_ -= 2;
                vcount_[i] -= 1;
              }
              
              if (stat_[pidx] == '0') 
              {               
                prev_m_[pidx] = m;
                prev_k_[pidx] = -1;
                
                edge.active_ = false;

                if (sbuf_ctr_[pidx] == bufsize_)
                {
                  stat_[pidx] = '1';

                  nbsend(owner);
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

    inline void process_messages()
    {
      MPI_Status status;
      int flag = -1;
      int source = -1;

      MPI_Iprobe(MPI_ANY_SOURCE, TAG_DATA, comm_, &flag, &status);

      if (flag)
      { 
        source = status.MPI_SOURCE;
        MPI_Recv(rbuf_->data(), rbuf_->nbits() + sizeof(GraphElem)*8, MPI_CHAR, source, TAG_DATA, comm_, MPI_STATUS_IGNORE);       
      }
      else
        return;

      char *num_char;
      GraphElem count = strtol(rbuf_->data() + rbuf_->nbits(), &num_char, 10);
      in_nghosts_ -= count;

      const GraphElem lnv = g_->get_lnv();
      
      for (GraphElem i = 0; i < lnv; i++)
      {
        GraphElem e0, e1;
        g_->edge_range(i, e0, e1);

        if ((e0 + 1) == e1)
          continue;

        const GraphElem g_i = g_->local_to_global(i);
        
        for (GraphElem e = e0; e < e1; e++)
        {
          Edge const& edge = g_->get_edge(e);
          if (rbuf_->contains(g_i, edge.tail_))
            ntriangles_ += 1;
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

      bool sends_done = false;
      int *inds = new int[pdegree_];
      int over = -1;

#if defined(USE_ALLREDUCE_FOR_EXIT)
      while(1)
#else
      while(!done)
#endif
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
            GraphElem idx = static_cast<GraphElem>(inds[i]);
            sbuf_ctr_[idx] = 0;
            stat_[idx] = '0';
            sbuf_[idx]->zfill();
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

      free(inds);

      return (ttc/3);
    }

  private:
    Graph* g_;

    GraphElem ntriangles_, combufsize_, bufsize_, nghosts_, out_nghosts_, in_nghosts_, pdegree_;
    GraphElem *prev_k_, *prev_m_, *sbuf_ctr_, *vcount_, *erange_;
    Bloomfilter **sbuf_, *rbuf_; 
    MPI_Request *sreq_;
    char *stat_;

    std::vector<int> targets_;

    int rank_, size_;
    std::unordered_map<int, int> pindex_; 
    MPI_Comm comm_;
};
#endif
