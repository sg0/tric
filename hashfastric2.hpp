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
#ifndef HASH_TFASTRIC_HPP
#define HASH_TFASTRIC_HPP

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

      if (k_%2 != 0)
        k_ += 1;

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
      std::cout << "Maximum number of Items (n): " << n_ << std::endl;
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

#if defined(USE_STD_UNO_MUSET) 
struct pair_hash {
  inline std::size_t operator()(const std::pair<GraphElem,GraphElem> & v) const {
    GraphElem key[2] = {v.first, v.second}, hashes[2] = {0,0};
    MurmurHash3_x64_128 ( key, 2*sizeof(GraphElem), 0, hashes );
    return hashes[0] ^ hashes[1];
  }
};
#endif

class MapVec
{
  public:
    MapVec(): data_(), nkv_{{0, 0}} {}
    ~MapVec() 
    { data_.clear(); }

    inline void insert(GraphElem key, GraphElem value)
    {
#if defined(USE_STD_MUMAP) || defined(USE_STD_UNO_MUMAP)
      data_.insert({key, value});
      nkv_[0] += 1;
      nkv_[1] += 1;
#elif defined(USE_STD_UNO_MUSET)
      data_.insert(std::make_pair(key, value));
      nkv_[0] += 1;
      nkv_[1] += 1;
#else
      if (data_.count(key) > 0)
      {
          data_[key].emplace_back(value);
          nkv_[1] += 1;
      }
      else
      {
        data_.emplace(key, std::vector<GraphElem>());
        data_[key].emplace_back(value);
        nkv_[0] += 1;
        nkv_[1] += 1;
      }
#endif
    }

    inline bool contains(GraphElem key, GraphElem value)
    {
#if defined(USE_STD_UNO_MUMAP)
      std::pair <std::unordered_multimap<GraphElem,GraphElem>::iterator, std::unordered_multimap<GraphElem,GraphElem>::iterator> ret;
      ret = data_.equal_range(key);
      for (std::unordered_multimap<GraphElem, GraphElem>::iterator it = ret.first; it != ret.second; ++it)
      {
        if (it->second == value)
          return true;
      }
      return false;
#elif defined(USE_STD_UNO_MUSET)
      std::pair<GraphElem, GraphElem> pr = std::make_pair(key, value);   
      if (std::find_if(data_.begin(), data_.end(), 
              [pr](std::pair<GraphElem, GraphElem> const& prelem){ return prelem.first == pr.first && prelem.second == pr.second;}) == data_.end())
          return false;
      return true;
#elif defined(USE_STD_MUMAP)
      std::pair <std::multimap<GraphElem,GraphElem>::iterator, std::multimap<GraphElem,GraphElem>::iterator> ret;
      ret = data_.equal_range(key);
      for (std::multimap<GraphElem, GraphElem>::iterator it = ret.first; it != ret.second; ++it)
      {
        if (it->second == value)
          return true;
      }
      return false;
#else
      if (std::find_if(data_[key].begin(), data_[key].end(), 
            [value](GraphElem const& element){ return element == value;}) == data_[key].end())
        return false;
      return true;
#endif
    }

    inline void clear() 
    {
#if !defined(USE_STD_UNO_MUMAP) && !defined(USE_STD_UNO_MUSET) && !defined(USE_STD_MUMAP)
      for (auto it = data_.begin(); it != data_.end(); ++it)
        it->second.clear();
#endif
      data_.clear();
    }

    GraphElem size() const
    { return data_.size(); }

    void reserve(const GraphElem count)
    { 
#if defined(USE_STD_MAP) || defined(USE_STD_MUMAP)
#else
      data_.reserve(count); 
#endif
    }

    void print() const
    {
      std::cout << "-------------Map statistics-------------" << std::endl;
      std::cout << "Number of keys: " << nkv_[0] << std::endl;
      std::cout << "Number of values: " << nkv_[1] << std::endl;
      std::cout << "----------------------------------------" << std::endl;
    }   
  private:
#if defined(USE_STD_MAP)
    std::map<GraphElem, std::vector<GraphElem>> data_;
#elif defined(USE_STD_MUMAP)
    std::multimap<GraphElem, GraphElem> data_;
#elif defined(USE_STD_UNO_MUMAP)
    std::unordered_multimap<GraphElem, GraphElem> data_;
#elif defined(USE_STD_UNO_MUSET)
    std::unordered_multiset<std::pair<GraphElem, GraphElem>, pair_hash> data_;
#else
    std::unordered_map<GraphElem, std::vector<GraphElem>> data_;
#endif
    std::array<GraphElem,2> nkv_;
};

class TriangulateAggrBufferedHash2
{
  public:

    TriangulateAggrBufferedHash2(Graph* g, const GraphElem bufsize): 
      g_(g), sbuf_ctr_(nullptr), sbuf_(nullptr), rbuf_(nullptr), pdegree_(0), 
      sreq_(nullptr), erange_(nullptr), vcount_(nullptr), ntriangles_(0), nghosts_(0), 
      out_nghosts_(0), in_nghosts_(0), pindex_(0), prev_m_(nullptr), prev_k_(nullptr), 
      stat_(nullptr), ebf_(nullptr), targets_(0), bufsize_(0)
  {
    comm_ = g_->get_comm();
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

    const GraphElem lnv = g_->get_lnv();
    const GraphElem nv = g_->get_nv();

    vcount_ = new GraphElem[lnv]();
    erange_ = new GraphElem[nv*2]();
    std::vector<int> rdispls(size_, 0), sdispls(size_, 0), scounts(size_, 0), rcounts(size_, 0);
    std::vector<int> source_counts(size_, 0);
    GraphElem *send_count  = new GraphElem[size_]();
    GraphElem *recv_count  = new GraphElem[size_]();

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
          if (std::find(targets_.begin(), targets_.end(), owner) 
              == targets_.end())
            targets_.push_back(owner);
          
          nedges += 1;
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

          GraphElem l0, l1;
          bool is_remote = false;
          const GraphElem lv = g_->global_to_local(edge_m.tail_);

          g_->edge_range(lv, l0, l1);

          for (GraphElem l = l0; l < l1; l++)
          {
            Edge const& edge = g_->get_edge(l);
            if (g_->get_owner(edge.tail_) != rank_)
            {
              is_remote = true;
              break;
            }
          }

          if (is_remote)
            nedges += 1;
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

    pdegree_ = targets_.size();

    for (int i = 0; i < pdegree_; i++)
      pindex_.insert({targets_[i], i});
    
    t0 = MPI_Wtime();

    MPI_Barrier(comm_);
    
    if (nedges)
    {
#if defined(USE_BLOOMFILTER) 
      ebf_ = new Bloomfilter(nedges*2);
      if (rank_ == 0)
        ebf_->print();
#else
      ebf_ = static_cast<MapVec*>(new MapVec());
      ebf_->reserve(nedges*2);
#endif
    }

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
          ebf_->insert(g_->local_to_global(i), edge_m.tail_);
        
          if (m < (e1 - 1))
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
              send_count[owner] += 1;
              vcount_[i] += 1;
            }
          }
        }
        else
        {
          GraphElem l0, l1;
          bool is_remote = false;
          const GraphElem lv = g_->global_to_local(edge_m.tail_);
          g_->edge_range(lv, l0, l1);
          for (GraphElem l = l0; l < l1; l++)
          {
            Edge const& edge = g_->get_edge(l);
            if (g_->get_owner(edge.tail_) != rank_)
            {
              is_remote = true;
              break;
            }
          }

          if (is_remote)
            ebf_->insert(g_->local_to_global(i), edge_m.tail_);
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
    
    nghosts_ = MAX(out_nghosts_, in_nghosts_);
    bufsize_ = ((nghosts_*2) < bufsize) ? (nghosts_*2) : bufsize;
    MPI_Allreduce(MPI_IN_PLACE, &bufsize_, 1, MPI_GRAPH_TYPE, MPI_MAX, comm_);

    if (rank_ == 0)
      std::cout << "Adjusted Per-PE buffer count: " << bufsize_ << std::endl;
    
    rbuf_     = new GraphElem[bufsize_];
    sbuf_     = new GraphElem[pdegree_*bufsize_];
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
  }

    ~TriangulateAggrBufferedHash2() {}

    void clear()
    {
      delete []sbuf_;
      delete []rbuf_;
      delete []sbuf_ctr_;
      delete []sreq_;
      delete []prev_k_;
      delete []prev_m_;
      delete []stat_;
      delete []vcount_;
      delete []erange_;
      
      if (ebf_)
      {
        ebf_->clear();
        delete ebf_;
      }

      pindex_.clear();
      targets_.clear();
    }

    void nbsend(GraphElem owner)
    {
      if (sbuf_ctr_[pindex_[owner]] > 0)
      {
        MPI_Isend(&sbuf_[pindex_[owner]*bufsize_], sbuf_ctr_[pindex_[owner]], 
            MPI_GRAPH_TYPE, owner, TAG_DATA, comm_, &sreq_[pindex_[owner]]);
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
                stat_[pidx] = '1'; // messages in-flight

                nbsend(owner);

                continue;
              }

              sbuf_[disp+sbuf_ctr_[pidx]] = edge.edge_->tail_;
              sbuf_ctr_[pidx] += 1;

              for (GraphElem n = ((prev_k_[pidx] == -1) ? (m + 1) : prev_k_[pidx]); n < e1; n++)
              {  
                Edge const& edge_n = g_->get_edge(n);                                                     
#if defined(DISABLE_EDGE_RANGE_CHECKS)
#else           
                if (!edge_within_max(edge.edge_->tail_, edge_n.tail_))
                  break;
                if (!edge_above_min(edge.edge_->tail_, edge_n.tail_) || !edge_above_min(edge_n.tail_, edge.edge_->tail_))
                  continue;
#endif
                if (sbuf_ctr_[pidx] == (bufsize_-1))
                {
                  prev_m_[pidx] = m;
                  prev_k_[pidx] = n;

                  sbuf_[disp+sbuf_ctr_[pidx]] = -1; // demarcate vertex boundary
                  sbuf_ctr_[pidx] += 1;
                  stat_[pidx] = '1'; 

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
                  stat_[pidx] = '1';

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
      GraphElem tup[2] = {-1,-1}, k = 0, prev = 0;
      int count = 0, source = -1;

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

        GraphElem curr_count = 0;
        tup[0] = rbuf_[k];

        for (GraphElem m = k + 1; m < count; m++)
        {
          if (rbuf_[m] == -1)
          {
            curr_count = m + 1;
            break;
          }

          tup[1] = rbuf_[m];

#if defined(DEFAULT_EDGE_QUERY)
          if (check_edgelist(tup))
#else
          if (ebf_->contains(tup[0], tup[1]))
#endif
            ntriangles_ += 1;

          in_nghosts_ -= 1;
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

      return (ttc / 3);
    }

  private:
    Graph* g_;

    GraphElem ntriangles_, bufsize_, nghosts_, out_nghosts_, in_nghosts_, pdegree_;
    GraphElem *sbuf_, *rbuf_, *prev_k_, *prev_m_, *sbuf_ctr_, *vcount_, *erange_;
    MPI_Request *sreq_;
    char *stat_;
#if defined(USE_BLOOMFILTER)
    Bloomfilter *ebf_;
#else
    MapVec *ebf_;
#endif

    std::vector<int> targets_;

    int rank_, size_;
    std::unordered_map<int, int> pindex_; 
    MPI_Comm comm_;
};
#endif
