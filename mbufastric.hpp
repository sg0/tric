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
#ifndef MAP_HFASTRIC_HPP
#define MAP_HFASTRIC_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <map>
#include <unordered_map>

#if __cplusplus >= 202002L
#include <ranges>
#endif

#ifndef TAG_DATA
#define TAG_DATA 100
#endif

class MapUniq
{
  public:
    MapUniq(): count_(0), data_() {}
    ~MapUniq() 
    { 
      data_.clear(); 
      count_ = 0;
    }

    inline void insert(GraphElem key, GraphElem value)
    {
      if (data_.count(key) > 0)
      {
#if __cplusplus >= 202002L
        auto it = std::ranges::find(data_[key], value, &std::pair<GraphElem, GraphElem>::first);
#else
        auto it = std::find_if(data_[key].begin(), data_[key].end(),
                [&value](const std::pair<GraphElem, GraphElem>& element){ return element.first == value;} );
#endif
        if (it != data_[key].end())
          it->second += 1;
        else
        {
#if defined(USE_STD_MAP_MAP) || defined(USE_STD_MAP_UNO_MAP) || defined(USE_STD_UNO_MAP_MAP) || defined(USE_STD_UNO_MAP_UNO_MAP)
          data_[key].insert(std::pair<GraphElem, GraphElem>(value, 1));
#else
          data_[key].emplace_back(std::pair<GraphElem, GraphElem>(value, 1));
#endif
          count_ += 2;
        }
      }
      else
      {
#if defined(USE_STD_MAP_MAP) || defined(USE_STD_MAP_UNO_MAP) || defined(USE_STD_UNO_MAP_MAP) || defined(USE_STD_UNO_MAP_UNO_MAP)
#if defined(USE_STD_MAP_MAP) || defined(USE_STD_UNO_MAP_MAP)
        data_.emplace(key, std::map<GraphElem, GraphElem>());
#else
        data_.emplace(key, std::unordered_map<GraphElem, GraphElem>());
#endif
        data_[key].insert(std::pair<GraphElem, GraphElem>(value, 1));
#else
        data_.emplace(key, std::vector<std::pair<GraphElem, GraphElem>>());
        data_[key].emplace_back(std::pair<GraphElem, GraphElem>(value, 1));
#endif
        count_ += 2;
      }
    }

    inline void clear() 
    {
      for (auto it = data_.begin(); it != data_.end(); ++it)
        it->second.clear();
      data_.clear();
      count_ = 0;
    }

    inline void serialize(GraphElem* ptr)
    {
      for (auto it = data_.begin(); it != data_.end(); ++it)
      {
        *ptr++ = it->first;
        for (auto vit = it->second.begin(); vit != it->second.end(); ++vit)
        {
          *ptr++ = vit->first;
          *ptr++ = vit->second;
        }
        *ptr++ = -1;
      }
    }

    GraphElem size() const
    { return data_.size(); }
    
    GraphElem count() const
    { return count_ + this->size(); }

    GraphElem do_count() const
    {
      GraphElem mcnt = this->size(); 
      
      for (auto it = data_.begin(); it != data_.end(); ++it)
        for (auto vit = it->second.begin(); vit != it->second.end(); ++vit)
          mcnt += 2;
      
      return mcnt;
    }

    void print() const
    {
      std::cout << "#Elements (keys/values): " << this->count() << std::endl;
      for (auto it = data_.begin(); it != data_.end(); ++it)
      {
        std::cout << "map[" << it->first << "]: ";
        for (auto vit = it->second.begin(); vit != it->second.end(); ++vit)
          std::cout << vit->first << "," << vit->second << " ";
        std::cout << std::endl;
      }
    }

    void reserve(const size_t count)
    { data_.reserve(count); }

  private:
    GraphElem count_;
#if defined(USE_STD_MAP_MAP)
    std::map<GraphElem, std::map<GraphElem, GraphElem>> data_;
#elif defined(USE_STD_UNO_MAP_MAP)
    std::unordered_map<GraphElem, std::map<GraphElem, GraphElem>> data_;
#elif defined(USE_STD_MAP_UNO_MAP)
    std::map<GraphElem, std::unordered_map<GraphElem, GraphElem>> data_;
#elif defined(USE_STD_UNO_MAP_UNO_MAP)
    std::unordered_map<GraphElem, std::unordered_map<GraphElem, GraphElem>> data_;
#elif defined(USE_STD_MAP)
    std::map<GraphElem, std::vector<std::pair<GraphElem, GraphElem>>> data_;
#else
    std::unordered_map<GraphElem, std::vector<std::pair<GraphElem, GraphElem>>> data_;
#endif
};

class TriangulateAggrBufferedMap
{
  public:

    TriangulateAggrBufferedMap(Graph* g, const GraphElem bufsize): 
      g_(g), sbuf_(nullptr), rbuf_(nullptr), pdegree_(0), sreq_(nullptr), 
      erange_(nullptr), vcount_(nullptr), ntriangles_(0), nghosts_(0), 
      out_nghosts_(0), in_nghosts_(0), pindex_(0), prev_m_(nullptr), 
      prev_k_(nullptr), stat_(nullptr), targets_(0), bufsize_(0), edge_map_(0)
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
          if (std::find(targets_.begin(), targets_.end(), owner) 
              == targets_.end())
            targets_.push_back(static_cast<GraphElem>(owner));

          for (GraphElem n = m + 1; n < e1; n++)
          {
            Edge const& edge_n = g_->get_edge(n);
                
            if (!edge_within_max(edge_m.edge_->tail_, edge_n.tail_))
              break;
            if (!edge_above_min(edge_m.edge_->tail_, edge_n.tail_) || !edge_above_min(edge_n.tail_, edge_m.edge_->tail_))
              continue;

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
    
    pdegree_ = targets_.size(); 

    for (int i = 0; i < pdegree_; i++)
      pindex_.insert({targets_[i], static_cast<GraphElem>(i)});

    MPI_Alltoall(send_count, 1, MPI_GRAPH_TYPE, recv_count, 1, MPI_GRAPH_TYPE, comm_);

    for (GraphElem p = 0; p < size_; p++)
    {
      out_nghosts_ += send_count[p];
      in_nghosts_ += recv_count[p];
    }
    
    nghosts_ = out_nghosts_ + in_nghosts_;
        
    bufsize_ = ((nghosts_*3) < bufsize) ? (nghosts_*3) : bufsize;
    bufsize_ -= bufsize_%3;
    MPI_Allreduce(MPI_IN_PLACE, &bufsize_, 1, MPI_GRAPH_TYPE, MPI_MAX, comm_);

    if (rank_ == 0)
      std::cout << "Adjusted Per-PE buffer count: " << bufsize_ << std::endl;
 
    rbuf_     = new GraphElem[bufsize_];
    sbuf_     = new GraphElem[pdegree_*bufsize_];
    prev_k_   = new GraphElem[pdegree_];
    prev_m_   = new GraphElem[pdegree_];
    stat_     = new char[pdegree_];
    sreq_     = new MPI_Request[pdegree_];

    edge_map_.resize(pdegree_);

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

    ~TriangulateAggrBufferedMap() {}

    void clear()
    {
      delete []sbuf_;
      delete []rbuf_;
      delete []sreq_;
      delete []prev_k_;
      delete []prev_m_;
      delete []stat_;
      delete []vcount_;
      delete []erange_;

      pindex_.clear();
      targets_.clear();
      edge_map_.clear();
    }

    // TODO
    inline void check()
    {
    }

    void flatten_nbsend(GraphElem owner)
    {
      if (edge_map_[pindex_[owner]].size() > 0)
      {
        edge_map_[pindex_[owner]].serialize(&sbuf_[pindex_[owner]*bufsize_]);

        MPI_Isend(&sbuf_[pindex_[owner]*bufsize_], 
            edge_map_[pindex_[owner]].size() + edge_map_[pindex_[owner]].count(),
            MPI_GRAPH_TYPE, owner, TAG_DATA, comm_, &sreq_[pindex_[owner]]);      
      }
    }

    void flatten_nbsend()
    {
      for (int const& p : targets_)
        flatten_nbsend(p);
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
              if ((edge_map_[pidx].size() + edge_map_[pidx].count()) >= (bufsize_ - 3)) // 3 because an insertion could be the triplet: key:{val,count}
              {
                prev_m_[pidx] = m;
                prev_k_[pidx] = -1;
                stat_[pidx] = '1'; // messages in-flight

                flatten_nbsend(owner);

                continue;
              }

              for (GraphElem n = ((prev_k_[pidx] == -1) ? (m + 1) : prev_k_[pidx]); n < e1; n++)
              {  
                Edge const& edge_n = g_->get_edge(n);                                
                
                if (!edge_within_max(edge.edge_->tail_, edge_n.tail_))
                  break;
                if (!edge_above_min(edge.edge_->tail_, edge_n.tail_) || !edge_above_min(edge_n.tail_, edge.edge_->tail_))
                  continue;

                if ((edge_map_[pidx].size() + edge_map_[pidx].count()) >= (bufsize_ - 3))
                {
                  prev_m_[pidx] = m;
                  prev_k_[pidx] = n;
                  stat_[pidx] = '1'; 

                  flatten_nbsend(owner);

                  break;
                }
                  
                out_nghosts_ -= 1;
                vcount_[i] -= 1;
                edge_map_[pidx].insert(edge.edge_->tail_, edge_n.tail_);
              }
              
              if (stat_[pidx] == '0') 
              {               
                prev_m_[pidx] = m;
                prev_k_[pidx] = -1;
                
                edge.active_ = false;
                
                if ((edge_map_[pidx].size() + edge_map_[pidx].count()) >= (bufsize_ - 3))
                {
                  stat_[pidx] = '1';
                  flatten_nbsend(owner);
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
      GraphElem tup[2] = {-1,-1};
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

      for (GraphElem k = 0; k < count;)
      {
        tup[0] = rbuf_[k];

        for (GraphElem m = k + 1; m < count; m+=2)
        {
          if (rbuf_[m] == -1)
          {
            k = m + 1;
            break;
          }

          tup[1] = rbuf_[m];

          if (check_edgelist(tup))
            ntriangles_ += rbuf_[m+1];

          in_nghosts_ -= rbuf_[m+1];
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
      int odd_or_even = 0;

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
            flatten_nbsend();
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
            stat_[idx] = '0';
            edge_map_[idx].clear();
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

    GraphElem ntriangles_, bufsize_, nghosts_, out_nghosts_, in_nghosts_, pdegree_;
    GraphElem *sbuf_, *rbuf_, *prev_k_, *prev_m_, *vcount_, *erange_;
    MPI_Request *sreq_;
    char *stat_;

    std::vector<GraphElem> targets_;
    std::vector<MapUniq> edge_map_;

    int rank_, size_;
    std::unordered_map<GraphElem, GraphElem> pindex_; 
    MPI_Comm comm_;
};
#endif
