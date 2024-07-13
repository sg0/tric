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
#ifndef MAP_NCOLL_HPP
#define MAP_NCOLL_HPP

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
    
    GraphElem size() const { return data_.size(); }
    GraphElem count() const { return count_ + this->size(); }

    void print() const
    {
      for (auto it = data_.begin(); it != data_.end(); ++it)
      {
        std::cout << "map[" << it->first << "]: ";
        for (auto vit = it->second.begin(); vit != it->second.end(); ++vit)
          std::cout << vit->first << "," << vit->second << " ";
        std::cout << std::endl;
      }
      std::cout << "#Elements (keys/values): " << count() << std::endl;
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

class TriangulateMapNcol
{
  public:

  TriangulateMapNcol(Graph* g): 
      g_(g), sbuf_(0), rbuf_(0), pdegree_(0), erange_(nullptr), 
      ntriangles_(0), pindex_(0), rindex_(0), sources_(0), targets_(0), 
      edge_map_(0), rdegree_(0), scounts_(0), rcounts_(0), sdispls_(0), 
      rdispls_(0), gcomm_(MPI_COMM_NULL)
  {
    comm_ = g_->get_comm();
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

    const GraphElem lnv = g_->get_lnv();
    GraphElem *send_count  = new GraphElem[size_]();
    GraphElem *recv_count  = new GraphElem[size_]();

    double t0 = MPI_Wtime();
    
    if (size_ > 1)
    {
      const GraphElem nv = g_->get_nv();

      erange_ = new GraphElem[nv*2];

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
    }

#if defined(USE_OPENMP)
#pragma omp declare reduction(merge : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for default(shared) reduction(+:ntriangles_) reduction(merge: targets_) schedule(static)
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
             
            if (!edge_above_min(edge_m.tail_, edge_n.tail_))
              continue;

            if (!edge_within_max(edge_m.tail_, edge_n.tail_))
              break;

#if defined(USE_OPENMP)
#pragma omp atomic
#endif
            send_count[owner] += 1;
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
     
    if (size_ > 1)
    {       
      // outgoing/incoming data and buffer size
      MPI_Alltoall(send_count, 1, MPI_GRAPH_TYPE, recv_count, 1, MPI_GRAPH_TYPE, comm_);

      for (GraphElem p = 0; p < size_; p++)
      {
        if (send_count[p] > 0)
          targets_.push_back(p);

        if (recv_count[p] > 0)
          sources_.push_back(p);
      }

      pdegree_ = targets_.size(); 
      rdegree_ = sources_.size();

      for (int i = 0; i < pdegree_; i++)
        pindex_.insert({targets_[i], i});

      for (int i = 0; i < rdegree_; i++)
        rindex_.insert({sources_[i], i});

      edge_map_.resize(pdegree_);
      scounts_.resize(pdegree_);
      sdispls_.resize(pdegree_);
      rcounts_.resize(rdegree_);
      rdispls_.resize(rdegree_);

      MPI_Dist_graph_create_adjacent(comm_, sources_.size(), (int*)sources_.data(), 
          MPI_UNWEIGHTED, targets_.size(), (int*)targets_.data(), MPI_UNWEIGHTED, 
          MPI_INFO_NULL, 0 /*reorder ranks?*/, &gcomm_);

      // indegree/outdegree checking...
      int weighted, indegree, outdegree;
      MPI_Dist_graph_neighbors_count(gcomm_, &indegree, &outdegree, &weighted);

      // to get sources/targets, use MPI_Dist_graph_neighbors
      assert(indegree == rdegree_);
      assert(outdegree == pdegree_);
    }

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

    ~TriangulateMapNcol() {}

    void clear()
    {
      if (size_ > 1)
        delete []erange_;

      sbuf_.clear();
      rbuf_.clear();

      pindex_.clear();
      rindex_.clear();
      targets_.clear();
      sources_.clear();
      
      for (int s = 0; s < pdegree_; s++)
          edge_map_[s].clear();
      edge_map_.clear();

      scounts_.clear();
      rcounts_.clear();
      sdispls_.clear();
      rdispls_.clear();

      if (gcomm_ != MPI_COMM_NULL)
          MPI_Comm_free(&gcomm_);
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

    inline void nalltoall_params()
    {
      MPI_Aint disp = 0;
      for (auto const& p: targets_)
      {
        sdispls_[pindex_[p]] = disp;

        if (edge_map_[pindex_[p]].size() > 0)
            scounts_[pindex_[p]] = edge_map_[pindex_[p]].count() + edge_map_[pindex_[p]].size();
        
        disp += scounts_[pindex_[p]];
      }

      sbuf_.resize(disp);
      disp = 0;

      if (scounts_.data() != nullptr)
          MPI_Neighbor_alltoall_c(scounts_.data(), 1, MPI_COUNT, rcounts_.data(), 1, MPI_COUNT, gcomm_);

      for (auto const& p: sources_)
      {
        rdispls_[rindex_[p]] = disp;
        disp += rcounts_[rindex_[p]];
      }

      rbuf_.resize(disp);
    }

    inline void nalltoallv()
    {
      nalltoall_params();
      
      for (auto const& p: targets_)
      {
          if (edge_map_[pindex_[p]].size() > 0)
              edge_map_[pindex_[p]].serialize(&sbuf_[sdispls_[pindex_[p]]]);
      }

      if (sbuf_.data() != nullptr)
          MPI_Neighbor_alltoallv_c(sbuf_.data(), scounts_.data(), sdispls_.data(), MPI_GRAPH_TYPE, 
          rbuf_.data(), rcounts_.data(), rdispls_.data(), MPI_GRAPH_TYPE, gcomm_);
    }

    inline void lookup_edges()
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

              if (!edge_above_min(edge_m.tail_, edge_n.tail_))
                continue;

              if (!edge_within_max(edge_m.tail_, edge_n.tail_))
                break;

              edge_map_[pidx].insert(edge_m.tail_, edge_n.tail_);
            }
          }
        }
      }
    }

    inline void process_incoming()
    {
      GraphElem tup[2] = {0};
      for (GraphElem k = 0; k < rbuf_.size();)
      {
        tup[0] = rbuf_[k];

        for (GraphElem m = k + 1; m < rbuf_.size(); m+=2)
        {
          if (rbuf_[m] == -1)
          {
            k = m + 1;
            break;
          }

          tup[1] = rbuf_[m];

          if (check_edgelist(tup))
            ntriangles_ += rbuf_[m+1];
        }
      }
    }

    inline GraphElem count()
    {
      if (size_ > 1)
      {
        lookup_edges();
        nalltoallv();
        process_incoming();
      }

      GraphElem ttc = 0, ltc = ntriangles_;
      MPI_Barrier(comm_);
      MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);

      clear();

      return (ttc/3);
    }

  private:
    Graph* g_;

    GraphElem ntriangles_, pdegree_, rdegree_;
    GraphElem *erange_;

    std::vector<GraphElem> sbuf_, rbuf_; 
    std::vector<int> targets_, sources_;
    std::vector<MapUniq> edge_map_;

    std::vector<MPI_Count> scounts_, rcounts_;
    std::vector<MPI_Aint> sdispls_, rdispls_;

    int rank_, size_;
    std::unordered_map<GraphElem, GraphElem> pindex_, rindex_; 
    MPI_Comm comm_, gcomm_;
};
#endif
