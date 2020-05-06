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
#ifndef COMM_HPP
#define COMM_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>

#define EDGE_SEARCH_TAG    1 
#define EDGE_INVALID_TAG   2
#define EDGE_VALID_TAG     3

class Triangulate
{
    public:

        Triangulate(Graph* g): 
            g_(g), sbuf_ctr_(0), sreq_ctr_(0),tot_ghosts_(0),
            nghosts_(0), sbuf_(nullptr), sreq_(nullptr),
            ntriangles_(0), g_ntriangles_(0)
        {
            comm_ = g_->get_comm();
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);

            const GraphElem lnv = g_->get_lnv();
                      
            ghost_count_.resize(lnv, 0);

            for (GraphElem i = 0; i < lnv; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);

                for (GraphElem e = e0; e < e1; e += 2)
                {
                    Edge const& edge = g_->get_edge(e);
                    if (g_->get_owner(edge.tail_) != rank_)
                        ghost_count_[i] += 1;
                }
               
                tot_ghosts_ += ghost_count_[i];
            }
            
            sbuf_ = new GraphElem[tot_ghosts_*2]; 
            sreq_ = new MPI_Request[tot_ghosts_*2];
            nghosts_ = tot_ghosts_;
        }

        ~Triangulate() {}

        void clear()
        {
            ghost_count_.clear();
            delete []sbuf_;
            delete []sreq_;
        }

        // TODO
        inline void check()
        {
        }
        
        inline void isend(int tag, int target, GraphElem data[2])
        {
            memcpy(&sbuf_[sbuf_ctr_], data, 2*sizeof(GraphElem));

            MPI_Isend(&sbuf_[sbuf_ctr_], 2, MPI_GRAPH_TYPE, 
                    target, tag, comm_, &sreq_[sreq_ctr_]);

	    MPI_Request_free(&sreq_[sreq_ctr_]);
            
	    sbuf_ctr_ += 2;
	    sreq_ctr_++;
        }
        
        inline void isend(int tag, int target)
        {
            MPI_Isend(&sbuf_[sbuf_ctr_], 0, MPI_GRAPH_TYPE, 
                    target, tag, comm_, &sreq_[sreq_ctr_]);

	    MPI_Request_free(&sreq_[sreq_ctr_]);

	    sreq_ctr_++;
        }

        inline void possible_edges()
        {
            const GraphElem lnv = g_->get_lnv();
            GraphElem pair[2] = {0};
                       
            for (GraphElem i = 0; i < lnv; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);

                for (GraphElem e = e0; e < e1; e += 2)
                {
                    Edge const& edge_p = g_->get_edge(e);
                    Edge const& edge_c = g_->get_edge(e + 1);
                    pair[0] = edge_p.tail_;
                    pair[1] = edge_c.tail_;

                    is_edge(pair);
                }
            }
        }

        inline void is_edge(GraphElem pair[2])
        {
            const int owner = g_->get_owner(pair[0]);
            if (owner == rank_)
                check_edgelist(pair);
            else
                isend(EDGE_SEARCH_TAG, owner, pair);
        }
        
        inline bool check_edgelist(GraphElem pair[2], bool remote=false)
        {
            GraphElem e0, e1;
            g_->edge_range(pair[0], e0, e1);

            for (GraphElem e = e0; e < e1; e++)
            {
                Edge const& edge = g_->get_edge(e);
                if (edge.tail_ == pair[1])
                {
                    if (!remote)
                        ntriangles_++;
                    else
                        g_ntriangles_++;
                    return true;
                }
            }

            return false;
        }

        inline void process_edges()
        {
            MPI_Status status;
            int flag = -1;
            GraphElem g_l[2] = {0};
            int count = 0;

            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_, 
                    &flag, &status);

            if (flag)
            {
                MPI_Get_count(&status, MPI_GRAPH_TYPE, &count);
                MPI_Recv(g_l, count, MPI_GRAPH_TYPE, status.MPI_SOURCE, 
                        status.MPI_TAG, comm_, MPI_STATUS_IGNORE);   
                nghosts_--;
            }
            else
                return;

            if (status.MPI_TAG == EDGE_SEARCH_TAG) 
            {
                g_l[0] = g_->global_to_local(g_l[0]);

                if (!check_edgelist(g_l, true))
                    isend(EDGE_INVALID_TAG, status.MPI_SOURCE);
                else
                    isend(EDGE_VALID_TAG, status.MPI_SOURCE);
            }

            if (status.MPI_TAG == EDGE_VALID_TAG)
                g_ntriangles_++;
        }

        inline GraphElem count()
        {
            GraphElem ng = 0;
            possible_edges();

            while(1)
            {
                process_edges();

                MPI_Allreduce(&nghosts_, &ng, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
                if (ng == 0)
                    break;
            }
            
	    MPI_Waitall(tot_ghosts_, sreq_, MPI_STATUSES_IGNORE);    
            GraphElem tcount =  ntriangles_ + (g_ntriangles_ / 2);
            GraphElem ttc;
            MPI_Reduce(&tcount, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);

            return ttc;
        }
       
    private:
        Graph* g_;
        GraphElem lnv_;
        GraphElem ntriangles_, g_ntriangles_;
        std::vector<GraphElem> ghost_count_;
        GraphElem tot_ghosts_, nghosts_;
        
	GraphElem *sbuf_;
        GraphElem sbuf_ctr_, sreq_ctr_;
        MPI_Request *sreq_;
        
	int rank_, size_;
        MPI_Comm comm_;
};

#endif
