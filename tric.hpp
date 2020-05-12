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

#define EDGE_SEARCH_TAG            1 
#define EDGE_INVALID_TAG           2
#define EDGE_VALID_NONCONS_TAG     3
#define EDGE_VALID_CONS_TAG        4

class Triangulate
{
    public:

        Triangulate(Graph* g): 
            g_(g), sbuf_ctr_(0), sreq_ctr_(0), tot_ghosts_(0),
            nghosts_(0), sbuf_(nullptr), sreq_(nullptr),
            ntriangles_(0)
        {
            comm_ = g_->get_comm();
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);

            const GraphElem lnv = g_->get_lnv();

            for (GraphElem i = 0; i < lnv; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);

                if ((e0 + 1) == e1)
                    continue;

                for (GraphElem e = e0 + 1; e < e1; e++)
                {
                    Edge const& edge_p = g_->get_edge(e - 1);
                    Edge const& edge_c = g_->get_edge(e);
                    if (g_->get_owner(edge_p.tail_) != rank_)
                        tot_ghosts_++;
                    if (g_->get_owner(edge_c.tail_) != rank_)
                        tot_ghosts_++;
                }
            }

            sbuf_ = new GraphElem[tot_ghosts_*3]; 
            sreq_ = new MPI_Request[tot_ghosts_*2*2];
            nghosts_ = tot_ghosts_; 
        }

        ~Triangulate() {}

        void clear()
        {
            delete []sbuf_;
            delete []sreq_;
        }

        // TODO
        inline void check()
        {
        }
        
        inline void isend(int tag, int target, GraphElem data[3])
        {
            memcpy(&sbuf_[sbuf_ctr_], data, 3*sizeof(GraphElem));

            MPI_Isend(&sbuf_[sbuf_ctr_], 3, MPI_GRAPH_TYPE, 
                    target, tag, comm_, &sreq_[sreq_ctr_]);

	    MPI_Request_free(&sreq_[sreq_ctr_]);
            
	    sbuf_ctr_ += 3;
	    sreq_ctr_++;
        }
        
        inline void isend(int tag, int target)
        {
            MPI_Isend(&sbuf_[sbuf_ctr_], 0, MPI_GRAPH_TYPE, 
                    target, tag, comm_, &sreq_[sreq_ctr_]);

	    MPI_Request_free(&sreq_[sreq_ctr_]);

	    sreq_ctr_++;
        }

        inline void lookup_edges()
        {
            const GraphElem lnv = g_->get_lnv();
            GraphElem tup_1[3] = {0}, tup_2[3] = {0};
                       
            for (GraphElem i = 0; i < lnv; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                
                if ((e0 + 1) == e1)
                    continue;

                tup_1[0] = g_->local_to_global(i);
                tup_2[0] = tup_1[0];
                
                for (GraphElem e = e0+1; e < e1; e++)
                {
                    Edge const& edge_p = g_->get_edge(e-1);
                    Edge const& edge_c = g_->get_edge(e);
                    
                    tup_1[1] = edge_p.tail_;
                    tup_1[2] = edge_c.tail_;
                    tup_2[1] = tup_1[2];
                    tup_2[2] = tup_1[1];
                    
                    const int owner_1 = g_->get_owner(tup_1[1]);
                    if (owner_1 == rank_)
                    {
                        int stat = check_edgelist(tup_1);
                        if (stat >= 0)
                        {
                            ntriangles_ += 1;
                            ntriangles_ += stat;
                        }
                    }
                    else
                        isend(EDGE_SEARCH_TAG, owner_1, tup_1);

                    const int owner_2 = g_->get_owner(tup_2[1]);
                    if (owner_2 == rank_)
                    {
                        if (check_edgelist(tup_2) == 1)
                            ntriangles_ += 1;
                    }
                    else
                        isend(EDGE_SEARCH_TAG, owner_2, tup_2);
                }
            }
        }
        
        inline int check_edgelist(GraphElem tup[3])
        {
            GraphElem e0, e1, c_e = 0, n_e = 0;
            const GraphElem lv = g_->global_to_local(tup[1]);
            g_->edge_range(lv, e0, e1);

            for (GraphElem e = e0; e < e1; e++)
            {
                Edge const& edge = g_->get_edge(e);

                if (edge.tail_ == tup[0])
                    c_e = e + 1;
                if (edge.tail_ == tup[2])
                    n_e = e + 1;

                if (c_e && n_e)
                    break;
            }

            if (n_e)
            {
                if (std::abs(n_e - c_e) > 1)
                    return 1;
                if (std::abs(n_e - c_e) == 1)
                    return 0;
            }

            return -1;
        }

        inline void process_edges()
        {
            MPI_Status status;
            int flag = -1;
            GraphElem tup[3] = {0};
            int count = 0;

            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_, 
                    &flag, &status);

            if (flag)
            {
                MPI_Get_count(&status, MPI_GRAPH_TYPE, &count);
                MPI_Recv(tup, count, MPI_GRAPH_TYPE, status.MPI_SOURCE, 
                        status.MPI_TAG, comm_, MPI_STATUS_IGNORE);   
            }
            else
                return;

            if (status.MPI_TAG == EDGE_SEARCH_TAG) 
            {
                int stat = check_edgelist(tup);

                if (stat == 1)
                    isend(EDGE_VALID_NONCONS_TAG, status.MPI_SOURCE);
                else if (stat == 0)
                    isend(EDGE_VALID_CONS_TAG, status.MPI_SOURCE);
                else // stat == -1
                    isend(EDGE_INVALID_TAG, status.MPI_SOURCE);
            }
            else if (status.MPI_TAG == EDGE_VALID_NONCONS_TAG)
            {
                ntriangles_ += 2;
                nghosts_ -= 1;
            }
            else if (status.MPI_TAG == EDGE_VALID_CONS_TAG)
            {
                ntriangles_ += 1;
                nghosts_ -= 1;
            }
            else // status.MPI_TAG == EDGE_INVALID_TAG
                nghosts_ -= 1;
        }

        inline GraphElem count()
        {
            GraphElem ng = 0;
            lookup_edges();

            while(1)
            {
                process_edges();

                MPI_Allreduce(&nghosts_, &ng, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
                if (ng == 0)
                    break;
            }
            
            GraphElem ttc;
            ntriangles_ /= 3;
            MPI_Reduce(&ntriangles_, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);

            return ttc;
        }
       
    private:
        Graph* g_;
        GraphElem lnv_;
        GraphElem ntriangles_;
        GraphElem tot_ghosts_, nghosts_;
        
	GraphElem *sbuf_;
        GraphElem sbuf_ctr_, sreq_ctr_;
        MPI_Request *sreq_;
        
	int rank_, size_;
        MPI_Comm comm_;
};

#endif
