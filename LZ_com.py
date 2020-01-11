#!/usr/bin/env python
# encoding:UTF-8

import networkx as nx
from networkx.algorithms import community    # for community structure later
import collections
from matplotlib import pyplot as plt
from networkx.algorithms import approximation as app
import operator
from networkx.algorithms.community import LFR_benchmark_graph
import math
import time
from itertools import repeat
import copy
import random
import pandas as pd
from functools import reduce
from scipy.special import comb, perm

from optparse import OptionParser
from funcs import *

def Matrix_shrink_D(d_temp,ii,jj):
    # shrink the distance matrix by one column and one row
    k1 = d_temp[:,ii]
    k2 = d_temp[:,jj]

    dd = np.delete(d_temp,[ii,jj],1)
    dd = np.delete(dd,[ii,jj],0)
    kk = np.maximum(k1,k2)
    kk = np.delete(kk,[ii,jj],0)
    m1 = np.vstack([dd,kk])
    m2 = np.append(kk,0)
    shrank = np.vstack([m1.T,m2])
    return shrank

def Matrix_shrink_J(jac_temp,ii,jj,temp,tempp):
    # shrink the Jaccard matrix by one column and one row
    # temp: current small-communities
    # tempp: two small-communities in process
    global source

    dd = np.delete(jac_temp,[ii,jj],1)
    dd = np.delete(dd,[ii,jj],0)
    
    kk = []
    b = set()
    num_jj = len(tempp[0])
    for j in range((num_jj)):
        temp_b = set([k[0] for k in community[source.index(tempp[0][j])]])
        b = b | temp_b 
        
    num_ii = len(tempp[1])
    for j in range((num_ii)):
        temp_b = set([k[0] for k in community[source.index(tempp[1][j])]])
        b = b | temp_b 
    
    for i in range(len(temp)-1):
        
        if type(temp[i]) != list:
            num_i = 1
            temp[i] = [temp[i]]
        else:
            num_i = len(temp[i])
            
        a = set()
       
        for j in range((num_i)):
            temp_a = set([k[0] for k in community[source.index(temp[i][j])]])
            a = a | temp_a 
        
        c = a.intersection(b)

        j_score = float(len(c))/(len(a) + len(b) - len(c))
        kk.append(j_score)
        
    m1 = np.vstack([dd,kk])
    m2 = np.append(kk,1)
    shrank = np.vstack([m1.T,m2])
    return shrank

def Merged_community(heirarchy,cutoff_t):
    merged_community = []
    merged_timestamp = []
    for i in range(len(heirarchy)):
        temp = []
        timestamp = []
        for j in range(len(heirarchy[i])):
            for k in community[source.index(heirarchy[i][j])]:
                if k[1] < cutoff_t:
                    temp.append(k[0])    
                    timestamp.append(k[1]) 
                     
        temp = list(set(temp))
        timestamp = list(set(timestamp))
        merged_community.append(temp)   
        merged_timestamp.append(timestamp)
        
    return merged_community,merged_timestamp

## hierarchy_pos ##

def hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, 
                  pos = None, parent = None):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
       pos: a dict saying where all nodes go if they have been assigned
       parent: parent of this branch.'''
    if pos == None:
        pos = {root:(xcenter,vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    neighbors = list(G.neighbors(root)) 
    if parent != None:   #this should be removed for directed graphs.
        neighbors.remove(parent)  #if directed, then parent not in neighbors.
    if len(neighbors)!=0:
        dx = width/len(neighbors) 
        nextx = xcenter - width/2 - dx/2
        for neighbor in neighbors:
            nextx += dx
            pos = hierarchy_pos(G,neighbor, width = dx, vert_gap = vert_gap, 
                                vert_loc = vert_loc-vert_gap, xcenter=nextx, pos=pos, 
                                parent = root)
    return pos

class MyParser(OptionParser):
    def format_epilog(self, formatter):
        return self.epilog


if __name__ == '__main__':
    usage = "usage: python LZ_com.py [options]"
    parser = MyParser(usage)
    parser.add_option("-d", "--delimiter", dest="delimiter", default=' ',
                      help="delimiter of input & output files [default: space]")
    parser.add_option("-f", "--input network file", dest="network_file", default='data/CA-GrQc.txt',
                      help="input file of edge list for clustering [default: data/CA-GrQc.txt]")
    parser.add_option("--out", "--output file", dest="output_file", default="output/",
                      help="output file of detected community [default: output/]")
   
    (options, args) = parser.parse_args()
    delimiter = options.delimiter
    network_file = options.network_file
    output_file = options.output_file

    G  = nx.read_edgelist(network_file, create_using = nx.Graph(), nodetype = int)
    
    graphname = str(random.randint(10000,99999))
    
    ## remove self loops & degree = 0 #
    G.remove_edges_from(G.selfloop_edges())
    isola = [k for k in nx.isolates(G)]
    G.remove_nodes_from(isola)

    Dict_Centrality = nx.degree_centrality(G)
    Centrality = list(Dict_Centrality.values())
    Name = list(Dict_Centrality.keys())
    A = nx.adjacency_matrix(G)
    
    # main #

    # STEP 1: Identification of sources ##

    print('#####  STEP 1  #####')
    print('--------------------')
    start_s1 = time.clock()

    source = []
    sink = []
    iso = []

    nodetemp = list(G.nodes)

    count_s1 = 0
    for i in nodetemp:

        count_s1 += 1
        if count_s1%1000 == 0:
            print('Time Elapsed--- '+ str((time.clock() - start_s1)) + ' Node:' + str(count_s1)+ '/' + str(len(G)) +'\n')

        nei = list(G.neighbors(i))
        iso_count = 0
        source_count = 0
        sink_count = 0

        if len(nei) == 1:
            continue

        for ii in nei:
            if Dict_Centrality.get(i) > Dict_Centrality.get(ii):   
                source_count += 1

            elif Dict_Centrality.get(i) == Dict_Centrality.get(ii):
                iso_count  += 1
                source_count += 1
            else:
                sink_count += 1
                continue

        if iso_count == G.degree(i):

            if all(Centrality[Name.index(p)] == Centrality[Name.index(i)] for p in list(G.neighbors(i))): # clique    
                if not any(w in source for w in list(G.neighbors(i))):
                    source.append(i)   # get one as hub, the other are inner members
                    Centrality[Name.index(i)] += 0.5  # additive value to this hub
            else: 
                iso.append(i)      # non-clique

        if source_count == G.degree(i):     
            if i not in iso and i not in source:    # source: greater than at least one neighbor in centrality score
                source.append(i)
        if sink_count == G.degree(i) & G.degree(i) > 1:
            sink.append(i)

    r_source = len(source)/len(G)   # proportion of source
    r_sink = len(sink)/len(G)       # proportion of sink

    leaf = sum([int(G.degree(i) == 1) for i in G.nodes()])
    inner = len(G) - len(source) - len(sink) - len(iso) - leaf


    # STEP 2: Propagation and Formulation of Local Communities ##  

    print('#####  STEP 2  #####')
    print('--------------------')

    from itertools import repeat
    import copy
    import time

    start_s2 = time.clock()

    History = [[] for i in repeat(None, len(nx.nodes(G)))]
    community = [[] for i in repeat(None, len(source))]

    t = 0
    tmax = 100

    time_record = []

    for i in range(len(source)):
        community[i].append((source[i],t))    # first label , first contagion time
        History[Name.index(source[i])] = [(source[i],0)]

    while t < tmax:
        old_community = copy.deepcopy(community)
        old_history = copy.deepcopy(History)
        t = t + 1

        for i in range(len(source)):   # all propagation happens at the same time
            if (i+1)%100 == 0:
                print('Iteration:'+ str(t) + '/' + str(tmax)+ '---' +'Source:' + str(i+1) + '/' + str(len(source)) + '---Time Elapsed---'+ str((time.clock() - start_s2)) + '---CommunitySize---'+ str(len(community[i])))

            for j in community[i]:             
                if j[1] == t-1: # newly join the community from last round propagation 
                    for s in G.neighbors(j[0]):
                        if Centrality[Name.index(s)] < Centrality[Name.index(j[0])]:    

                            if s not in [k[0] for k in community[i]]:
                                community[i].append((s,t))  
                                History[Name.index(s)].append((source[i],t))

        print('Time Elapsed--- '+ str((time.clock() - start_s2)))
        time_record.append((time.clock() - start_s2))

        if old_community == community or old_history == History:   # no change in History or community membership        
            break
        # check History and community are consistent #

        if sum(len(History[i]) for i in range(len(History))) != sum(len(community[i]) for i in range(len(community))):
            print('WRONG! COMMUNITY AND HISTORY DONT MATCH!')


    ave_membership = sum(len(History[i]) for i in range(len(History)))/len(History)
    ave_size = sum(len(community[i]) for i in range(len(community)))/len(community)


    elapsed = (time.clock() - start_s2)


    # plot local communities #
    from matplotlib import colors as mcolors
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    old_co = list(community)
    old_his = list(History)


    len_hist = [len(hh) for hh in History]
    r_crossover = len(len_hist) - len_hist.count(1)

    # STEP 3&4: Aggregation of Small Communities ##

    print('##### STEP 3&4 #####')
    print('--------------------')
    # distance and Jaccard index between communities #
    import numpy as np

    start_s3 = time.clock()

    d_source = np.zeros((len(source),len(source)))
    jac_source = np.zeros((len(source),len(source)))

    dummy = 9999

    for i in range(len(source)):
        if i%100 == 0:
            print('Shortest Path:' + str(i) + '/'+ str(len(source))+ '---' +  'Time Elapsed:'+ str((time.clock() - start_s3)))
        for j in range(len(source)):

            if nx.has_path(G,source[i],source[j]):
                d_source[i,j] = nx.shortest_path_length(G,source=source[i],target=source[j])
            else:
                d_source[i,j] = dummy      # unconnected components

            a = set([k[0] for k in community[i]])
            b = set([k[0] for k in community[j]])

            c = a.intersection(b)
            jac_source[i,j] = float(len(c))/(len(a) + len(b) - len(c))

    mixed_source = d_source/jac_source

    print('Time Elapsed--- '+ str((time.clock() - start_s3)))


    # epsilon #
    epsilon_max = int(d_source.max())   # the largest distance between sources  --- could be dummy!
    heirarchy_community = [list(source)]
    heirarchy_for_plot = list(source)
    epsilon_community_size = [(len(d_source),0)]

    d_temp = d_source
    d_record = [list(d_temp)]

    jac_temp = jac_source
    jac_record = [list(jac_temp)]

    all_jd_count = 0
    jd_record = []  ## record good merging

    phi_list = []  ## list of phi-epsilon
    phi_ref_list = [] ## list of reference phi-epsilon
    merging_count = 0    # count of num of merging (in each epsilon)

    for l in range(epsilon_max):

        print('Epsilon:' + str(l) + '/' + str(epsilon_max) + '---' + 'Time Elapsed:'+ str((time.clock() - start_s3)))
        temp = list(heirarchy_community[-1])
        jd_count = 0 ## quality check
        merging_count = 0    # count of num of merging (in each epsilon)

        while True:
            # find element == l + 1
            ij = np.argwhere(d_temp==l+1) # Note: l starts from 0

            if len(ij) == 0:   # no element == l+1
                break

            merging_count += 1

            ii = ij[0][0]
            jj = ij[0][1]

            if type(temp[ii]) != list:    # str to list
                temp[ii] = [temp[ii]]
            if type(temp[jj]) != list:    # str to list
                temp[jj] = [temp[jj]]

            temp_com = temp[ii] + temp[jj]

            tempp = [temp[ii],temp[jj]]

            tempp_copy = list(tempp)
            if len(temp[ii]) == 1:
                tempp_copy[0] = temp[ii][0]
            if len(temp[jj]) == 1:
                tempp_copy[1] = temp[jj][0]

            tempp_copy = [heirarchy_for_plot[ii],heirarchy_for_plot[jj]]

            # Append the new combined community !!! as the last element, 
            # so that the new community should be put in the last row of the new matrix
            temp.remove(tempp[0])   # remove old small community 1
            temp.remove(tempp[1])   # remove old small community 2
            temp.append(temp_com)  

            ## Find the JD inconsistency element
            JD_flag_not = 0

            if JD_flag_not == 0:    # row
                lk = np.where(jac_temp[ii,:] > jac_temp[ii,jj])
                lk = lk[0].tolist()
                for h in range(len(lk)):
                    if d_temp[lk[h],ii] > l+1 or d_temp[lk[h],jj] > l+1:   # not the same level with ii and jj
                        JD_flag_not = 1
                        break

            if JD_flag_not == 0:    # column
                lk = np.where(jac_temp[:,jj] > jac_temp[ii,jj])
                lk = lk[0].tolist()
                for h in range(len(lk)):
                    if d_temp[lk[h],ii] > l+1 or d_temp[lk[h],jj] > l+1:   # not the same level with ii and jj
                        JD_flag_not = 1
                        break

            # JD consistent
            if JD_flag_not == 0: 

                jd_count += 1
                all_jd_count += 1

                jd_record.append(tempp)

            # Shrink d and jac
            d_temp = Matrix_shrink_D(d_temp,ii,jj)        
            jac_temp = Matrix_shrink_J(jac_temp,ii,jj,temp,tempp) ## temp: community assignment in the current round

            # for plot

            heirarchy_for_plot.remove(tempp_copy[0])

            heirarchy_for_plot.remove(tempp_copy[1])

            heirarchy_for_plot.append(tempp_copy) 

        d_record.append(d_temp)
        jac_record.append(jac_temp)
        heirarchy_community.append(temp)
        epsilon_community_size.append((len(d_temp),l+1))


        if merging_count > 0:
            phi_list.append([jd_count/merging_count,jd_count,merging_count])
            phi_ref_list.append(jd_count/merging_count - (comb(merging_count,jd_count)*(0.5**merging_count)))
        else:
            phi_list.append([1,jd_count,merging_count])
            phi_ref_list.append(1)

        ## unconnected components ##
        if len(np.argwhere(d_temp==dummy)) == len(d_temp)*(len(d_temp)-1):
            break

    epsilon_max = l+1

    ## refine heirarchy_community 0 ##
    for i in range(len(heirarchy_community[0])):
        heirarchy_community[0][i] = [(heirarchy_community[0][i])]

    # The heirarchy of communities is obtained

    # The quality factor
    if len(source)-2 > 0:
        phi = sum([k[1] for k in phi_list[:-1]])/sum([k[2] for k in phi_list[:-1]])
    else:
        phi = 1

    ecs = [k[0] for k in epsilon_community_size]

    print('##  [Detection Results]  ##')
    print('---------------------------')
    # Detection info #
    print("Graph:" + graphname)
    print('Num of nodes:'+ str(len(G.nodes)))
    print('Num of edges:'+ str(len(G.edges)))   
    print('Num of sources:'+ str(len(source)))
    print('Num of sinks:'+ str(len(sink)))
    print('Num of isolated nodes:'+ str(len(iso)))
    print('Num of leaf nodes:'+ str(leaf))  
    print('Num of inner members:'+ str(inner))  
    print('Num of crossovers:'+ str(r_crossover))
    print('Propagation time steps:' + str(t))
    print('Average end-membership of each node:' + str(ave_membership))
    print('Average size of each end-community:' + str(ave_size))
    print('Epsilon Max:' + str(epsilon_max))
    print('Phi(J-D consistency factor):' + str(phi))
    print('Phi(J-D consistency factor)-ref:' + str(phi-comb(len(source)-2,all_jd_count)*(0.5**(len(source)-2))))
    print('Epsilon_Community_size:\n')
    print(epsilon_community_size)
    print('\n')
    print('Phi-Epsilon:\n')
    print([k[0] for k in phi_list])
    print('Phi-Epsilon-ref:\n')
    print(phi_ref_list)
    print((sum([phi_ref_list[i]*phi_list[i][2] for i in range(len(phi_list)-1)]))/(len(source)-1-phi_list[-1][2]))
    print('\n')
    print('Time complexity:\n')
    print(time_record)
    
    # Save Output Results #

    #1
    text_file = open(output_file+graphname+"_History.txt", "w")

    for i in range(len(History)):
        text_file.write(str(History[i])+'\n')

    text_file.close()
    #2
    text_file = open(output_file+graphname+"_End-community.txt", "w")

    for i in range(len(community)):
        text_file.write(str(community[i])+'\n')

    text_file.close()
    #3
    text_file = open(output_file+graphname+"_Hierarchy-community.txt", "w")

    for i in range(len(heirarchy_community)):
        text_file.write(str('####Epsilon:') + str(i) + '---' + str(heirarchy_community[i])+'\n')

    text_file.close()
    #4
    text_file = open(output_file+graphname+"_D_record.txt", "w")

    for i in range(len(d_record)):
        text_file.write(str('####Epsilon:') + str(i) + '---' + str(d_record[i])+'\n')

    text_file.close()
    #5
    text_file = open(output_file+graphname+"_jac_record.txt", "w")

    for i in range(len(jac_record)):
        text_file.write(str('####Epsilon:') + str(i) + '---' + str(jac_record[i])+'\n')

    text_file.close()

    print('##  Results Saved ##')
    print('---------------------------')
