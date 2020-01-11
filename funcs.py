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
