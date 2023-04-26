# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015
@author: Balázs Hidasi
Based on https://github.com/hidasib/GRU4Rec
Extended to suit the framework
"""

import numpy as np
import pandas as pd
import pickle
import os
import time
from math import log10, pow
from collections import defaultdict


class ItemKNN:
    '''
    ItemKNN(n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time')
    
    Item-to-item predictor that computes the the similarity to all items to the given item.
    
    Similarity of two items is given by:
    
    .. math::
        s_{i,j}=\sum_{s}I\{(s,i)\in D & (s,j)\in D\} / (supp_i+\\lambda)^{\\alpha}(supp_j+\\lambda)^{1-\\alpha}
        
    Parameters
    --------
    n_sims : int
        Only give back non-zero scores to the N most similar items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    lmbd : float
        Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)
    alpha : float
        Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')
    
    '''    
    
    def __init__(self, n_sims = 1500, lmbd = 20, alpha = 0.5, steps=100, remind=False, weighting='same', 
                 idf_weight=None, pop_weight=None, session_key = 'playlist_id', item_key = 'track_id', 
                 time_key = 'pos', folder=None, return_num_preds=500 ):

        self.n_sims = n_sims
        self.lmbd = lmbd
        self.alpha = alpha
        self.steps = steps
        self.weighting = weighting
        self.idf_weight = idf_weight
        self.pop_weight = pop_weight
        self.remind = remind
        self.item_key = item_key
        self.session_key = session_key
        self.time_key = time_key
        self.folder = folder
        self.return_num_preds = return_num_preds

        self.item_session_map = defaultdict(set)
        self.session_item_map = defaultdict(set)
        # self.session_time = dict()



    def train(self, data, test=None):
        


        train = data['actions']
        test = test['actions']
        test_items = set(test[self.item_key].unique())
        folder = self.folder
        itemids = train[self.item_key].unique()
        

        name = folder + 'iknn_sims.pkl'
        if self.alpha != 0.5:
            name += '.'+str(self.alpha)
        if self.lmbd != 20:
            name += '.'+str(self.lmbd)

        if folder is not None and os.path.isfile( folder + 'item_session_map.pkl'):
            self.session_item_map = pickle.load( open( folder + 'session_item_map.pkl', 'rb') )
            self.session_time = pickle.load( open( folder + 'session_time.pkl', 'rb' ) )
            self.item_session_map = pickle.load( open( folder + 'item_session_map.pkl', 'rb' ) )
        else:
            col_playlist = train.columns.get_loc(self.session_key)
            col_track = train.columns.get_loc(self.item_key)

        for row in train.itertuples(index = False):
            self.item_session_map[row[col_track]].add(row[col_playlist])
            self.session_item_map[row[col_playlist]].add(row[col_track])

        playlists_list = self.session_item_map.values()


        # def coexists_count(lst, item):
        #     res = {}
        #     for s in lst:
        #         if item in s:
        #             for key in s:
        #                 if key != item:
        #                     if key not in res:
        #                         res[key] = 1
        #                     else:
        #                         res[key] += 1
        #     return res


        cnt = 0
        tstart = time.time()
        self.sims = dict()
        print(len(self.item_session_map))
        for itemi in self.item_session_map:

            if itemids[itemi] not in test_items:
                continue
            
            iarray_mapping = defaultdict(int)
            n = len(self.item_session_map[itemi])
            c = 0
            for itemj in self.item_session_map:
                c += 1
                print(c)
                intersection = self.item_session_map[itemi].intersection(self.item_session_map[itemj])
                if itemi == itemj:
                    iarray_mapping[itemj] = len(intersection)
                    continue
                m = len(self.item_session_map[itemj])
                iarray_mapping[itemj] = len(intersection) / (pow((n + self.lmbd), self.alpha) * pow((m + self.lmbd), 1 - self.alpha))
            list_keys = [[val, idx] for idx, val in enumerate(list(self.item_session_map.keys()))]
            indices = [x[0] for x in sorted(list_keys, key = lambda x: iarray_mapping[x[1]], reverse = True)[:self.n_sims]]
            filtered_dict = {k: v for k, v in iarray_mapping.items() if k in indices}


            # dic_i = coexists_count(playlists_list, itemi)
            # self.sims[itemids[itemi]] = pd.Series(dic_i)

            

            cnt += 1
            if cnt % 1000 == 0:
                print( ' -- finished {} of {} items in {}s'.format( cnt, len(test_items), (time.time() - tstart) ) )

            if folder is not None:
                pickle.dump( self.sims, open( name, 'wb') )
        
        if self.idf_weight != None:
            self.idf = pd.DataFrame()
            self.idf['idf'] = data.groupby( self.item_key ).size()
            self.idf['idf'] = np.log( data[self.session_key].nunique() / self.idf['idf'] )
            self.idf['idf'] = ( self.idf['idf'] - self.idf['idf'].min() ) / ( self.idf['idf'].max() - self.idf['idf'].min() )
            self.idf = pd.Series( index=self.idf.index, data=self.idf['idf']  )
            
        if self.pop_weight != None:
            self.pop = pd.DataFrame()
            self.pop['pop'] = data.groupby( self.item_key ).size()
            self.pop['pop'] = ( self.pop['pop'] - self.pop['pop'].min() ) / ( self.pop['pop'].max() - self.pop['pop'].min() )
            self.pop = pd.Series( index=self.pop.index, data=self.pop['pop']  )
        
            
    def predict( self, name=None, tracks=None, artists=None, playlist_id=None, num_hidden=None ):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        name : int or string
            The session IDs of the event.
        tracks : int list
            The item ID of the event. Must be in the set of item IDs of the training set.
            
        Returns
        --------
        res : pandas.DataFrame
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
        items = tracks if tracks is not None else []      
        
        sim_list = pd.Series()
        if len(items) > 0 and items[-1] in self.sims:
            sim_list = self.sims[items[-1]]
        
        # Create things in the format
        res_dict = {}
        res_dict['track_id'] =  sim_list.index
        res_dict['confidence'] = sim_list
        if len(items) > 0 and self.idf_weight != None:
            res_dict['confidence'] = res_dict['confidence'] * self.idf[ items[-1] ]
        res = pd.DataFrame.from_dict(res_dict)
        
        if self.steps is not None:
            
            if len( items ) < 10:
                self.step_size = 0.1
            else:
                self.step_size = 1.0 / (len( items ) + 1)
            
            for i in range( self.steps ):
                
                if len( items ) >= i + 2:
                    prev = items[ -(i+2) ]
                    
                    if prev not in self.sims:
                        continue
                    
                    sim_list = self.sims[prev]
                    
                    res = res.merge( sim_list.to_frame('tmp'), how="left", left_on='track_id', right_index=True )
                    if self.idf_weight != None:
                        res['tmp'] = res['tmp'] * self.idf[ prev ]
                    if self.pop_weight != None:
                        res['tmp'] = res['tmp'] * self.pop[ prev ] # * (1 - self.pop[ prev ])
                    res['confidence'] += getattr(self, self.weighting)( res['tmp'].fillna(0), i + 2 )
                    
                    #res['confidence'] += res['tmp'].fillna(0)
                    del res['tmp']
                    
                    mask = ~np.in1d( sim_list.index, res['track_id'] )
                    if mask.sum() > 0:
                        res_add = {}
                        res_add['track_id'] =  sim_list[mask].index
                        if self.idf_weight != None:
                            res_add['confidence'] = sim_list[mask] * self.idf[ prev ]
                        else:
                            res_add['confidence'] = sim_list[mask]
                        res_add['confidence'] = getattr(self, self.weighting)( res_add['confidence'], i + 2 )
                        #res_add['confidence'] = sim_list[mask]
                        res_add = pd.DataFrame.from_dict(res_add)
                        res = pd.concat( [ res, res_add ] )
        
        if not self.remind:
            res = res[ np.in1d( res.track_id, items, invert=True ) ]
        
        res.sort_values( 'confidence', ascending=False, inplace=True )
        
        #if self.normalize:
        #    res['confidence'] = res['confidence'] / res['confidence'].sum()
        
        res=res.head(self.return_num_preds) 
        
        return res
    
    
    def same(self, confidences, step):
        return confidences
    
    def div(self, confidences, step):
        return confidences / step
    
    def log(self, confidences, step):
        return confidences/(log10(step+1.7))
    
    def linear(self, confidences, step):
        return confidences * (1 - (self.step_size * step))
    
    def set_return_num_preds(self, num):
        self.return_num_preds = num
