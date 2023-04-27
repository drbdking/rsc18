from _operator import itemgetter
import gc
from math import sqrt, log10
import math
import os
import pickle
import random
import time
from scipy.sparse import *
from scipy.sparse.linalg import norm

import psutil

from nltk import tokenize as tokenise, stem
import numpy as np
import pandas as pd


class SessionKNN: 
    '''
    SessionKNN(k, sample_size=1000, sampling='recent', similarity='cosine', title_boost=0, seq_weighting=None, idf_weight=None, pop_weight=False, pop_boost=0, artist_boost=0, remind=False, sim_cap=0, normalize=True, neighbor_decay=0, session_key = 'playlist_id', item_key= 'track_id', time_key= 'pos', folder=None, return_num_preds=500 )

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__( self, k, sample_size=1000, sampling='recent', similarity='cosine', title_boost=0, seq_weighting=None, idf_weight=None, pop_weight=False, pop_boost=0, artist_boost=0, remind=False, sim_cap=0, normalize=True, neighbor_decay=0, session_key = 'playlist_id', item_key= 'track_id', time_key= 'pos', folder=None, return_num_preds=500 ):
       
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity
        self.pop_boost = pop_boost
        self.artist_boost = artist_boost
        self.title_boost = title_boost
        self.seq_weighting = seq_weighting
        self.idf_weight = idf_weight
        self.pop_weight = pop_weight
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.remind = remind
        self.normalize = normalize
        self.sim_cap = sim_cap
        self.neighbor_decay = neighbor_decay
        self.return_num_preds = return_num_preds
        
        #updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.playlist_track_matrix = None
        self.track_playlist_matrix = None
        self.num_tracks = 0
        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.session_time = dict()
        self.folder = folder
        
        self.sim_time = 0
        
    def train(self, data, test=None):
        '''
        Training process. Use existing item-session map and session-item map or build from scratch. 
        Both dict or dict-like, item-session map key: item, value: {sessions involving item},
        session-item map key: session, value: {items in the session}. Calculate pop, IDF, title_boost.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        
        interactions = data['actions']
        playlists = data['playlists']
        tracks = data['tracks']
        num_playlists = playlists.shape[0]
        num_tracks = tracks.shape[0]
        self.num_tracks = num_tracks
        
        start = time.time()
        
        playlist_index = interactions[self.session_key].values
        track_index = interactions[self.item_key].values - 1 

        data = np.ones(interactions.shape[0])

        self.playlist_track_matrix = csr_matrix((data, (playlist_index, track_index)), shape=(num_playlists, num_tracks))
        self.track_playlist_matrix = csr_matrix((data, (track_index, playlist_index)), shape=(num_tracks, num_playlists))
        
        end = time.time()
        print('Matrix build time: ', end - start)

        self.item_pop = pd.DataFrame()
        # num of sessions involving item
        self.item_pop['pop'] = interactions.groupby(self.item_key).size()
        # percentage
        self.item_pop['pop'] = self.item_pop['pop'] / len(interactions)
        self.item_pop = self.item_pop['pop'].to_dict()
        
        start = time.time()
        # IDF: np.matrix
        if self.idf_weight != None:
            self.idf = np.log(num_playlists / self.playlist_track_matrix.sum(axis=0))

        end  = time.time()
        print('IDF calculate time: ', end - start)
                        
        self.tall = 0
        self.tneighbors = 0
        self.tscore = 0
        self.tartist = 0
        self.tformat = 0
        self.tsort = 0
        self.tnorm = 0
        self.thead = 0
        self.count = 0
                
    def predict( self, playlist, tracks, playlist_id=None, artists=None, num_hidden=None ):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        playlist : string
            The session IDs of the event.
        tracks : np array
            The item ID of the event. Must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
         
        tracks = np.unique(tracks) - 1
        
        if len(tracks) == 0: 
            res_dict = {'track_id': [], 'confidence': []}
            res = pd.DataFrame.from_dict(res_dict)
            return res
                
        start = time.time()
        neighbors, similarity = self.find_neighbors(tracks)
        end = time.time()
        n_time = end - start
        

        start = time.time()
        candidates, scores = self.score_items(neighbors, similarity)
        sim_sum = np.sum(similarity)

        end = time.time()
        s_time = end - start

        # Create things in the format
        # if self.normalize:
        #     scores = scores / sim_sum

        res_dict = {'track_id': candidates, 'confidence': scores}
        res = pd.DataFrame.from_dict(res_dict)       

        res.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
        res = res.reset_index(drop=True)

        res = res.head(self.return_num_preds)
                
        self.count += 1
                
        return res, n_time, s_time


    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second )
        res = intersection / union
        
        self.sim_time += (time.clock() - sc)
        
        return res 
    
    def cosine_for_set(self, first, second):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / ( sqrt(la) * sqrt(lb) )

        return result
    
    def cosine(self, element_1, element_2):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        element_1: np matrix
        element_2: np array
        
        Returns 
        --------
        out : float value           
        '''
        n1 = norm(element_1)
        n2 = np.sqrt((element_2 ** 2).sum())
        result = element_1.dot(element_2) / (n1 * n2)
        return result[0]
    
    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / ( la + lb -li )

        return result
    
    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = len(first&second)
        b = len(first)
        c = len(second)
        
        result = (2 * a) / ((2 * a) + b + c)

        return result
    
    def random(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        return random.random()
    

    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session)
    
    
    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map.get( item_id )
        
        
    def most_recent_sessions( self, sessions, number ):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get( session )
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))
            
        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        #print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add( element[0] )
        #print 'returning sample of size ', len(sample)
        return sample
        
        
    def possible_neighbor_sessions(self, tracks):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        
        Parameters
        --------
        tracks: np array (track ids)
        
        Returns 
        --------
        relevant_playlists : np array (playlist ids)           
        '''
        
        _, relevant_playlists = self.track_playlist_matrix[tracks].sum(axis=0).nonzero()

        if self.sample_size == 0: #use all session as possible neighbors
            #print('!!!!! runnig KNN without a sample size (check config)')
            return relevant_playlists
        else: #sample some sessions                 
            if len(relevant_playlists) > self.sample_size:
                # if self.sampling == 'recent':
                #     sample = self.most_recent_sessions( relevant_playlists, self.sample_size )
                if self.sampling == 'random':
                    sample = np.random.choice(relevant_playlists, self.sample_size, replace=False)
                else:
                    sample = relevant_playlists[:self.sample_size]
                return sample
            else: 
                return relevant_playlists
                        

    def calc_similarity(self, tracks, possible_neighbors):
        '''
        Calculates the configured similarity for the items in tracks and each session in possible_neighbors.
        
        Parameters
        --------
        tracks: np array (track ids)
        possible_neighbors: np array (palylist ids)
        
        Returns 
        --------
        possible_neighbors: np array  
        similarity: np array
        '''
    
        threshold = 0      
        # similarity = []
        row = tracks
        col = np.zeros(len(tracks))
        data = np.ones(len(tracks))
        tracks_vector = csr_matrix((data, (row, col)), shape=(self.num_tracks, 1))
        neighbor_playlists = self.playlist_track_matrix[possible_neighbors]
        inverse_norm_playlist = 1 / norm(neighbor_playlists, axis=1).reshape(-1, 1)  
        similarity = (neighbor_playlists.dot(tracks_vector)).multiply(inverse_norm_playlist) / norm(tracks_vector)
        similarity = similarity.toarray().reshape(-1)

        # for p in neighbor_playlists:
        #     similarity.append(getattr(self, self.similarity)(p, tracks)) 

        # similarity = np.array(similarity)
        indices = (similarity > threshold).nonzero()
        similarity = similarity[indices]
        possible_neighbors = possible_neighbors[indices]      

        return possible_neighbors, similarity


    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def find_neighbors(self, tracks):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        tracks: np.array
        
        Returns 
        --------
        possible_neighbors : np.array
        similarity: np.array         
        '''
        # based on tracks in the playlist give out possible neighbors
        possible_neighbors = self.possible_neighbor_sessions(tracks)
        possible_neighbors, similarity = self.calc_similarity(tracks, possible_neighbors)
        
        # descending sort
        sort_index = np.argsort(-similarity)
        possible_neighbors = possible_neighbors[sort_index][:self.k]
        similarity = similarity[sort_index][:self.k]

        return possible_neighbors, similarity
    
            
    def score_items(self, neighbors, similarity):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: np.array of playlist ids
        similarity: np.array 
        Returns 
        --------
        score_vector : np.array       
        '''
        # idf weights
        score_matrix = csr_matrix.multiply(self.playlist_track_matrix[neighbors], self.idf)
        # similarity weights
        similarity = similarity.reshape(-1, 1)
        score_matrix = csr_matrix.multiply(score_matrix, similarity)
        score_vector = np.asarray(score_matrix.sum(axis=0)).reshape(-1)
        candidate_index = score_vector.nonzero()[0]
        score = score_vector[candidate_index]
        return candidate_index, score
    
    def linear(self, i):
        return 1 - (0.1*i) if i <= 100 else 0
    
    def same(self, i):
        return 1
    
    def div(self, i):
        return 1/i  
    
    def log(self, i):
        return 1/(log10(i+1.7))
    
    def quadratic(self, i):
        return 1/(i*i)
    
    def normalise(self, s, tokenize=True, stemm=True):
        if tokenize:
            words = tokenise.wordpunct_tokenize(s.lower().strip())
        else:
            words = s.lower().strip().split( ' ' )
        if stemm:
            return ' '.join([self.stemmer.stem(w) for w in words])
        else:
            return ' '.join([w for w in words])
        