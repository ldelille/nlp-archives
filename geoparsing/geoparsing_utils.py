#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:11:40 2021

@author: eva
"""
####################################
### Geoparsing utility functions ###
####################################

## Import data ##

import pickle

with open('country_name_to_ref.pickle', 'rb') as handle:
    country_name_to_ref = pickle.load(handle)
with open('my_continents_fr.pickle', 'rb') as handle:
    my_continents_fr = pickle.load(handle)
with open('country_to_cont_dic.pickle', 'rb') as handle:
    country_to_cont_dic = pickle.load(handle)
with open('my_space_voc.pickle', 'rb') as handle:
    my_space_voc = pickle.load(handle)
with open('list_continents_fr.pickle', 'rb') as handle:
    list_continents_fr = pickle.load(handle)
with open('list_countries_fr.pickle', 'rb') as handle:
    list_countries_fr = pickle.load(handle)
with open('my_continent_codes.pickle', 'rb') as handle:
    my_continent_codes = pickle.load(handle)

my_ref_zones = ['afrique', 'l\'afrique', 'asie', 'l\'asie', 'europe', 'l\'europe', 
                'l\'amérique du nord', 'amérique du nord', 'amérique', 'l\'amérique', 
                'océanie', 'l\'océanie', 'l\'amérique du sud', 'l\'amérique latine', 'l\'antarctique',
                'amérique du sud', 'amérique latine', 'antarctique', 'moyen-orient', 'le moyen-orient', "l'espace"]

cont_names_to_ref = {'afrique':'afrique', 
                     'l\'afrique':'afrique', 
                     'asie':'asie', 
                     'l\'asie':'asie', 
                     'europe':'europe', 
                     'l\'europe':'europe', 
                     'l\'amérique du nord':'amérique du nord', 
                     'amérique du nord':'amérique du nord', 
                     'amérique':'amérique', 
                     'l\'amérique':'amérique', 
                     'océanie':'océanie', 
                     'l\'océanie':'océanie', 
                     'l\'amérique du sud':'amérique du sud', 
                     'l\'amérique latine':'amérique latine', 
                     'l\'antarctique':'antarctique',
                     'amérique du sud':'amérique du sud', 
                     'amérique latine':'amérique latine', 
                     'antarctique':'antarctique', 
                     'moyen-orient':'moyen-orient', 
                     'le moyen-orient':'moyen-orient', 
                     "l'espace":"espace", 
                     "espace":"espace"}


## Import packages ## 
import geonamescache
from geonamescache.mappers import country
gc = geonamescache.GeonamesCache()
continents = gc.get_continents()
countries = gc.get_countries()

import geopy
import pandas as pd

import spacy
try: 
    print("fr_core_news_sm loaded")
    nlp = spacy.load("fr_core_news_sm") # load pre-trained models for French
except:
    print("fr loaded")
    nlp=spacy.load('fr') # fr calls fr_core_news_sm 
    
from spacy.lang.fr import French
parser = French()

### --- Utility functions for geographical resolution --- ###
    
iso_to_cont_mapper = country(from_key='iso', to_key='continentcode')


def continent_only(input_list, ref = my_ref_zones, ref_space = my_space_voc):
    cont_list=set()
    for string in list(set(input_list)):
        if string in ref:
            cont_list.add(cont_names_to_ref[string])
        elif string in ref_space:
            cont_list.add("espace")
            cont_list.add(string)
        else:
            pass
    return cont_list

def country_only(input_list, ref = list_countries_fr):
    country_list=set()
    for string in set(input_list):
        if string in ref:
            try : 
                country_list.add(country_name_to_ref[string])
            except: 
                country_list.add(string)
        else:
            pass
    return country_list


def city_info(city_list):
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut    
    geopy.geocoders.options.default_user_agent = "my-application2"
    geolocator = Nominatim(timeout=2)

    loc_list=[]
    loc_dic = {key:[] for key in ['cont', 'country', 'country_code', 'state', 'city', 'misc']}

    for city in city_list : 
        print('\n', city)
        try:
            location = geolocator.geocode(city, addressdetails=True, language="fr")
            if location:
                print(location.latitude, location.longitude)
                loc_list.append(location)
                
                address = location.raw['address'] 
                print(address)
                
                if len(set(['shop', 'amenity', 'building', 'neighbourhood', 'leisure', 'hamlet', 'locality', 'isolated_dwelling'])&address.keys())>0:
                    pass     # discard the entity because of high probability of mistake           
                  
                elif 'city' in address.keys(): # city, or road in a city
                    loc_dic['misc'].append(address.get('tourism', '').lower() )
                    loc_dic['misc'].append(address.get('road', '').lower() )  
                    loc_dic['city'].append(address.get('city', '').lower() )
                    loc_dic['state'].append(address.get('state', '') .lower() )
                    loc_dic['country'].append(address.get('country', '').lower()  )
                    cc = address.get('country_code', '').upper()
                    loc_dic['country_code'].append(cc)
                    loc_dic['cont'].append(my_continent_codes[iso_to_cont_mapper(cc)].lower() )
                elif 'town' in address.keys(): # town, or road in a town 
                    loc_dic['misc'].append(address.get('tourism', '').lower() )
                    loc_dic['misc'].append(address.get('road', '').lower() )  
                    loc_dic['city'].append(address.get('town', '').lower()  )                                   
                    loc_dic['state'].append(address.get('state', '') .lower() )
                    loc_dic['country'].append(address.get('country', '').lower()  )
                    cc = address.get('country_code', '').upper()
                    loc_dic['country_code'].append(cc)
                    loc_dic['cont'].append(my_continent_codes[iso_to_cont_mapper(cc)].lower() )
                elif 'place' in address.keys(): # ! can be a continent
                    loc_dic['state'].append(address.get('state', '') .lower() )
                    loc_dic['country'].append(address.get('country', '').lower()  )
                    cc = address.get('country_code', '').upper()
                    loc_dic['country_code'].append(cc)
                    try :
                        loc_dic['cont'].append(my_continent_codes[iso_to_cont_mapper(cc)].lower() ) 
                        loc_dic['state'].append(address.get('place', '').lower())
                    except:
                        pass
                elif 'region' in address.keys():
                    loc_dic['misc'].append(address.get('road', '').lower() )  
                    loc_dic['state'].append(address.get('region', '').lower())
                    loc_dic['state'].append(address.get('state', '') .lower() )
                    loc_dic['country'].append(address.get('country', '').lower()  )
                    cc = address.get('country_code', '').upper()
                    loc_dic['country_code'].append(cc)
                    loc_dic['cont'].append(my_continent_codes[iso_to_cont_mapper(cc)].lower() )                    
                elif 'village' in address.keys():
                    loc_dic['misc'].append(address.get('road', '').lower() )  
                    loc_dic['misc'].append(address.get('tourism', '').lower() )
                    loc_dic['misc'].append(address.get('village', '').lower() )
                    loc_dic['city'].append(address.get('municipality', '').lower()  )
                    loc_dic['state'].append(address.get('state', '') .lower() )
                    loc_dic['country'].append(address.get('country', '').lower()  )
                    cc = address.get('country_code', '').upper()
                    loc_dic['country_code'].append(cc)
                    loc_dic['cont'].append(my_continent_codes[iso_to_cont_mapper(cc)].lower() )                    
                elif 'waterway' in address.keys():
                    loc_dic['misc'].append(address.get('waterway', '').lower() )
                    loc_dic['country'].append(address.get('country', '').lower()  )
                    cc = address.get('country_code', '').upper()
                    loc_dic['country_code'].append(cc)
                    loc_dic['cont'].append(my_continent_codes[iso_to_cont_mapper(cc)].lower() )                                    
                else : # name of a state already, or a "boundary", or a country
                    loc_dic['state'].append(address.get('state', '') .lower() )
                    loc_dic['state'].append(address.get('boundary', '') .lower() ) # region-like zone
                    loc_dic['country'].append(address.get('country', '').lower()  )
                    cc = address.get('country_code', '').upper()
                    loc_dic['country_code'].append(cc)
                    loc_dic['cont'].append(my_continent_codes[iso_to_cont_mapper(cc)].lower() )
                   
        except GeocoderTimedOut as e:
            print("Error: geocode failed on input %s with message %s" %(city, e))
            loc_dic['misc'].append(city.lower())

    loc_dic = {key : set([string for string in loc_dic[key] if len(string)>0]) for key in loc_dic.keys()}
    print('\n---- List of Locations ---')
    print(loc_list)
    print('\n---- Dictionnary ---')
    print(loc_dic)
    return loc_dic

### --- All in one : ontain dico --- ###

def to_lower(input_text):
    if isinstance(input_text, pd.Series):
        output=input_text.apply(lambda x: x.lower())
        pass
    elif isinstance(input_text, list):
        output=[[w.lower() for w in L] for L in input_text]
        pass
    elif isinstance(input_text, str):
        output=input_text.lower()
        pass
    else:
         output = ''   
    return output


def entity_extractor(text_low, ent_type='LOC'):
    """
    Input:
    ------
    text_low : pandas series containing strings to process, or list of lists containing strings. 
    !!! Words must be lower case already
    ent_type : type of entity to extract, LOC by default
    
    Output:
    ------
    ent_list : list of list of extracted entities, same length as the input text_series
    """
    LOC_stopwords = ['etat', 'état', 'pays', 'continent', 'endroit', 'lieu', 'de france', 'état-', 'major', 'état-major']

    def remove_mistakes(input_list, stopwords=LOC_stopwords):
        return [w for w in input_list if w not in stopwords]

    nlp = spacy.load("fr") # reload
    # Create pipe containing all titles
    bodies=list(nlp.pipe(text_low, disable=["tagger", "parser"]) ) 
    
    ent_list = []
    for doc in bodies: 
        ent_list.append([ent.text for ent in doc.ents if ent.label_ == ent_type])

    return remove_mistakes(ent_list)


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            pass
        elif '@' in str(token):
            lda_tokens+=str(token).split('@')
        else:
            lda_tokens.append(token.lower_)
    return [t for t in lda_tokens if len(str(t))>1]


def geo_info(entity_list, ref_cont = my_ref_zones, ref_countries = list_countries_fr): # to add : , **kwargs
    cont_set = continent_only(entity_list, ref=ref_cont)
    country_set = country_only(entity_list, ref=ref_countries)
    for c in country_set:
        cont_set.add(country_to_cont_dic[c])
    city_list = list(set(entity_list)-(cont_set|country_set))
    geo_dico = city_info(city_list)
    geo_dico['cont']=geo_dico['cont']|cont_set
    geo_dico['country']=geo_dico['country']|country_set
    return geo_dico

def geoparsing_str(input_text):
    """
    input_text : article body as 1 string 
    """
    # preprocessing
    split_text = tokenize(input_text)
    # first run with default voc : requires a finer preprocessing --> preprocesssed text as input
    countries = country_only(split_text)
    cont = continent_only(split_text)
    # extract entities
    LOC_list = entity_extractor(pd.Series([input_text]))
    print(LOC_list)
    LOC_list = LOC_list[0]
    # get hierarchical geo info
    geo_dico = geo_info(LOC_list)
    geo_dico["cont"]=geo_dico["cont"]|cont
    geo_dico["country"]=geo_dico["country"]|countries
    return geo_dico


### --- Compararison of Geo Dicts --- ###

def jaccard_sim(set1, set2):
    return len(set1&set2)/len(set1|set2)

def dice_sim(set1, set2):
    return 2*len(set1&set2)/(len(set1)+len(set2))

def geo_sim(dico1, dico2, weights = {'cont':1, 'country':2, 'state':3, 'city':6, 'misc':1}, similarity=dice_sim):
    """
    Input:
    ------
    dico1 : dict with keys 'cont', 'country','country_code', 'state', 'city', 'misc' 
    dico2 : idem
    similarity : str, 'jaccard or 'dice'
    
    Output:
    ------
    sim : float, similarity score for geographic locations contained in the dictionnaries
    """
            
    dico_sim = dict()
    for key in ['cont', 'country', 'state', 'city', 'misc']:
        dico_sim[key]=similarity(dico1[key], dico2[key])
    
    w_sum = sum(list(weights.values()))
    weights = {key : weights[key]/w_sum for key in weights.keys()}
    
    sim=sum([dico_sim[key]*weights[key] for key in dico_sim.keys()])
    
    return sim
           

