#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:16:12 2021

@author: eva
"""
## Imports ##


import geonamescache
import pandas as pd

import time
start = time.time()

#import pprint
#pp=pprint.PrettyPrinter(indent=4)

## Load data ##
    
# data from opendata.gouv
country_df = pd.read_csv("./liste_197_etats_2020.csv", encoding = "ISO-8859-1", delimiter=';')
country_df['ARTICLE'].fillna("", inplace = True) 
country_df.NOM = country_df.NOM.apply(lambda x : x.lower()) 
country_df.NOM_ALPHA = country_df.NOM.apply(lambda x : x.lower()) 
country_df.NOM_LONG = country_df.NOM.apply(lambda x : x.lower()) 
#print(country_df[:10])


# planets and satellites  from github repo
planet_df = pd.read_csv("./planets.csv")
star_df = pd.read_csv("./star_names.csv", delimiter = ';')
satellite_df = pd.read_csv("./satellites.csv")

## Build references ##

gc = geonamescache.GeonamesCache()
continents = gc.get_continents()
countries = gc.get_countries()

## -- Continents -- ##
# keys = continent names in French
# values = dictionnaries with keys 'continantCode', 'country_codes', 'country_code_iso3', 'country_names'

my_continents_fr = dict() # continents.keys()
my_continent_codes = dict() # basic continent_code : french name

for continent_code in continents.keys():
    for dico in continents[continent_code]['alternateNames']:
        # get the french name ?
        if dico['lang']=='fr':
            name_fr = dico['name']
            cont_dico = {'continentCode':continent_code}
            my_continents_fr[name_fr]=cont_dico
            my_continent_codes[continent_code]=name_fr
            if name_fr!='Antarctique':
                my_continents_fr[name_fr]['country_codes']=continents[continent_code]['cc2'].split(',')
            else:
                my_continents_fr[name_fr]['country_codes']=[]



## -- Countries -- ##

# general continents dictionnary
# map country name to unique country name (ex: la france --> france)
def add_article(article, nom):
    if article in ['le', 'la', 'les']:
        res = article+' '+nom
    elif article == "l'":
        res = article+nom
    else:
        res = nom
    return res

country_name_to_ref = dict()

for key in my_continents_fr.keys():
    my_continents_fr[key]['country_code_iso3']=[]
    my_continents_fr[key]['country_names']=[]
    set_noms_fr = set()
    for country_code in my_continents_fr[key]['country_codes']:
        iso3 = countries[country_code]['iso3']
        my_continents_fr[key]['country_code_iso3'].append(iso3)
        
        try: 
            nom = country_df[country_df['CODE']==iso3]['NOM'].tolist()[0]
            nom_alpha = country_df[country_df['CODE']==iso3]['NOM_ALPHA'].tolist()[0]
            nom_long = country_df[country_df['CODE']==iso3]['NOM_LONG'].tolist()[0]
            article = country_df[country_df['CODE']==iso3]['ARTICLE'].tolist()[0]
            nom_article = add_article(article,nom)
            nom_alpha_article = add_article(article,nom_alpha)
            set_noms_fr.add(nom)
            set_noms_fr.add(nom_alpha)
            set_noms_fr.add(nom_long)
            set_noms_fr.add(nom_article)
            set_noms_fr.add(nom_alpha_article)
            country_name_to_ref[nom]=nom
            country_name_to_ref[nom_alpha]=nom
            country_name_to_ref[nom_article]=nom
            country_name_to_ref[nom_alpha_article]=nom
            country_name_to_ref[nom_long]=nom
        except:
            #print(countries[country_code])
            pass
        set_noms_fr.add(countries[country_code]['name'].lower())
            
    my_continents_fr[key]['country_names']=list(set_noms_fr)

#pp.pprint(my_continents_fr) # to delete - OK

# map country name to continent
country_to_cont_dic = dict()
for cont in my_continents_fr.keys():
    for country_name in my_continents_fr[cont]['country_names']:
        country_to_cont_dic[country_name.lower()]=cont.lower()        
#pp.pprint(country_to_cont_dic) # to delete - OK

# Final lists for continents and countries

list_countries_fr = [] # contains country names in French and in English
for key in my_continents_fr.keys():
    list_countries_fr+=[string.lower() for string in my_continents_fr[key]['country_names']]
#print(list_countries_fr)

list_continents_fr = [key.lower() for key in my_continents_fr.keys()] # continent names in French
#print(list_continents_fr)

#print(my_continent_codes) # basic dic : code : name


## -- Planets and satellites -- ##
planet_names = planet_df.planet.apply(lambda x : x.lower()).tolist()
planet_names += ['mercure', 'terre', 'la terre', 'saturne', 'pluton'] # alright, pluto is not a real planet
#print(planet_names)

star_names = star_df['IAU Name'].apply(lambda x : x.lower()).tolist()
#print(star_names[:10])
# note : we would need the constellation names in plain text too !

satellite_names = satellite_df.name.apply(lambda x : x.lower()).tolist()
#print(satellite_names[:10])

my_space_voc = satellite_names + planet_names + star_names
#print("Number of words in ref list: ", len(my_space_voc))

## -- Zones -- ##

# utility function : from a list of entities, look for continent names and country names
# store them in a dictionnary

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

def continent_info(input_list, ref = my_ref_zones):
    """
    not used
    """
    geo_dic={key:[] for key in ['cont', 'country', 'country_code', 'state', 'city', 'misc']}
    for string in set(input_list):
        if string in ref:
            geo_dic['cont'].append(string)
    return geo_dic

def country_info(input_list, ref = list_countries_fr):
    """
    not_used
    """    
    geo_dic={key:[] for key in ['cont', 'country', 'country_code', 'state', 'city', 'misc']}
    for string in set(input_list):
        if string in ref:
            geo_dic['country'].append(string)
        else:
            pass
    return geo_dic


#print(continent_info(['israël', 'genève', 'new york', 'abou ammar', 'alger', 'israël', 'aviv', 'koweït', 'palestine', 'terre', 'jordanie', 'liban', 'beyrouth', 'syrie', 'israël', 'liban', 'damas', 'jérusalem', 'syrie', 'beyrouth', 'tunis', 'egypte', 'israël', 'syrie', 'damas', 'tripoli', 'iran', 'jordanie', 'egypte', 'palestine', 'israël', 'genève', 'israël', 'rusé', 'habile', 'new york', 'kenya', 'amérique du sud', 'genève', 'israël', 'new york', 'jérusalem', 'diplomatique', 'washington']))
#print(continent_info(['afrique', 'asie', 'afrique']))
#print(country_info(['israël', 'genève', 'new york', 'abou ammar', 'alger', 'israël', 'aviv', 'koweït', 'palestine', 'terre', 'jordanie', 'liban', 'beyrouth', 'syrie', 'israël', 'liban', 'damas', 'jérusalem', 'syrie', 'beyrouth', 'tunis', 'egypte', 'israël', 'syrie', 'damas', 'tripoli', 'iran', 'jordanie', 'egypte', 'palestine', 'israël', 'genève', 'israël', 'rusé', 'habile', 'new york', 'kenya', 'amérique du sud', 'genève', 'israël', 'new york', 'jérusalem', 'diplomatique', 'washington']))
#print(country_info(['afrique', 'asie', 'afrique']))


def continent_only(input_list, ref = my_ref_zones, ref_space = my_space_voc):
    cont_list=set()
    for string in set(input_list):
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

#print(continent_only(['israël', 'genève', 'new york', 'abou ammar', 'alger', 'israël', 'aviv', 'koweït', 'palestine', 'terre', 'jordanie', 'liban', 'beyrouth', 'syrie', 'israël', 'liban', 'damas', 'jérusalem', 'syrie', 'beyrouth', 'tunis', 'egypte', 'israël', 'syrie', 'damas', 'tripoli', 'iran', 'jordanie', 'egypte', 'palestine', 'israël', 'genève', 'israël', 'rusé', 'habile', 'new york', 'kenya', 'amérique du sud', 'genève', 'israël', 'new york', 'jérusalem', 'diplomatique', 'washington']))
#print(continent_only(['mars', 'l\'afrique', 'afrique', 'asie', 'afrique', 'amérique latine']))
#print(country_only(['nouvelle-zélande', 'israël', 'genève', 'new york', 'abou ammar', 'alger', 'israël', 'aviv', 'koweït', 'palestine', 'la terre', 'jordanie', 'liban', 'beyrouth', 'la syrie', 'israël', 'liban', 'damas', 'jérusalem', 'syrie', 'beyrouth', 'tunis', 'egypte', 'israël', 'syrie', 'damas', 'tripoli', 'iran', 'jordanie', 'egypte', 'palestine', 'israël', 'genève', 'israël', 'rusé', 'habile', 'new york', 'kenya', 'amérique du sud', 'genève', 'israël', 'new york', 'jérusalem', 'diplomatique', 'washington']))
#print(country_only(['l\'afrique', 'asie', 'afrique', 'amérique latine']))

print("Elapsed time: %s " %(time.time()-start))


### Saving lists and dics ###

import pickle

with open('country_name_to_ref.pickle', 'wb') as handle:
    pickle.dump(country_name_to_ref, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open('my_continents_fr.pickle', 'wb') as handle:
    pickle.dump(my_continents_fr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('country_to_cont_dic.pickle', 'wb') as handle:
    pickle.dump(country_to_cont_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('my_space_voc.pickle', 'wb') as handle:
    pickle.dump(my_space_voc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('list_continents_fr.pickle', 'wb') as handle:
    pickle.dump(list_continents_fr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('list_countries_fr.pickle', 'wb') as handle:
    pickle.dump(list_countries_fr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('my_continent_codes.pickle', 'wb') as handle:
    pickle.dump(my_continent_codes, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('my_continent_codes.pickle', 'rb') as handle:
    test = pickle.load(handle)

print( my_continent_codes == test)