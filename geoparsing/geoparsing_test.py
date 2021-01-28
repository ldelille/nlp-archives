#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 21:38:35 2021

@author: eva
"""

from geoparsing_utils import *

#from spacy.lang.fr import French
#parser = French()

##  >>  Tests: LOC entities extraction and analysis << ##
    

t1 = """Chronique. En moins de vingt ans, Jack Ma a connu une ascension vertigineuse en créant le géant chinois de l’Internet, Alibaba. L’un de ses services, Alipay, permet, d’un simple glissement du pouce sur son téléphone mobile, d’acheter à distance, de demander un prêt, d’investir en Bourse, de payer ses factures ou encore d’appeler un taxi. Sa disgrâce, elle, n’a pris que quelques minutes. Pas à cause de son pouce, mais de sa langue, qu’il n’a pas su tenir.

« Je ne vais pas lâcher de bombe », avait-il pourtant affirmé en préambule d’une conférence internationale à Shanghaï, le 24 octobre 2020. Voire. Quelques instants plus tard, il s’emportait sans filtre contre les banques chinoises accusées d’avoir conservé une « mentalité de prêteur sur gage », susceptible d’étouffer l’innovation dans le pays.

Naïve inconscience ou sentiment total d’impunité de l’un des Chinois les plus riches et les plus influents ? Toujours est-il qu’une semaine plus tard l’homme d’affaires était convoqué par les autorités à propos de l’introduction en Bourse de la filiale d’Alibaba, Ant Financial. Prévue le lendemain, l’opération, d’un montant record, était brutalement suspendue après l’intervention en personne du président chinois, Xi Jinping. Tandis que l’Occident tente de réguler ses géants du Web grâce au droit, la Chine dissuade en faisant des exemples."""

t2 = """Chronique. Après l’invasion du Capitole, le monde éberlué se demande comment le pays qui s’est longtemps présenté comme le leader du monde « libre » a pu tomber aussi bas. Pour comprendre ce qui s’est passé, il est urgent de sortir des mythes et de l’idolâtrie, et de revenir à l’histoire. En réalité, la République étatsunienne est traversée depuis ses débuts par des fragilités, des violences et des inégalités considérables.

Emblème du Sud esclavagiste pendant la guerre civile de 1861-1865, le drapeau confédéré brandi il y a quelques jours par les émeutiers au cœur du Parlement fédéral n’était pas là par hasard. Il renvoie à des conflits très lourds qui doivent être regardés en face.

Le système esclavagiste a joué un rôle central dans le développement des Etats-Unis, comme d’ailleurs du capitalisme industriel occidental dans son ensemble. Sur les quinze présidents qui se sont succédé jusqu’à l’élection de Lincoln en 1860, pas moins de onze étaient propriétaires d’esclaves, dont Washington et Jefferson, tous deux natifs de Virginie, qui en 1790 compte 750 000 habitants (dont 40 % d’esclaves), soit l’équivalent de la population cumulée des deux Etats nordistes les plus peuplés (la Pennsylvanie et le Massachusetts)."""


dico1 , dico2 = geoparsing_str(t1), geoparsing_str(t2)

score = geo_sim(dico1, dico2)

print(dico1)
print(dico2)
print("\nSimilarity score : %.3f" %score)

## >> end tests << ##
