# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:58:57 2020

@author: utente
"""

Commedia= open('Divina_Commedia.txt' ,'r')

Clean= open('Commedia_cleaned.txt','w')



Clean.truncate(0)
for el in Commedia.readlines():

    Clean.write (el.lstrip())