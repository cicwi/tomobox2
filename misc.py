#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

Just usefull stuff...

"""

class subclass(object):
    def __init__(self, parent):
        self._parent = parent
        

def progress_bar(progress):
    """
    Plot progress in pseudographics:
    """
    
    txt = '\r |'
    
    for ii in range(20):
        if ii / 20 <= progress:
            txt = txt + '\u2588'
        else:
            txt = txt + ' '
            
    txt = txt + '|'        
    
    print(txt, end = " ") 
    
    if progress == 1:
        print('\r')