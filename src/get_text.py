# -*- coding: utf-8 -*-
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import os
os.path.abspath('..')

def gettext(html_List, r):
    i = 1
    for html0 in html_List:
        html = urlopen(html0)
        bsObj = BeautifulSoup(html)
        text = bsObj.find("div",{"class":"con_msg"})
        # print(text.get_text().encode("utf-8"))
        path = '../data/' + str(r)  + '/' + str(i) +'.txt'

        with open (path,"w", encoding='utf-8') as file_object:
            file_object.write(text.get_text())
            file_object.close()
        i += 1
    return 0