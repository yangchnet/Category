# -*- coding: utf-8 -*-
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import os
import urllib
from src.get_text import gettext
os.path.abspath('..')

html_List = [[0 for i in range(0)]for j in range(65)]
str1 = "http://jobs.51job.com/hy"        #http://jobs.51job.com/hy37/p1/
r = 10
while r<=64:
    if r<10:
        str1 = "http://jobs.51job.com/hy0"
    elif r >= 10:
        str1 = "http://jobs.51job.com/hy"
    try:
        for k in range(20):
            html1 = str1 + str(r) + "/" + "p" + str(k+1) + '/'
            html = urlopen(html1)
            bsObj = BeautifulSoup(html)
            for link in bsObj.findAll("a",{"target":"_blank"},{"class":"name"},
                                      href = re.compile("http://jobs.51job.com/[a-z\-]+/co[0-9]+\.html")):
                print(link.attrs['href'])
                html_List[r].append(link.attrs['href'])
            gettext(html_List[r],r)
    except urllib.error.URLError:
        continue
    r += 1
print(html_List)




