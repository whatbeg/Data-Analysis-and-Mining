# -*- coding: cp936 -*-
import urllib2
import re
from BeautifulSoup import BeautifulSoup

f = open('howtoTucao.txt','w')     #open the file

for pagenum in range(1,21):      

    strpagenum = str(pagenum)
    print "Getting data for Page " + strpagenum   #for we can see the process in shell
    url = "http://www.zhihu.com/collection/27109279?page="+strpagenum
    page = urllib2.urlopen(url)     #get the web page
    soup = BeautifulSoup(page)      #use BeautifulSoup to parsing the web page
    
    ALL = soup.findAll(attrs = {'class' : ['zm-item-title','content hidden'] })

    for each in ALL :
        if each.name == 'h2' :
            nowstring = re.sub('<s.+>\n<a.+>\n<.+>\n','',each.a.string)
            nowstring = re.sub('&lt;br&gt;','\n',nowstring)
            nowstring = re.sub('&lt;\w+&gt;','',nowstring)
            nowstring = re.sub('&lt;/\w+&gt;','',nowstring)
            nowstring = re.sub('&lt;.+&gt;','\n图片\n',nowstring)
            nowstring = re.sub('&quot;','"',nowstring)
            print nowstring
            if nowstring:
                f.write(nowstring)
            else :
                f.write("\n No Answer \n")
        else :
            nowstring = re.sub('<s.+>\n<a.+>\n<.+>\n','',each.string)
            nowstring = re.sub('&lt;br&gt;','\n',nowstring)
            nowstring = re.sub('&lt;\w+&gt;','',nowstring)
            nowstring = re.sub('&lt;/\w+&gt;','',nowstring)
            nowstring = re.sub('&lt;.+&gt;','\n图片\n',nowstring)
            nowstring = re.sub('&quot;','"',nowstring)
            print nowstring
            if nowstring:
                f.write(nowstring)
            else :
                f.write("\n No Answer \n")
f.close()                                 #close the file
