# -*- coding: utf-8 -*- 

import urllib2
import re
import sys
import jieba
import jieba.analyse
import MySQLdb
import chardet
from chardet.universaldetector import UniversalDetector
from socket import error as SocketError
import errno
import json
import StringIO
reload(sys)
sys.setdefaultencoding('utf-8')

"""
提取站点首页关键词：
1、获取站点首页源码
2、正则提取源码中的中文
3、分词提取top5存入数据库

Args:
    url:需要提取的站点首页url，从Community表中读

Return:
    站点标签(5个)，存储在数据库Tags_simple中。

Created on 20161201
@author: HU Yi
"""


'''
处理数据库
'''
db_host = '192.168.1.103'
db_username = 'root'
db_password = 'mysql'
db_database_name = 'Freebuf_Secpulse'
db_table_name = 'GrabSite'


'''
提取标签数
'''
topN = 20


'''
url提取位置
'''
start_id = 7003


'''
最大查询条目数
'''
record_limit = 100


def getMysqlConn():
	return MySQLdb.connect(host = db_host,user = db_username,passwd = db_password,db = db_database_name,charset = "utf8")

def insert():
	insert_sql = "insert into Tags_simple(siteID,siteDomain,tags) "+"values(%s,%s,%s)"
	return insert_sql

def getSelectCountSql(id):
    select_sql = "select id,siteDomain from " + db_table_name + " where id > " + str(id) +" order by id asc limit " + str(record_limit);
    return select_sql




def gettips(url):

	"""
	except urllib2.HTTPError, e:
		print e.code	
	except urllib2.URLError, e:
		print e.reason
	except SocketError as e:
		if e.errno == errno.ECONNRESET:
			print "pass socket.error:Connection reset by peer"
	except httplib.HTTPException:
	missing.put(tmpurl)
	continue
	except (httplib.IncompleteRead, urllib2.URLError):
	continue
	"""

	try:
		headers = {'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6' }
		req = urllib2.Request(url, headers = headers) 
		html = urllib2.urlopen(req).read()
	except:
		print "pass error"
		return 0
	
	#coding = chardet.detect(str1)
	detector = UniversalDetector()
	buf = StringIO.StringIO(html)
	for line in buf.readlines():
		#print line
		detector.feed(line)
		if detector.done: 
			break
	detector.close()
	buf.close()
	coding = detector.result



	if coding['encoding']:
		#content = unicode(str1,coding['encoding'])
		content = html.decode(coding['encoding'],'ignore')

		re_words = re.compile(u"[\u4e00-\u9fa5]+")
	  
		res = re.findall(re_words, content)    
		str_convert = ' '.join(res)
	
		#tags = jieba.analyse.extract_tags(str_convert,topK) 
		tags = jieba.cut(str_convert, cut_all=False)
		print "=============="
		print "tags in %s:"% url
		tag = ",".join(tags)
		tag = tag.encode('utf-8')
		tags = tag.split(',')
		return tags

def getTopNjson(tags = []):
	top_tags = {}
	for tag in tags:
		if not top_tags.has_key(tag):
			top_tags[tag] = 0

		top_tags[tag] = top_tags[tag] +1



	top_tags = getTopNTag(top_tags)
	return json.dumps(top_tags)


def getTopNTag(top_tags = {}):
	#if len(top_tags) <= topN:
	#	return top_tags

	sorted(top_tags.items(),key=lambda item:item[1])
	startN =  len(top_tags) - topN
	new_top_new = {}

	for (k,v) in  top_tags.items():
		if startN > 0:
			startN = startN -1
			continue

		new_top_new[k] = v


	return new_top_new
	



if __name__ == "__main__":


	#url = 'https://www.secsilo.com'
	conn=getMysqlConn()
	cur=conn.cursor()


	domainCount = record_limit
	while domainCount >= record_limit :
		siteID_select_sql = 'select max(siteID) from Tags_simple'
		start_id = cur.execute(siteID_select_sql)
		print '===========start_id:%s=============' %(start_id)
		insert_sql = insert()
		select_sql = getSelectCountSql(start_id)
		cur.execute(select_sql)
		domainList = cur.fetchall()
		domainCount = len(domainList)
		print '===========domainCount:%s=============' %(domainCount)
		for domain in domainList:
			tips = gettips(domain[1])
			if tips:
				jsonTag = getTopNjson(tips)
				print '===========url:%s=============' %(domain[1])
				print jsonTag
				item_value = []
				item_value.append(domain[0])
				item_value.append(domain[1])
				item_value.append(jsonTag)
				cur.execute(insert_sql,item_value)
				conn.commit()
	
	print "Finish " 


