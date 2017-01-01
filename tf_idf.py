# -*- coding: utf-8 -*-
import urllib2
import re
import sys
import MySQLdb
import chardet
from chardet.universaldetector import UniversalDetector
import errno
import json
import StringIO
import logging
from gensim import corpora, models, similarities
reload(sys)
sys.setdefaultencoding('utf-8')


"""
计算站点画像的tf-idf文档对象模型：：
1、
2、
3、

Args:
    url:需要提取的站点首页url，从Community表中读

Return:
    站点标签(5个)，存储在数据库Tags_simple中。

Created on 20170101
@author:
"""



'''
处理数据库
'''
db_host = '192.168.1.103'
db_username = 'root'
db_password = 'mysql'
db_database_name = 'Freebuf_Secpulse'
db_table_name = 'Tags_simple'

'''
最大查询条目数
'''
record_limit = 100


def getMysqlConn():
	return MySQLdb.connect(host = db_host,user = db_username,passwd = db_password,db = db_database_name,charset = "utf8")


def loadSiteTagSql(start_id = 0):
	return "select id, siteID,siteDomain,tags from " + db_table_name + " where id > " + str(start_id)  + ' order by id asc ' +  " limit "   + str(record_limit)



def loadSiteTagList():
	'''
	读数据库Tags_simple中的tag内容，以字典方式传递
	'''
	conn=getMysqlConn()
	cur=conn.cursor()
	start_id = 0

	select_sql = loadSiteTagSql(start_id)
	cur.execute(select_sql)
	siteTagList = cur.fetchall()
	allSiteTagList = []
	for tag in siteTagList:
		allSiteTagList.append(tag)

	while len(siteTagList) >= record_limit:
		start_id = siteTagList[len(siteTagList) - 1][0]
		select_sql = loadSiteTagSql(start_id)
		cur.execute(select_sql)
		siteTagList = cur.fetchall()
		for tag in siteTagList:
			allSiteTagList.append(tag)


	item_site_tags = []
	for tag in allSiteTagList:
		item = {}
		item['id'] = tag[0]
		item['siteID'] = tag[1]
		item['siteDomain'] = tag[2]
		item['tags'] = tag[3]

		item_site_tags.append(item)


	return item_site_tags


def convertItemTagToDoc(item_site_tags = {}):
	'''
	字典存储sitDomain和tags
	'''
	urlDocDic = {}
	for item in item_site_tags:
		doc = itemTagToDoc(item['tags'])
		key = item['siteDomain']
		urlDocDic[key] = doc


	return urlDocDic



def itemTagToDoc(tag = ''):
	tagDic = json.loads(tag)
	doc = ''
	for (k,v) in tagDic.items():
		while v > 0:
			doc = doc +" " + k
			v = v - 1

	return doc


def urlDocToTfidf(urlDoc = {}):
	class MyDoc(object):
		def __iter__(self):
			for (url,doc) in urlDoc.items():
				yield doc.split()


	myDoc = MyDoc()
	#分词
	dictionary = corpora.Dictionary(myDoc)

    ##把所有评论转化为词包（bag of words）
	corpus = [dictionary.doc2bow(text) for text in myDoc]

    #使用tf-idf 模型得出该评论集的tf-idf 模型
	tfidf = models.TfidfModel(corpus)

    #词包（corpus） 出所有评论的tf-idf 值
	corpus_tfidf = tfidf[corpus]




if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


	item_site_tags =loadSiteTagList()
	urlDoc = convertItemTagToDoc(item_site_tags)
	for (k,v) in urlDoc.items():
		print '========'
		print v
	#urlDocToTfidf(urlDoc)


	 #for item in item_site_tags:
	 	#print item['tags']





