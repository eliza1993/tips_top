#!usr/bin/python
# -*- coding: utf-8 -*-

import MySQLdb
import json
from gensim.models import Word2Vec
import logging,gensim,os
import httplib, urllib

import cairo
from igraph import *

import time
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

"""
实现连接关联和主题关联结合的louvain算法：
1、读数据库获取链接关系
2、开始louvian算法计算
3、计算ΔQ的时候加入主题相似度
2、将分得的社区结果存储入数据库Community

Args:
    SiteRelation:存储链接关系的表，SiteTopic：存储链接关系的数据库

Return:
    Community：站点归属表

Created on 20161227
@author: HU Yi
"""

'''
数据库
'''
db_host = '127.0.0.1'
db_port = 3307
db_username = 'root'
db_password =  '123456'
db_database_name = 'Freebuf_Secpulse'
db_relation_name = 'SiteRelation'

'''
Merge factor 合并因子
'''

merge_factor = 0.5
cos_similar_limit = 0.5
httpClient = None


'''
 阿尔法 S 和 Q的权重参数


'''
f_a = 1.0

def itemTagToDoc(tag = ''):
    tagDic = json.loads(tag)
    doc = ''
    for (k,v) in tagDic.items():
        while v > 0:
            doc = doc +" " + k
            v = v - 1

    return doc







class PyLouvain:
    '''
    从SiteRelation构建图
    从_path构建图.
    _path: 路径——指向包含边"node_from node_to" （每行一个）的文件

    ####加载初始化站点主题画像####
    '''
    @classmethod
    def from_database(cls,dbname):
        record_limit = 1000
        id = 0
        nodes = {}
        edges = []
        conn = MySQLdb.connect(host=db_host, user=db_username, passwd=db_password, db=db_database_name, port=db_port, charset='utf8')
        cur = conn.cursor()
        sql  = "select id,masterSite,outLinkSite,outLinkCount from " + dbname + " where id > " + str(id) +" order by id asc limit " + str(record_limit);
        cur.execute(sql)
        records = cur.fetchall()
        while records:
            for record in records:
                if not record:
                    break
                nodes[record[1]] = 1
                nodes[record[2]] = 1
                w = int(record[3])
                edges.append(((record[1],record[2]),w))
            #nodes_,edges_ = in_order(nodes,edges)
            #print("%d nodes, %d edges" % (len(nodes_), len(edges_)))
            id += 1000
            sql  = "select id,masterSite,outLinkSite,outLinkCount from " + dbname + " where id > " + str(id) +" order by id asc limit " + str(record_limit);
            cur.execute(sql)
            records = cur.fetchall()


        start_id = 0
        site_tags_sql = "select id,siteDomain,tags from Tags_simple where id > "+ str(start_id)+" order by id asc  limit " + str(record_limit)
        cur.execute(site_tags_sql)
        tag_records = cur.fetchall()
        site_tags = {}
        while tag_records:
            for record in tag_records:
                site_tags[record[1]] = itemTagToDoc(record[2])

            start_id = tag_records[len(tag_records) -1][0]

            site_tags_sql = "select id,siteDomain,tags from Tags_simple where id > "+ str(start_id)+" order by id asc  limit " + str(record_limit)
            cur.execute(site_tags_sql)
            tag_records = cur.fetchall()


        nodes_,edges_,site_tags = in_order(nodes,edges,site_tags)
        return cls(nodes_, edges_,site_tags)



    def __init__(self, nodes, edges,site_tags={}):
        self.nodes = nodes
        self.edges = edges
        self.site_tags = site_tags

        # 预计算 m (网络中所有链路的权重和)
        #       k_i (入射到节点i的链路的权重和)
        self.m = 0
        self.k_i = [0 for n in nodes]
        self.edges_of_node = {}
        self.w = [0 for n in nodes]
        for e in edges:
            self.m += e[1]
            self.k_i[e[0][0]] += e[1]
            self.k_i[e[0][1]] += e[1] # 最初没有自循环
            # 按节点保存边
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)

        # 在O（1）的时间中访问节点的社区
        self.communities = [n for n in nodes]
        #社区主题
        self.site_communities_tags = [n for n in nodes]
        self.actual_partition = []



    '''
        应用Louvain算法.
    '''
    def apply_method(self):
        network = (self.nodes, self.edges)
        best_partition = [[node] for node in network[0]]
        best_q = -1
        i = 1
        while 1:
            print("pass #%d" % i)
            i += 1
            partition = self.first_phase(network) #初始分区
            q = self.compute_modularity(partition)
            k_similar = self.compute_community_similar_variance(partition)
            _tsta_fa = self.tsta_fa(k_similar,q)


            print '====================================='
            print '============k_similar:%s================'%(k_similar)
            print '============= _tsta_fa =============='
            print '============= %s ==============' %(_tsta_fa)
         

            print "q = %s" % q
            partition = [c for c in partition if c]
            #print("%s (%.8f)" % (partition, q))
            # 用分区聚簇初始节点  压缩
            if self.actual_partition:
                actual = []
                for p in partition:
                    part = []
                    for n in p:
                        part.extend(self.actual_partition[n])
                    actual.append(part)
                self.actual_partition = actual
            else:
                self.actual_partition = partition

            #没有变化就退出完成
            if q == best_q:
                break
            network = self.second_phase(network, partition)

            draw_networkx(len(network[0]),network[1])


            exit(1)
            best_partition = partition
            best_q = q
            #print("pass #%d" % i)
            #i += 1
            print "best Q = %s" % (best_q)
        return (self.actual_partition, best_q)

    '''
    计算社区的平均相似度
    1.计算社区站点之间主题相似度
    2.获取平均相似度
    3.
    '''
    def compute_community_similar_variance(self,partition = []):
        community_similar = []
        for index in range(0,len(partition)):
            comm_nodes = partition[index]
            if len(comm_nodes) > 1:
                community_similar.append(self.compute_single_community_variance(comm_nodes))



        k_similar = 0.0
        if len(community_similar) > 0:
            total_similar = 0.0
            for single_similar in community_similar:
                total_similar = total_similar + single_similar

            print '======================='
            print community_similar
            print total_similar
            k_similar = total_similar / len(community_similar)

        return k_similar


    def compute_variance(self,community_similar,node_pair_count,comm_similar_avge):
        variance = 0.0
        for index in range(0,len(community_similar)):
            if community_similar[index] > 0.0:
                tempValue = community_similar[index] - comm_similar_avge
                if tempValue < 0.0:
                    tempValue = 0 - tempValue;
                tempValue = tempValue*tempValue
                variance = variance +  tempValue


        return variance/node_pair_count;





    def compute_single_community_variance(self,comm_nodes=[]):
        node_pair_count = 0
        comm_similar_sum = 0.0
        comm_similar_arr = []
        for index in range(0,len(comm_nodes)):
            index2 = index +1
            #print '==========index:%s len:%s========'%(index,len(comm_nodes))
            while index2 < len(comm_nodes):
                if self.site_tags.has_key(comm_nodes[index]) and self.site_tags.has_key(comm_nodes[index2]):
                    site_similar = self.getCosSimilarity(self.site_tags[comm_nodes[index]],self.site_tags[comm_nodes[index2]])
                    comm_similar_sum = comm_similar_sum + site_similar
                    comm_similar_arr.append(site_similar)
                    
                    node_pair_count = node_pair_count + 1
                index2 = index2+1


        if node_pair_count == 0:
            return 0.0



        comm_similar_avge = comm_similar_sum/node_pair_count
        #print '==============node_pair_count%s comm_similar_avge%s'%(node_pair_count,comm_similar_avge)

        #return comm_similar_avge

        #print '==============node_pair_count%s comm_similar_avge%s'%(node_pair_count,comm_similar_avge)
        #variance = self.compute_variance(comm_similar_arr,node_pair_count,comm_similar_avge)
        return comm_similar_avge








    '''
        计算当前网络的模块度。
        partition：节点列表
    '''
    def compute_modularity(self, partition):
        q = 0.0
        m2 = self.m * 2.0
        for i in range(len(partition)):
            q += self.s_in[i] / m2 - (self.s_tot[i] / m2) ** 2
        return q

    '''
        计算社区_c中具有节点的模块化增益。
         _node：int
         _c：int community
         _k_i_in：从_node到_c中的节点的链接的权重的总和
         k_i_in为什么要乘2 ?????????????????（源代码k_in_in前面有‘2 *’,根据公式认为2*多余，故这里删去）
    '''
    def compute_modularity_gain(self, node, c, k_i_in):
        return 2*k_i_in - self.s_tot[c] * self.k_i[node] / self.m

    '''
        计算社区_c中具有节点的汇聚值（模块化增益+主题相似度）。
         _node：int
         _c：int community
         _k_i_in：从_node到_c中的节点的链接的权重的总和

    '''
    def modularity_gain_similar_topic(self,node,c,k_i_in):
        modularity_gain = 2*k_i_in - self.s_tot[c] * self.k_i[node] / self.m
        similar_topic



    '''
        执行方法的第一阶段。
         _network：（nodes，edges）
    '''
    def first_phase(self, network):
        # 进行初始分区
        best_partition = self.make_initial_partition(network)
        loop_count = 0
        while 1:
            print '=========== loop_count:%s ===========' %(loop_count)
            loop_count = loop_count + 1
            improvement = 0
            for node in network[0]:
                node_community = self.communities[node]
                # 默认最佳社区是其自身
                best_community = node_community

                ### 模块度增益 变化：Max(合并因子)
                best_gain = 0
                # 从其社区中删除_node
                best_partition[node_community].remove(node)
                best_shared_links = 0

                for e in self.edges_of_node[node]:
                    if e[0][0] == e[0][1]:
                        continue
                        #如果点和邻居节点在同一个社区，则best_shared_links +1
                    if e[0][0] == node and self.communities[e[0][1]] == node_community or e[0][1] == node and self.communities[e[0][0]] == node_community:
                        best_shared_links += e[1]
                #一个点移除社区后内部权重外部权重同时减少
                self.s_in[node_community] -= 2 * (best_shared_links + self.w[node])
                self.s_tot[node_community] -= self.k_i[node]

                #把原来节点所在社区置为-1
                self.communities[node] = -1

                communities = {} # 只考虑不同社区的邻居
                for neighbor in self.get_neighbors(node):
                    #邻居节点所在社区
                    community = self.communities[neighbor]
                    #社区已经被计算过了
                    if community in communities:
                        continue

                    communities[community] = 1
                    shared_links = 0
                    for e in self.edges_of_node[node]:
                        if e[0][0] == e[0][1]:
                            continue

                        #计算新社区的增加的内部权重
                        if e[0][0] == node and self.communities[e[0][1]] == community or e[0][1] == node and self.communities[e[0][0]] == community:
                            shared_links += e[1]



                    # 计算通过将_node移动到_neighbor的社区获得的模块性增益
                    gain = self.compute_modularity_gain(node, community, shared_links)
                    cosValue = self.getCosSimilarityById(node_community,community)
                    site_merge_gain = self.getMegeFactor(gain,cosValue)
                    site_merge_gain = gain
                    
                    if site_merge_gain > best_gain:
                        #print "gain %s > best_gain: %s" % (gain,best_gain)
                        best_community = community
                        best_gain = site_merge_gain
                        best_shared_links = shared_links




                # 将_node插入模块性增益最大的社区
                best_partition[best_community].append(node)
                self.communities[node] = best_community
                #记录主题该节点的最佳社区
                self.site_communities_tags[node] = best_community
                self.s_in[best_community] += 2 * (best_shared_links + self.w[node])
                self.s_tot[best_community] += self.k_i[node]
                if node_community != best_community:
                    improvement = 1


            if not improvement:
                break
        return best_partition

    '''
        产生与_node相邻的节点。
         _node：int
    '''
    def get_neighbors(self, node):
        for e in self.edges_of_node[node]:
            if e[0][0] == e[0][1]: # 节点不与其自身相邻
                continue
            if e[0][0] == node:
                yield e[0][1]
            if e[0][1] == node:
                yield e[0][0]

    '''
        从_network构建初始分区。
          _network：（nodes，edges）
    '''
    def make_initial_partition(self, network):
        partition = [[node] for node in network[0]]
        self.s_in = [0 for node in network[0]]
        self.s_tot = [self.k_i[node] for node in network[0]]
        for e in network[1]:
            if e[0][0] == e[0][1]: # 只有自循环
                self.s_in[e[0][0]] += e[1]
                self.s_in[e[0][1]] += e[1]
        return partition

    '''
       执行方法的第二阶段。
         _network：（nodes，edges）
         _partition：节点的列表
    '''
    def second_phase(self, network, partition):
        nodes_ = [i for i in range(len(partition))]

        # 重新分配社区
        communities_ = []
        site_communities_tags_ = []
        d = {}
        i = 0
        for community in self.communities:
            if community in d:
                communities_.append(d[community])
                site_communities_tags_.append(d[community])
            else:
                d[community] = i
                communities_.append(i)
                site_communities_tags_.append(i)
                i += 1


        self.communities = communities_
        self.site_communities_tags = site_communities_tags_

        # 重造相连的边
        edges_ = {}
        for e in network[1]:
            #用社区id作为新节点坐标 并计算权重
            ci = self.communities[e[0][0]]
            cj = self.communities[e[0][1]]
            try:
                edges_[(ci, cj)] += e[1]
            except KeyError:
                edges_[(ci, cj)] = e[1]

        edges_ = [(k, v) for k, v in edges_.items()]


        #合并主题
        site_tags_ = {}
        for node in network[0]:
            if self.site_tags.has_key(node):
                tags = self.site_tags[node]
                newNode = self.site_communities_tags[node]
                site_tags_ = self.mergeTags(newNode,site_tags_,tags)


        self.site_tags = site_tags_


        # 重新计算k_i向量并且按节点存储边缘
        self.k_i = [0 for n in nodes_]
        self.edges_of_node = {}
        self.w = [0 for n in nodes_]
        for e in edges_:
            self.k_i[e[0][0]] += e[1]
            self.k_i[e[0][1]] += e[1]
            if e[0][0] == e[0][1]:
                self.w[e[0][0]] += e[1]
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        # 重置社区
        self.communities = [n for n in nodes_]
        return (nodes_, edges_)


    def mergeTags(self,newNode,site_tags_,tags):
        if not tags :
            return site_tags_

        if not site_tags_.has_key(newNode):
            site_tags_[newNode] = tags
            return site_tags_;


        site_tags_[newNode] = site_tags_[newNode] + ' ' + tags

        return site_tags_

        

    def getMegeFactor(self,best_gain,cosValue):

        site_merge_gain = merge_factor * best_gain + (1.0 -  merge_factor) * cosValue * best_gain
        return site_merge_gain
    


    def getCosSimilarityById(self,source_comm_id,des_comm_id):
        if not self.site_tags.has_key(source_comm_id) or not self.site_tags.has_key(des_comm_id):
            return 0.0

        texta = self.site_tags[source_comm_id]
        textb = self.site_tags[des_comm_id]

        return self.getCosSimilarity(texta,textb) 
    


    def getCosSimilarity(self,textA='',textB=''):
        if textA == '' or textB == '':
            return 0;

        paramstr = {}
        paramstr['texta'] = textA
        paramstr['textb'] = textB


        params = urllib.urlencode(paramstr)
        response =  urllib.urlopen('http://127.0.0.1:8882/similarity/cos',params).read()
        resDic = json.loads(response)
        if resDic.has_key('data'):
            return resDic['data']
        
        return 0

    def tsta_fa(self,s = 0.0,q = 0.0):
        if s+q - 0.0 == 0.0:
            return 0.0

        if s*q - 0.0 == 0.0:
            return 0.0
        
        fenzi = (1 + f_a) * (1 + f_a)*(s*q)
        fenmu = f_a*f_a*(s+q)

        return (fenzi / fenmu)


'''
    重建具有连续节点标识的图。
     _nodes：int型
     _edges：（（int，int），weight）
'''
def in_order(nodes, edges,site_tags={}):
        # 重建具有连续标识符的图
        nodes = list(nodes.keys()) #key按顺序输出为list

        nodes.sort() #排序
        i = 0
        nodes_ = []
        d = {}
        for n in nodes:
            nodes_.append(i)
            d[n] = i
            i += 1

        edges_ = []
        for e in edges:
            edges_.append(((d[e[0][0]], d[e[0][1]]), e[1]))

        site_tags_={}
        for (k,v) in site_tags.items():
            site_tags_[d[k]] = v
        return (nodes_, edges_,site_tags_)

def draw_networkx(vertices,edges):
    newedges = []
    for tupe in edges:
        newedges.append(tupe[0])

    g = Graph()
    g.add_vertices(vertices)
    g.add_edges(newedges)
    plot(g)

