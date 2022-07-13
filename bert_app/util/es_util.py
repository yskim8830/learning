from elasticsearch import Elasticsearch, helpers

class elastic_util:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        server_list = [ {'host':host, 'port':port}]
        self.es = Elasticsearch( server_list )
    
    #get es object
    def getEs(self):
        return self.es
    
    #getHealth
    def getHealth(self):
        return self.es.indices()
    
    #getInfo
    def getInfo(self):
        return self.es.info()
    
    #existIndex
    def existIndex(self, idx):
        return self.es.indices.exists(index=idx)
    
    #createindex
    def createindex(self, idx, mapping):
        if self.es.indices.exists(index=idx):
            pass
        else:
            return self.es.indices.create(index=idx, body=mapping)
    
    #deleteindex
    def deleteindex(self, idx):
        return self.es.indices.delete(index=idx, ignore=[400, 404])
    
    #createAlias
    def createAlias(self, aidx, idx):
        if self.es.indices.exists_alias(name=aidx):
            pass
        return self.es.indices.put_alias(name=aidx, index=idx)
        
    #searchAll
    def searchAll(self, idx, size=10):
        response = self.es.search(index=idx, size=size, body={"query": {"match_all": {}}})
        fetched = len(response['hits']['hits'])
        result = []
        for i in range(fetched):
            result.append(response['hits']['hits'][i]['_source'])
        return result
    
    #searchById
    def searchById(self, idx, id):
        response = self.es.search(index=idx, body={"query": {"match": { "_id" : id}}})
        fetched = len(response['hits']['hits'])
        result = []
        for i in range(fetched):
            result.append(response['hits']['hits'][i]['_source'])
        return result
    
    #search
    def search(self, idx, body):
        response = self.es.search(index=idx, body=body)
        fetched = len(response['hits']['hits'])
        result = []
        for i in range(fetched):
            result.append(response['hits']['hits'][i]['_source'])
        return result
    
    #countBySearch
    def countBySearch(self, idx, body):
        response = self.es.count(index=idx, body=body)
        return response['count']
    
    #scroll search
    def search_srcoll(self, idx, body):
        _KEEP_ALIVE_LIMIT='30s'
        response = self.es.search(index=idx, body=body, scroll=_KEEP_ALIVE_LIMIT, size = 100,)
        
        sid = response['_scroll_id']
        fetched = len(response['hits']['hits'])
        result = []
        for i in range(fetched):
            result.append(response['hits']['hits'][i]['_source'])
        while(fetched>0): 
            response = self.es.scroll(scroll_id=sid, scroll=_KEEP_ALIVE_LIMIT)
            fetched = len(response['hits']['hits'])
            for i in range(fetched):
                result.append(response['hits']['hits'][i]['_source'])
        return result
    
    def close(self):
        self.es.close()
    
    #insertData
    def insertData(self, idx, id, doc):
        return self.es.index(index=idx, id=id, body=doc)
    
    #updateData
    def updateData(self, idx, id, doc):
        return self.es.index(index=idx, id=id, body=doc)
    
    #deleteData
    def deleteData(self, idx, id):
        return self.es.delete(index=idx, id=id)
        
    #deleteAllData
    def deleteAllData(self, idx):
        return self.es.delete_by_query(index=idx, body={"query":{"match_all":{}}})
    
    #bulk
    def bulk(self, body):
        # body.append({
    	# '_index': [인덱스_이름],
        # '_source': {
        #     "category": "test"
        #     "c_key": "test"
        #     "status": "test"
        #     "price": 1111
        #     "@timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        #     }
        # })
        return helpers.bulk(self.es, body)

#es = elastic_util('192.168.0.5', '6251')
#print(es.countBySearch('@prochat_dic', ''))