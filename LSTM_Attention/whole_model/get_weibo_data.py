import requests
import re
from bs4 import BeautifulSoup
from xml.dom import minidom
import js2xml

class WeiboData():
    url = 'https://m.weibo.cn/comments/hotflow?'
    headers = {
        'cookie':'WEIBOCN_FROM=1110006030; loginScene=102003; SUB=_2A25MuTsHDeRhGeBL7FEX8S7FzT-IHXVsQkVPrDV6PUJbkdAKLRXEkW1NRsp5QzK3eqk7xWpYHfAVDHnRN9IdnYvF; _T_WM=40255559233; XSRF-TOKEN=d76051; MLOGIN=1; M_WEIBOCN_PARAMS=oid%3D4715427044000803%26luicode%3D20000061%26lfid%3D4715427044000803%26uicode%3D20000061%26fid%3D4715427044000803',
        'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1'
    }
    max_id, max_id_type = 0, 0
    params = {
        'id': None,
        'mid': None,
        'max_id': max_id,
        'max_id_type': max_id_type
    }

    '''
        利用正则表达式匹配
        去掉< >之间的内容
    '''
    def handle_string_by_re(self, input_string):
        return re.sub('<.*?>', '', input_string)

    '''
        num代表要爬取几页的评论
        id代表要爬取的用户id
        返回所有的评论(list),若没有评论返回['抱歉，您所查询的微博暂无评论信息']
    '''
    def catch_review(self, id, num):
        reviewList = []
        self.params['id'], self.params['mid'] = id, id
        for i in range(0, num + 1):
            self.params['max_id'], self.params['max_id_type'] = self.max_id, self.max_id_type
            response = requests.get(url=self.url, headers=self.headers, params=self.params)
            # 考虑到没有评论的情况
            try:
                self.max_id, self.max_id_type = response.json()['data']['max_id'], response.json()['data']['max_id_type']
                # 获取每一页评论的条数
                l = len(response.json()['data']['data'])
                for k in range(0, l):
                    # 先使用正则表达式对评论进行清洗
                    review = self.handle_string_by_re(response.json()['data']['data'][k]['text'])
                    # 只记录长度大于10的评论
                    if len(review) > 10:
                        reviewList.append(review)
                return reviewList
            except:
                return ['抱歉，您所查询的微博暂无评论信息']



    '''
        根据Url获取微博的文本
        return 通过re处理好的文本
    '''
    def getBlobByUrl(self, url):
        r = requests.get(url)
        demo = r.text
        # 使用BeautifulSoup的lxml HTML解析器
        soup = BeautifulSoup(demo, 'lxml')
        # 获取<body>标签中第一个<script>标签中的内容
        src = soup.select('body script')[0].string
        src_text = js2xml.parse(src, encoding='utf-8', debug=False)
        # 由tree_element转成string类型
        src_tree = js2xml.pretty_print(src_text)
        dom = minidom.parseString(src_tree)
        # 获取所有的<property>标签
        properties = dom.getElementsByTagName('property')
        # 保存获取拿到的<property>标签 name='text'的内容
        content = ''
        for i in range(len(properties)):
            if properties[i].getAttribute('name') == 'text':
                # 读取标签名字
                # print(properties[i].childNodes[1].tagName)
                # 读取标签值
                content = properties[i].childNodes[1].childNodes[0].data
                break
        # 通过正则表达式对获取到的文本进行处理
        return self.handle_string_by_re(content)


    '''
        根据url返回元组: (微博文本, 评论信息)
        (整体封装)
    '''
    def get_blob_review_by_url(self, url):
        # 获取微博文本信息
        blob = self.getBlobByUrl(url)
        # 对字符串进行切割，查找最后的id
        index = url.rindex('/')
        uid = url[index + 1:]
        # 返回微博评论信息
        reviews = self.catch_review(uid, 2)
        return blob, reviews


if __name__ == '__main__':
    # url = 'https://m.weibo.cn/1699432410/4715427044000803'
    url = 'https://m.weibo.cn/1241148864/4715914630530847'
    wb = WeiboData()
    print(wb.get_blob_review_by_url(url))

