# -*- coding:utf-8 -*-


# Author: Dandy Qi
# Created time: 2018/12/1 14:39
# File usage: 问答系统前端交互设置


from flask import Flask, request, make_response
import hashlib
import configparser
import time
import xml.etree.ElementTree as ET

from QAdataprocess import QAProcess

app = Flask(__name__)

# 微信公众平台消息格式，包含收信人，寄信人，时间，消息类型，消息内容
xml_rep = "<xml>\
    <ToUserName><![CDATA[{0}]]></ToUserName>\
    <FromUserName><![CDATA[{1}]]></FromUserName>\
    <CreateTime>{2}</CreateTime>\
    <MsgType><![CDATA[text]]></MsgType>\
    <Content><![CDATA[{3}]]></Content>\
    <FuncFlag>0</FuncFlag>\
    </xml>"

qa = QAProcess()
cf = configparser.ConfigParser()


# 路由配置，处理http://0.0.0.0/wx下的get与post请求
@app.route("/wx", methods=['GET', 'POST'])
def wechat():
    cf.read("config.conf")
    wechat_token = cf.get("token", "weChat")
    # get请求用于验证该服务权限，需要使用微信公众平台中配置的token进行验证
    if request.method == "GET":
        try:
            data = request.args
            token = wechat_token
            signature = data.get('signature', '')
            timestamp = data.get('timestamp', '')
            nonce = data.get('nonce', '')
            echostr = data.get('echostr', '')
            s = [timestamp, nonce, token]
            s.sort()
            s = ''.join(s)
            if hashlib.sha1(s).hexdigest() == signature:
                return make_response(echostr)
        except Exception as e:
            return e

    # post请求用于问答交互，通过解析xml格式数据获取相应内容
    else:
        rec = request.stream.read()
        xml_rec = ET.fromstring(rec)
        to_user = xml_rec.find('ToUserName').text
        from_user = xml_rec.find('FromUserName').text
        content = xml_rec.find('Content').text

        # content中包含用户query，调用问答流程进行处理
        reply = qa.response(content)
        res = xml_rep.format(from_user, to_user, int(time.time()), reply)
        print(res)
        response = make_response(res)
        return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
