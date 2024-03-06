import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("model-name", trust_remote_code=True)
model = AutoModel.from_pretrained("model-name", trust_remote_code=True).half().cuda()
model = model.eval()
question1="""
假设你是一个金融领域的细粒度情感分析模型，我会给你一些情感极性列表、一级事件列表、二级事件列表和相关的一个金融新闻，
请分析这个金融新闻中是哪一个公司的事件，进而判断这个事件属于哪个一级事件，并进一步根据一级事件判断属于哪个二级事件，最后判断这篇金融新闻属于情感极性中的哪一种。
情感极性：[积极, 中性, 消极]。
一级事件类型：['财务', '股东', '股票', '管理、合规、信用问题', '经营问题', '融资、投资、并购']。
二级事件类型：['利润公布', '利润预告', '其他财务动态', '股东增减持', '股东质押',
'其他股东事件', '股价变动', '股票状态', '限售股解禁', '股票回购', '股票评级调整',
'股权激励&员工持股计划','董监高动态', '评级变动', '法律事务', '监管问询', '员工动态',
'项目中标', '开展合作', '新公司成立', '销量、份额变动', '知识产权', '资质变动',
'技术质控变动', '政府补贴', '机构调研', '产能产量变动', '其他经营问题', '资金流动', '投资事件', '融资融券',’公司上市’]。
金融新闻：浦东建设公告,近日公司子公司上海市浦东新区建设(集团)有限公司、上海浦东路桥(集团)有限公司中标多项重大工程项目,中标金额总计15.66亿元。
"""
end="""
请用四元组列表的形式[('公司名称'，'一级事件'，'二级事件'，'情感极性')]进行回答。
"""
response, history = model.chat(tokenizer, question1, history=None)
print(response)
   


