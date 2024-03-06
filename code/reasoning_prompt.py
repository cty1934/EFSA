from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("model-name", trust_remote_code=True)
model = AutoModel.from_pretrained("model-name", trust_remote_code=True).half().cuda()
model = model.eval()


question = """假设你是一个金融领域的细粒度情感分析模型，我会给你一些情感极性列表、一级事件列表、二级事件列表和相关的一些金融新闻，
请分析这个新闻报道中是哪一个公司的事件，进而判断这个事件属于哪个一级事件，并进一步根据一级事件判断属于哪个二级事件，最后判断这篇金融新闻属于情感极性中的哪一种。
注意结果不能为空，不要回答多余的话。
情感极性：[积极, 中性, 消极]。
一级事件类型：['财务', '股东', '股票', '管理、合规、信用问题',  '经营问题',  '融资、投资、并购']。
二级事件类型：['利润公布', '利润预告', '其他财务动态', '股东增减持', '股东质押', 
'其他股东事件', '股价变动', '股票状态', '限售股解禁', '股票回购', '股票评级调整',  
'股权激励&员工持股计划','董监高动态', '评级变动', '法律事务', '监管问询', '员工动态',
 '项目中标', '开展合作', '新公司成立', '销量、份额变动', '知识产权', '资质变动', 
 '技术质控变动', '政府补贴', '机构调研', '产能产量变动', '其他经营问题', '资金流动', '投资事件', '融资融券',’公司上市’]。
 金融新闻：
"""
end="请用四元组列表的形式（公司名称，一级事件，二级事件，情感极性）进行回答，只给出一个答案即可。"
finance_event_list={
    '财务':['利润公布', '利润预告', '其他财务动态'],
    '股东':['股东增减持', '股东质押', '其他股东事件'],
    '股票':['股价变动', '股票状态', '限售股解禁', '股票回购', '股权激励&员工持股计划', '限制股解禁', '股票分红','其他股票事件'],
    '管理':['董监高动态', '员工动态'],
    '合规信用':['监管问询', '公司涉诉', '立案调查', '行政处罚', '澄清公告', '法律事务', '评级变动', '其他违规事件'],
    '经营问题':['项目中标','其他经营事件', '开展合作', '新公司成立', '销量、份额变动', '知识产权', '技术质控、资质变动', '政府补贴', '机构调研', '产能产量变动', '项目动态', '产品动态',],
    '融资投资':['资金流动', '投资事件', '融资融券', '公司上市', '并购重组', '股票增发', '其他融资事件']

}
question1="上述金融新闻中，描述的是哪一个公司发生的事件？只回答出公司名称，不要说多余的话以及标点符号。"
with open("  ",'r',encoding='utf-8') as f:
    for line in f.readlines():
        answer=[]
        tuple=()
        text=line.split('####')[0]
        jump=text+question1

        response, history = model.chat(tokenizer, jump, history=None)

        jump1=text+"""上述金融新闻中这个"+response+"发生的是什么类别的事情，请从下列一级事件类型：
        [财务,股东,股票,管理,合规信用,经营,融资投资]，选出对应的一级事件类型，必须从给定的一级事件列表中选择。
        只说出一级事件类型，不要回答多余的话以及标点符号。"""

        response_jump1, history = model.chat(tokenizer, jump1, history=history)
        if response_jump1 in finance_event_list.keys():
            jump2=text+"上述金融新闻中这个"+response+"发生的一级事件是"+response_jump1+"，请从下列二级事件类型："+"["+', '.join(finance_event_list[response_jump1])+"]"+"中，选出一个对应的二级事件类型。只说出二级事件类型，不要回答多余的话以及标点符号。"
        else:
            jump2=text+"上述金融新闻中这个"+response+"发生的一级事件是"+response_jump1+"""，请从下列二级事件类型：['利润公布', '利润预告', '其他财务动态', '股东增减持', '股东质押', 
            '其他股东事件', '股价变动', '股票状态', '限售股解禁', '股票回购', '股票评级调整',  
            '股权激励&员工持股计划','董监高动态', '评级变动', '法律事务', '监管问询', '员工动态',
            '项目中标', '开展合作', '新公司成立', '销量、份额变动', '知识产权', '资质变动', 
            '技术质控变动', '政府补贴', '机构调研', '产能产量变动', '其他经营问题', '资金流动', '投资事件', '融资融券',’公司上市’]。选出一个对应的二级事件类型。只说出二级事件类型，不要回答多余的话以及标点符号。"""
      
        response_jump2, history = model.chat(tokenizer, jump2, history=history)

        jump3=text+"上述金融新闻中这个"+response+"发生的一级事件是"+response_jump1+"，二级事件是"+response_jump2+"，从情感极性列表：[积极, 中性, 消极]，利用你所学的金融知识选出相应的情感极性。只回答情感极性，不要回答多余的话以及标点符号。"
        response_jump3, history = model.chat(tokenizer, jump3, history=history)

        tuple=(response,response_jump1,response_jump2,response_jump3)
        answer.append(tuple)
        with open("    ",'a',encoding='utf-8') as f:
            f.write(str(answer)+'\n')

