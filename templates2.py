template_entity=[
        (
            "system",
            "You are extracting noun and verb entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following"
            "input: {question}",
        ),
    ]

template1 = """Given the following d a follow up question, rephrase the follow up question to be a standalone question,
in its original language.conversation an
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501


template_search = """Finds all the contextual information that hit the question based on the given context:
{context}

Question: {question}
Use natural language and be concise.
Use Chinese.
information:
"""

template_grade="""
## instruction下面我需要你对一个知识库的召回效果进行打分，\
我会给你一个expected_result和coll_result，\
coll_result为top5的result的数组，每个result包含content，score，source三个部分，\
你只需要关注content。随后请定位expected_result在coll_result出现的位置(需要基本一致或者result包含expected_result才能算作出现)，\
越早在coll_result出现说明效果越好，请按照下列标准对其进行相关性评分：top1--100，top2--80，top3--60，top4--40，top5--20，未出现--0,\
用中文回答。

## output_paser:
你最后输出的结果应该为：```相关性评分:xx```
下面分别是expected_result和coll_result
## expected_result:{expected_result}
## coll_result:{coll_result}
"""