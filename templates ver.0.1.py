template_process=[(
            "system",
            "Using knowledge graph theory, analyze the following article and summarize the entities."
        ),  
        ("human", "Summarize entities based on the context of the article using knowledge graph theory. Here the article i give you:{input}"
                  "You must Describe in Chinese.")
        ]
template_relation=[(
    "system",
    "使用知识图谱理论分析给你的所有实体间的关系，例如人和车的关系是依赖，代码和文件的关系是容器"
    "When you analyze relationships between entities, be sure to be precise and detailed futhermore, you have to Summarize all relationships into one word."
),
("human", "这是给你的文章{put2}"),
("human", "Use the given format to analyze relationships between entities:{put1}"),
]

