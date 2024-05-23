template_process=[(
            "The objective is to ensure the knowledge graph is straightforward and intelligible for broad use."
        ),  
        ("human", "Please process the article according to knowledge graph theory. {input}")
        ]
template_relation=[(
    "system",
    "You are designed to analyze relationships between entities and chain entities together to build a knowledge graph."
    "When you analyze relationships between entities, be sure to be precise and detailed."
    "Analyze from the cluster, which entities are highly similar, and find out the connection between the entities with high similarity."
    "You must specify which entity, for example :'facebook'"
    "You must explain why they are relevant."
    "You must Describe in Chinese."
),
("human", "Use the given format to analyze relationships between entities:{put1}"),
("human", "The relationships between entities must be detailed and precise. Be sure to include all relevant information."),
("human", "dont describe the relationship in simple terms of related to"),
("human", "Tip: Make sure to answer in the correct format."
          "Describe in Chinese."
          "No entity should be left out."
          "Be sure to write what is the connection between the entities."
          "Cannot output attribute."
)
]

