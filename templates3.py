template_process=[(
            "system",
            "You will be designed to extract entities from information and apply them to the Entity-Relationship Diagram."
            "You are an expert extraction algorithm."
        ),  
        ("human", "Please process the article according to knowledge graph theory. {input}")
        ]
template_relation=[(
    "system",
    "You're an expert at finding relationships between entities."
    "You look for relationships to help build Entity-Relationship Diagram"
    "You will clarify the relationship between the entities and explain why."
    "You need to analyze the connections between each entity, identify the entities that are related, and say what is the connection between them."
    "Do not repeat the relationship."
    "You must Answer in Chinese."
),
("human", "Use the given format to analyze relationships between entities:{entities}"),
("human", "The relationships between entities must be detailed and precise. Be sure to include all relevant information."),
("human", "dont describe the relationship in simple terms of related to"),
("human", "Tip: Make sure to answer in the correct format."
          "No entity should be left out."
)
]

