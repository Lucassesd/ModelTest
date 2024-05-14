import json
from openai import OpenAI
from openai import OpenAI
from templates import permissible_nodes_to_extract
from models.ai_answer import AI_answer
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("mysql+pymysql://root:root@localhost:3306/crawler", echo=True)
Session = sessionmaker(bind=engine)
@contextmanager
def scoped_session():
        db_session = Session()
        try:
            yield db_session
        finally:
            db_session.close()

client = OpenAI(api_key="sk-33ed26e61471401ba1cf2899e855bcec", base_url="https://api.deepseek.com/v1")
contents_id=int(input("input the id of the content: "))
with scoped_session() as conn:
    contents=conn.query(AI_answer.content).offset(contents_id-1).limit(1).all()
    content = contents[0][0] if contents else None
if content:
  response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content":
            "You are engineered for organising data into knowledge graphs."
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value."
            "Nodes: Represent entities and ideas."
            "The objective is to ensure the knowledge graph is straightforward and intelligible for broad use."
            "Uniformity: Stick to simple labels for nodes. For instance, label any entity that is an organisation as 'company', rather than using terms like 'Facebook' or 'Amazon'."
            "Identifiers for Nodes: Opt for textual or comprehensible identifiers over numerical ones."
            "Permissible Node Labels**: If there are specific allowed node labels, list them here."
            "Permissible Relationship Types**: If there are specific allowed relationship types, list them here."
        },
        {"role": "user", "content":content},
        {"role": "user","content": "Use the given permissible_nodes_to_extract to help you fine nodes","permissible_nodes":permissible_nodes_to_extract},
        
        {"role": "assitant", "content": "Use the given format to extract information from the following input."
        "Managing Numerical Data and Dates:Integrate numerical information directly as attributes of nodes."
        "Integrated Dates/Numbers**: Refrain from creating distinct nodes for dates or numbers, attaching them instead as attributes."
        "Format for Properties**: Use a key-value pairing format."
        "Avoiding Quotation Marks**: Do not use escaped quotes within property values."
        "Key Naming**: Adopt camelCase for naming keys, such as `dateTime`."
        "Uniformity:Entity Uniformity: Ensure consistent identification for entities across various mentions or references."
        "Adherence to Guidelines:Strict adherence to these instructions is mandatory. Non-adherence will result in termination."
        "Tip: Make sure to answer in the correct format"
        "question must be correctly formatted,question_detail must be correctly formatted,keywords must be correctly formatted,solution must be correctly formatted,relationship must be correctly formatted"
        "must use chinese characters to describe the content."}
        
        
  ],
    max_tokens=1024,
    temperature=0,
    stream=False
  )
    
def extract_json_text(input_text):
    start_index = input_text.find("{")
    end_index = input_text.rfind("}") + 1
    json_text = input_text[start_index:end_index]
    return json_text


def parse_extraction(json_text):
    data = json.loads(json_text)
    output = "Extracted Information:\n\n"
    for node in data["nodes"]:
        identifier = node.get("identifier") or node.get("id") or "Unknown Identifier"
        properties = node.get("properties")
        attributes = node.get("attributes")
        output += f"Identifier: {identifier}\n"
        if properties:
            output += "Properties:\n"
            for key, value in properties.items():
                output += f"{key}: {value}\n"
        elif attributes:
            output += "Attributes:\n"
            for key, value in attributes.items():
                output += f"{key}: {value}\n"
        else:
            output += "No properties or attributes found.\n"
        output += "\n"
    return output


if content:
    for message in response.choices:
        role = message.message.role
        content = message.message.content
        if content:
            # json_text=extract_json_text(content)
            # parse=parse_extraction(json_text)
            # print(parse)
            print(content)