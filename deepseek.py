import json
from openai import OpenAI
from openai import OpenAI
from templates import Examples_of_entities
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
            "You will be designed to extract entities from information and apply them to the knowledge graph."
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value."
            "Use the entity examples provided to help you find entities"
            "The objective is to ensure the knowledge graph is straightforward and intelligible for broad use."
        },
        {"role": "user", "content":content},
        {"role": "user","content":Examples_of_entities},
        
        {"role": "assitant", "content": "Use the given format to extract information from the following input."
        "Numerical information is directly integrated as attributes of entities."
         "Avoid creating different nodes for dates or numbers, and instead attach them as attributes."
         "Don't use escape quotes in property values."
         "Use camel case for keys, such as' dateTime '."
         "Entity consistency: Ensures consistent identification of entities across various mentions or references."
         "Strict adherence to these guidelines is mandatory. Failure to comply will result in dismissal."
         "The attributes of the entity must not be omitted.Attributes must be detailed."
         "The attributes of an entity must be sought based on the provided information without any omission, the attributes can be problem solutions, error messages, names, etc."
         "Opt for textual or comprehensible identifiers over numerical ones."
         "Make sure to answer in the correct format."}
        
        
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