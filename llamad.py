
from models.passage import Passage
from config import(
    NEO4J_URI,
    NEO4J_PASSWORD,
    NEO4J_USERNAME,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    DATABASE,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_community.document_loaders import WikipediaLoader
query = "Dune (Frank Herbert)"


tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")


graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

api_key=OPENAI_API_KEY
api_url=OPENAI_API_BASE

engine = create_engine(DATABASE, echo=True)
Session = sessionmaker(bind=engine)


text_splitter = RecursiveCharacterTextSplitter(
         chunk_size=512,
         length_function=len, 
         is_separator_regex=False,)
raw_documents = WikipediaLoader(query=query).load_and_split(text_splitter=text_splitter)

def extract_relations_from_model_output(text):
    relations = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("", "").replace("", "").replace("", "")
    
    for token in text_replaced.split():
        if token == "":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
                subject = ''
        elif token == "":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                object_ = ''
        elif token == "":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token

    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
        
    return relations

class KB:
    def __init__(self):
        self.relations = []

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def add_relation(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)

    def print(self):
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")

def from_small_text_to_kb(text, verbose=False):
    kb = KB()

    # Tokenize text
    model_inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')

    if verbose:
        print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

    # Generate
    gen_kwargs = {
        "max_length": 216,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3
    }

    generated_tokens = model.generate(
        **model_inputs,
        **gen_kwargs,
    )

    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # Create kb
    for sentence_pred in decoded_preds:
        relations = extract_relations_from_model_output(sentence_pred)
        for r in relations:
            kb.add_relation(r)

    return kb

for doc in raw_documents:
    kb = from_small_text_to_kb(doc.page_content, verbose=True)


    for relation in kb.relations:
        head = relation['head']
        relationship = relation['type']
        tail = relation['tail']


        cypher = f"MERGE (h:`{head}`)" + f" MERGE (t:`{tail}`)" + f" MERGE (h)-[:`{relationship}`]->(t)"
        print(cypher)
        graph.query(cypher)


graph.refresh_schema()

