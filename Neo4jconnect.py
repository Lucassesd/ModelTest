from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = "neo4j" 
password = "root" 

driver = GraphDatabase.driver(uri, auth=(user, password))


def create_entities_and_relationships(tx):
    pass

with driver.session() as session:
    session.write_transaction(create_entities_and_relationships)
    
driver.close()