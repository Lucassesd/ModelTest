import json
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





# 使用示例
input_text = '''{
  "nodes": [
    {
      "id": "node1",
      "label": "NonPointerGC",
      "properties": {
        "description": "垃圾回收不会扫描不含指针的slice，这可以用于设计零GC的本地缓存map。",
        "rootCause": "垃圾回收不会扫描不含指针的slice，因此可以利用这一特性设计零GC的本地缓存map，提高系统性能。"
      }
    },
    {
      "id": "node2",
      "label": "PaddingAvoidFalseSharing",
      "properties": {
        "description": "在性能要求特别高的并发访问同一个对象的场景中，可以通过增加padding的方式避免false sharing，提升CPU cache的命中率，从而提升性能。",
        "rootCause": "在性能要求特别高的并发访问同一个对象的场景中，通过增加padding的方式避免false sharing，可以提升CPU cache的命中率，从而提升性能。"
      }
    },
    {
      "id": "node3",
      "label": "EscapeToHeap",
      "properties": {
        "description": "在编写代码时，需要关注可能会逃逸到堆上的行为，以避免性能问题和内存泄漏。",
        "rootCause": "在编写代码时，可能会出现以下情况导致数据逃逸到堆上：将指针或包含指针的值发送到通道、将指针或包含指针的值存储在切片中、切片的底层数组因append操作而重新分配、在接口类型上调用方法。这些行为可能导致性能下降和内存 泄漏。"
      }
    },
    {
      "id": "node4",
      "label": "ReuseAllocatedMemory",
      "properties": {
        "description": "通过复用已分配的内存，可以提高系统的性能和效率。",
        "rootCause": "通过使用sync.Pool存放临时变量，可以实现协程间共享已分配内存，从而提高系统的性能和效率。另外，通过对[]byte进行复用，可以避免string转换到[]byte造成的内存分配和拷贝。"
      }
    },
    {
      "id": "node5",
      "label": "ZeroCopyConversion",
      "properties": {
        "description": "实现字符串和[]byte切片之间的转换，要求是zero-copy。",
        "rootCause": "通过使用底层数据结构StringHeader和SliceHeader，可以实现字符串和[]byte切片之间的zero-copy转换。"
      }
    },
    {
      "id": "node6",
      "label": "MemoryAlignment",
      "properties": {
        "description": "在编写代码时，需要关注内存对齐，以提高CPU访问内存的吞吐量。",
        "rootCause": "Go在编译时会自动进行内存对齐，以减少CPU访问内存的次数，提高吞吐量。因此，关注内存对齐可以提高系统的性能。"
      }
    },
    {
      "id": "node7",
      "label": "MapReadHeavy",
      "properties": {
        "description": "在map读多写少的场景中，可以采用读写分离和同步机制，降低并发抢锁概率，提高性能。",
        "rootCause": "在map读多写少的场景中，通过使用读写分离和合适的同步机制，可以降低并发抢锁概率，从而提高性能。"
      }
    }
  ],
  "relationships": []
}
'''


json_text = extract_json_text(input_text)
parse=parse_extraction(json_text)
print(parse)