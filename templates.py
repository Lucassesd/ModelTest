example_question = """
问题的相关特征的类型一共有6种：
1.堆栈：堆栈是程序运行时函数调用的记录。作为问题的相关特征，堆栈可以提供问题发生时的函数调用路径，帮助我们理解问题发生的上下文和可能的原因。
2.火焰图：火焰图是一种可视化工具，用于展示程序运行时的资源利用情况。作为问题的相关特征，火焰图可以帮助我们发现程序中的性能瓶颈，比如哪些函数占用了过多的CPU时间。
3.异常指标：异常指标是指与正常行为相比，显著偏离或突变的统计指标。作为问题的相关特征，异常指标可以帮助我们识别和定位可能的性能问题或资源泄漏。
4.异常日志：异常日志是指程序运行过程中记录的错误或警告信息。作为问题的相关特征，异常日志可以提供问题发生时的详细情况，帮助我们更准确地识别问题。
5.异常配置：异常配置是指程序的配置信息错误或不合理，可能导致程序行为异常。作为问题的相关特征，异常配置可以帮助我们发现配置错误或不合适的配置选项。
6.代码：代码是指构成程序的源代码。作为问题的相关特征，代码的特定部分（如可能存在错误的函数或模块）可以帮助我们理解问题的原因，特别是当问题可能由代码错误或设计缺陷引起时。
"""

example_feature = """
问题特征1：堆栈信息导致的性能瓶颈
问题描述：用户反映系统响应缓慢，特别是在执行特定操作时，用户反馈体验不佳。
根因分析：系统性能瓶颈的根本原因在于calculateComplexAlgorithm函数被频繁调用且调用深度较大，该函数占用了大量CPU时间，导致系统响应缓慢和CPU使用率偏高。
相关特征：
    堆栈：获取用户反映的慢操作时的函数调用堆栈，发现堆栈中多次出现calculateComplexAlgorithm函数，且该函数调用深度较大。
    火焰图：通过火焰图分析，calculateComplexAlgorithm函数占用了大量的CPU时间，成为性能瓶颈。
    异常指标：系统平均响应时间增加，CPU使用率持续偏高。
    代码：分析calculateComplexAlgorithm函数的源代码，发现其中存在大量的循环和递归调用，导致函数执行效率低下。
        具体内容：
            public void calculateComplexAlgorithm(int input) {
                // 递归调用
                int result = recursiveFunction(input);
                // 循环操作
                for (int i = 0; i < result; i++) {
                    // 执行循环体内的操作
                }
            }
            private int recursiveFunction(int n) {
                if (n <= 1) {
                    return 1;
                }
                // 递归调用
                return n * recursiveFunction(n - 1);
            }
解决方案：
    优化算法：
        -重新设计calculateComplexAlgorithm函数的算法逻辑，减少不必要的计算量，提高执行效率。
        -使用迭代代替递归，避免递归调用带来的栈空间消耗和性能开销。
        -简化循环操作，优化循环体内的计算逻辑，减少每次循环的耗时。
        具体内容：
            public void OptimizeCalculateComplexAlgorithm(int input) {
            // 使用迭代代替递归调用
            int result = iterativeFunction(input);
            // 简化循环操作
            // 优化后的循环操作...
        }
        private int iterativeFunction(int n) {
            int result = 1;
            for (int i = 1; i <= n; i++) {
                result *= i;
            }
            return result;
        }
    异步处理：将calculateComplexAlgorithm函数的调用放到后台线程或异步任务中执行，避免阻塞主线程，提高系统的响应性；使用线程池或异步框架来管理异步任务，确保任务的合理调度和执行。
    缓存机制：如果calculateComplexAlgorithm函数的计算结果具有可重用性，可以引入缓存机制来存储计算结果；在函数被调用时，首先检查缓存中是否已存在结果，如果存在则直接返回缓存结果，避免重复计算。
    代码审查：对调用calculateComplexAlgorithm函数的代码进行审查，识别并消除不必要的重复调用；对调用逻辑进行重构，合并相似的调用或优化调用顺序，减少函数调用次数和深度。
    监控与调优：建立持续的性能监控机制，定期收集和分析系统性能数据，及时发现并解决性能瓶颈；根据监控结果进行调优，不断优化系统性能，提升用户体验。

问题特征2：异常日志指示的数据库连接问题
问题描述：应用程序在尝试连接数据库时失败。
根因分析：数据库连接问题的根本原因在于数据库连接字符串中的数据库地址配置错误，导致应用程序无法成功连接到数据库，错误的配置导致了连接超时，引发多条“数据库连接超时”的错误记录，并使得应用程序无法正常访问数据库，进而影响了系统的正常运行。
相关特征：
    异常日志：检查应用程序日志，发现多条“数据库连接超时”的错误记录，包含了错误的时间戳、错误级别、错误信息以及出错位置。
        具体内容：
            [ERROR] 2023-04-25 15:30:12,123 - DatabaseConnectionService
            at DatabaseConnectionService.ConnectToDatabase() in C:\Project\DatabaseConnectionService.cs:line 45  
            at ApplicationLayer.ExecuteQuery() in C:\Project\ApplicationLayer.cs:line 89  
    代码：检查数据库连接相关的代码，发现连接字符串中的数据库地址配置错误，导致连接失败。
    异常指标： database_connection_failures变量增加，应用程序的 failure_rate 上升，连接失败的严重程度加大；\\
        应用程序 response_time 延长，数据库查询操作的平均 response_time 增加，影响了整体系统性能。
解决方案：
    修正连接字符串：检查并修正数据库连接字符串中的数据库地址配置。
        具体内容：
            public class DatabaseConnectionService  {  
                private static readonly string CorrectDatabaseAddress = "your_correct_database_address";  
                public void ConnectToDatabase()  {  
                    string connectionString = $"Server={ CorrectDatabaseAddress };Database=your_database;User Id=your_username;Password=your_password;";  
                    // 使用修正后的连接字符串连接数据库  
                    // ... 连接数据库的代码 ...  
                }  
            }  
    重试机制：在数据库连接失败时，实现重试逻辑，避免单次连接失败导致整个操作失败。
    异常处理：增加异常捕获和处理逻辑，记录详细的错误信息，并给出友好的用户提示。

问题特征3：数据指标异常指示的存储问题
问题描述：用户反馈数据库查询速度变慢，且系统存储空间告警。
根因分析：存储问题的根本原因在于数据库查询操作涉及大量数据的JOIN操作，导致查询响应时间显著增加，同时系统剩余存储空间急剧下降，触发存储空间告警阈值，提示存储空间接近极限。
相关特征：
    异常指标：数据库查询的平均响应时间从之前的50毫秒增加至现在的200毫秒，超出正常波动范围；系统剩余存储空间从30%下降至5%，触发存储空间告警阈值。这一现象不符合预期，存储空间不应该是性能瓶颈。
    堆栈：获取查询操作时的函数调用堆栈，发现查询操作涉及到大量数据的JOIN操作，导致执行时间延长。
        具体内容：
            at DatabaseQueryExecutor.ExecuteComplexQuery() in C:\Project\DatabaseLayer\DatabaseQueryExecutor.cs:line 102  
            at DataAccessService.GetDataWithJoins() in C:\Project\DataAccessService.cs:line 157  
            at BusinessLogic.ProcessDataWithJoins() in C:\Project\BusinessLogic.cs:line 89  
            at WebController.GetDataAction() in C:\Project\WebController.cs:line 45  
    异常日志：数据库日志中显示多条“磁盘空间不足”的警告信息，指示存储空间接近极限。
解决方案：
    优化查询：对涉及大量JOIN操作的查询进行优化，减少不必要的数据关联，提高查询效率。
    分页处理：对于返回大量数据的查询，实现分页处理，避免一次性加载过多数据。
        具体内容：
            public List<DataEntity> GetDataWithPagination(int pageNumber, int pageSize)  {  
                // 构造分页查询的SQL语句  
                string sql = "SELECT * FROM your_table ORDER BY some_column OFFSET @Offset ROWS FETCH NEXT @FetchRows ONLY";  
                // 设置分页参数  
                int offset = (pageNumber - 1) * pageSize;  
                int fetchRows = pageSize;  
                // 执行查询并返回结果  
                // ... 执行查询的代码，使用参数offset和fetchRows ...  
            }
    清理无用数据：定期清理数据库中不再需要的数据，释放存储空间。
    监控与告警：加强存储空间的监控，设置合理的告警阈值，及时发现并处理存储空间不足的问题。
"""

template_output = """
目标：
严格参考信息中的输出模板，根据信息中的特征类型定义和文章内容，找到文章的主要问题、特征和解决方案，并按要求的形式输出。
如果输出的内容是输出模板里的内容，请把输出内容修改为与原文强相关的内容。

步骤：
1.通过文章的开头部分，总结出文章的主要希望解决的问题，也就是作者的意图和目的，一般只有1个，少数情况下为多个。
2.如果文章的主要问题只有1个，请只总结出1个围绕主要问题的问题特征。
3.如果文章的主要问题有多个，请总结出和主要问题数量一致的多个问题特征。
4.检查所有的相关特征和解决方案是否存在具体内容，如果匹配到的具体内容有遗漏，请完善补充具体内容。
5.检查输出，如果输出存在输出模板中的原内容，请删除。

规则：
1.如果输出的内容是输出模板里的内容，请把输出内容修改为与原文强相关的内容。
2.具体内容必须是代码、日志、数据结构与算法、配置、测试结果和性能数据、API文档链接和调用示例。
3.相关特征的类型，一定是给定的6种类型之一，如果不在6种类型的范围内，需要改成最接近的范围内的类型。
4.根因分析中，不要出现“存在问题”这种类似的说法。直接输出存在的问题是什么。
5.如果输出模板中存在不是“问题描述、根因分析、相关特征、解决方案”的其他部分，请对输出进行修改，使和输出模板保持一致。
6.具体内容必须和原文中的内容一模一样，不能生成，不能用文字总结。
7.如果没有具体内容或问题特征，不要填“无”，直接跳过，不用输出内容。
8.每个问题特征必须是独立的，如果有相似的问题特征，请把它们合并。

输出格式：
只输出主要问题的问题特征，格式必须和输出模板的格式保持一致。

信息：
输出模板：{example_feature}
问题的相关特征类型：{example_question}
原文：{passage}
目录：{catalog}
"""


template_process=[(
            "system",
            "You will be designed to extract entities from information and apply them to the knowledge graph."
            "You are an expert extraction algorithm."
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value."
            "Use the entity examples provided to help you find entities:{Examples_of_entities}"
            "The objective is to ensure the knowledge graph is straightforward and intelligible for broad use."
        ),  
        ("human", "Use the given format to extract information from the following input: {input}"),
        ("human","Numerical information is directly integrated as attributes of entities."
         "Avoid creating different nodes for dates or numbers, and instead attach them as attributes."
         "Don't use escape quotes in property values."
         "Use camel case for keys, such as' dateTime '."
         "Entity consistency: Ensures consistent identification of entities across various mentions or references."
         "Strict adherence to these guidelines is mandatory. Failure to comply will result in dismissal."),
        ("human", "The attributes of the entity must not be omitted.Attributes must be detailed."
         "The attributes of an entity must be sought based on the provided information without any omission, the attributes can be problem solutions, error messages, names, etc."
         "Opt for textual or comprehensible identifiers over numerical ones."),
        ("human", "Tip: Make sure to answer in the correct format.")
        ]

Examples_of_entities = [
    "非指针区域GC不扫描",
    "垃圾回收不会扫描不含指针的slice",
    "性能问题",
    "Concept",
    "Digitalsolution",
    "Division",
    "Entity",
    "Feature",
    "Fundinginitiative",
    "Initiative",
    "Link",
    "Location",
    "Organization",
    "Person",
    "Platform",
    "Policy",
    "Program"
    "Resource",
    "Role",
    "Schema",
    "Service",
    "Standard",
    "Technology",
    "Technologyplatform",
    "Technologystack",
    "Webframework",
    "Webresource",
    "Website"
]

template_relation=[(
    "system",
    "You are designed to analyze relationships between entities and chain entities together to build a knowledge graph."
    "When you analyze relationships between entities, be sure to be precise and detailed."
    "Analyze from the cluster, which entities are highly similar, and find out the connection between the entities with high similarity."
    "You must specify which entity, for example :'facebook'"
    "You must explain why they are relevant."
),
("human", "Use the given format to analyze relationships between entities:{clusters}"),
("human", "The relationships between entities must be detailed and precise. Be sure to include all relevant information."),
("human", "Tip: Make sure to answer in the correct format."
          "Describe in Chinese."
          "No entity should be left out."
          "Be sure to write what is the connection between the entities."
          "Cannot output attribute."
)
]

template_test="""
目的:总结出文章的摘要

步骤:
1.总结文章摘要

输出格式:
只输出文章的摘要

规则:
1.输出的内容必须是原文的摘要，不能生成，不能用文字总结

信息:
原文:{passage}
"""