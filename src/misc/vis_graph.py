import networkx as nx
import matplotlib.pyplot as plt

# 示例三元组列表：[(实体1, 关系, 实体2)]
knowledge_graph = [
            {
                "head": "cat",
                "relation": "playing",
                "tail": "screen"
            },
            {
                "head": "laptop",
                "relation": "has screen",
                "tail": "cat"
            },
            {
                "head": "laptop",
                "relation": "is white",
                "tail": "color"
            },
            {
                "head": "screen",
                "relation": "is on",
                "tail": "laptop"
            },
            {
                "head": "cat",
                "relation": "is on",
                "tail": "screen"
            },
            {
                "head": "laptop",
                "relation": "has cat",
                "tail": "playing"
            },
            {
                "head": "screen",
                "relation": "is part of",
                "tail": "laptop"
            },
            {
                "head": "cat",
                "relation": "is a part of",
                "tail": "laptop"
            },
            {
                "head": "laptop",
                "relation": "is a device",
                "tail": "computer"
            },
            {
                "head": "cat",
                "relation": "is a pet",
                "tail": "animal"
            }
        ]

# 创建有向图
G = nx.DiGraph()

# 添加边和节点到图中
for item in knowledge_graph:
    head, relation, tail = item['head'], item['relation'], item['tail']
    G.add_edge(head, tail, label=relation)

# 绘制图形
pos = nx.spring_layout(G)  # 使用 spring 布局
plt.figure(figsize=(10, 8))

# 绘制节点和边
nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=5000, font_size=12, font_weight='bold')
# edge_labels = nx.get_edge_attributes(G, 'label')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Knowledge Graph Visualization")
# plt.show()
plt.savefig('./graph_demo.png')
