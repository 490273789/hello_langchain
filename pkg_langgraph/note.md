# LangGraph
LangGraph = langchain + 图编排 + 状态机
langchain的agent缺点： 
 - 黑盒，很难精细化控制和干预
 - 错误的思路循环几十次，浪费token
 - 行为不稳定

## 灵魂
State - 状态
Nodes - 节点：执行逻辑
Edge - 边：定义顺序
Graph - 图

# 高级特性
1. Streaming 流式处理
2. Persistence 状态持久化 
3. Time-Travel 时间回溯
4. Subgraphs 子图

## Streaming
langchain的流式输出，主要是处理回复长篇文字信息内容
langgraph流式输出，实时看流程状态