semantic  ->  向量

## Agents
统一agent创建接口：create_agent

## 中间件
- 中间件系统 - 细粒度流程控制机制
- 中间件在每个步骤之前和之后会暴露钩子函数
- SummarizationMiddleware - 总结摘要

## 标准还输出
标准化输出: .content_blocks  统一为结构化内容块

结构化输出原生支持：将结构化输出（如返回JSON对象）直接集成到住循环中

## 命令行启动FastAPI
uvicorn main:app --reload