# AI 写报告工具

这是一个基于硅基流动 `deepseek-ai/DeepSeek-V3.2` 的长报告生成工具，面向“资料导入 -> RAG 检索 -> 大纲规划 -> 分段生成 -> 审查复审 -> 多格式导出”的完整链路。

## 能力概览

- 输入层：解析 `PDF / DOCX / TXT / Markdown`，提取文档文本、元信息、用户画像和术语库
- 知识底座层：双维度分块、向量检索、重排与溯源，构建私有 RAG
- 框架层：生成多级大纲，锁定全文逻辑
- 生成层：按最小章节单元生成，并保持引用可追溯
- 校验层：检查事实性、结构、引用、字数、格式与 AI 写作痕迹
- 输出层：导出 `Markdown / HTML / DOCX / trace JSON / review JSON`

## 目录结构

- `资料/`：原始输入资料
- `key.env`：硅基流动 API Key
- `data/projects/`：项目、分块、向量、报告、溯源和审查结果

## 安装

```bash
pip install -r requirements.txt
```

## 启动 API

```bash
uvicorn app:app --reload
```

## CLI 示例


## key.env 格式

支持两种格式：

```text
sk-xxxxxxxx
```

或：

```text
SILICONFLOW_API_KEY=sk-xxxxxxxx
```
