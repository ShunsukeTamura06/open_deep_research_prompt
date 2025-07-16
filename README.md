# Open Deep Research (LangSmith-Free Version)

深層的なリサーチエージェントの完全なオープンソース実装です。このバージョンはLangSmithに依存せず、スタンドアロンで動作します。

## 特徴

- **複数の検索API対応**: Tavily、OpenAI Web Search、Anthropic Web Searchをサポート
- **モデルプロバイダー対応**: OpenAI、Anthropic、Google Vertex AIなど
- **階層的研究**: スーパーバイザーエージェントが複数の研究ユニットを並列実行
- **完全設定可能**: モデル、検索API、並行性レベルを柔軟に設定可能
- **LangSmith不要**: スタンドアロンで動作、外部追跡サービス不要

## インストール

```bash
git clone https://github.com/ShunsukeTamura06/open_deep_research_prompt.git
cd open_deep_research_prompt
pip install -r requirements.txt
```

## 環境設定

`.env`ファイルを作成し、必要なAPI キーを設定してください：

```bash
# OpenAI (必須：GPTモデル使用時)
OPENAI_API_KEY=your_openai_api_key

# Tavily (必須：Tavily検索使用時)
TAVILY_API_KEY=your_tavily_api_key

# Anthropic (オプション：Claudeモデル使用時)
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google (オプション：Geminiモデル使用時)
GOOGLE_API_KEY=your_google_api_key
```

## 基本的な使用方法

```python
import asyncio
from langgraph.checkpoint.memory import MemorySaver
from src.open_deep_research.deep_researcher import deep_researcher
import uuid

async def research_example():
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "allow_clarification": False,
            "search_api": "tavily",
            "research_model": "openai:gpt-4o-mini",
            "final_report_model": "openai:gpt-4o-mini"
        }
    }
    
    graph = deep_researcher.compile(checkpointer=MemorySaver())
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "AI推論市場の最新動向について調査してください"}]},
        config
    )
    
    print(result["final_report"])

asyncio.run(research_example())
```

## 設定オプション

### 検索API
- `tavily`: Tavily検索API（推奨）
- `openai`: OpenAI Web Search
- `anthropic`: Anthropic Web Search
- `none`: 検索なし

### モデル設定
- `research_model`: 研究実行用モデル
- `final_report_model`: 最終レポート生成用モデル
- `summarization_model`: ウェブページ要約用モデル
- `compression_model`: 研究結果圧縮用モデル

### 並行性制御
- `max_concurrent_research_units`: 並列実行する研究ユニット数（デフォルト: 5）
- `max_researcher_iterations`: 研究スーパーバイザーの最大反復回数（デフォルト: 3）
- `max_react_tool_calls`: 単一研究ステップでの最大ツール呼び出し回数（デフォルト: 5）

## サンプル実行

```bash
python examples/simple_research.py
```

## 元の実装との違い

- **LangSmith依存関係を削除**: すべてのLangSmith関連のコードを削除
- **評価システムを簡素化**: ローカル実行可能な評価システムに変更
- **スタンドアロン実行**: 外部追跡サービス不要
- **シンプルな設定**: 最小限の設定で動作

## ライセンス

MIT License

## 貢献

プルリクエストとイシューを歓迎します！

## サポート

問題が発生した場合は、GitHubのIssuesページで報告してください。