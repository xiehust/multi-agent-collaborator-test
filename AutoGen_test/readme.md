# 配置环境
conda create -n "py312" python=3.12
conda activate py312

# 下载
pip install -U "autogen-agentchat" "autogen-ext[openai,magentic-one]" nest_asyncio pyngrok yfinance google-search-results rich
# AutoGen Studio
pip install -U "autogenstudio"

# 启动
autogenstudio ui --port 8081 --host 0.0.0.0

# 使用litellm
```
docker run \
    -v $(pwd)/litellm_config.yaml:/app/config.yaml \
    -e AWS_REGION=us-east-1 \
    -p 4000:4000 \
    ghcr.io/berriai/litellm:main-latest \
    --config /app/config.yaml
```
