# ReAct-Agent_RAG-Chat

## env

TAVILY_API_KEY=

REDIS_URL =

REDIS_TTL=

WEAVIATE_CLIENT_URL=

WEAVIATE_COLLECTION_NAME=

WEAVIATE_COLLECTION_PROPERTY=

LLM =

GROQ_API_KEY =

## Installation

Install the LangChain CLI if you haven't yet

```bash
poetry install
```

## Adding packages

```bash
# adding packages from
# https://github.com/langchain-ai/langchain/tree/master/templates
langchain app add $PROJECT_NAME

# adding custom GitHub repo packages
langchain app add --repo $OWNER/$REPO
# or with whole git string (supports other git providers):
# langchain app add git+https://github.com/hwchase17/chain-of-verification

# with a custom api mount point (defaults to `/{package_name}`)
langchain app add $PROJECT_NAME --api_path=/my/custom/path/rag
```

Note: you remove packages by their api path

```bash
langchain app remove my/custom/path/rag
```

## Setup LangSmith (Optional)

LangSmith will help us trace, monitor and debug LangChain applications.
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/).
If you don't have access, you can skip this section

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

## Launch LangServe

```bash
langchain serve
```

## Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

### Building the Image

To build the image, you simply:

```shell
docker build . -t my-langserve-app
```

If you tag your image with something other than `my-langserve-app`,
note it for use in the next step.
