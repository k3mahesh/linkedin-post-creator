from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
import logging
import requests
import feedparser
import os
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[logging.FileHandler("/tmp/linked_post_creator.log", mode='w'),
              logging.StreamHandler()]
              )

LLM_MODEL_URL = "http://localhost:11434"
LLM_MODEL = "gemma:2b"
LLM_MODEL_TEMP = 0.7
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHROMA_DIR = "./chroma_db"
LINKED_IN_POST_LOC= "/tmp" 

mcp = FastMCP(name="linkedin-post-creator")

BLOG_FEEDS_URL = {
    "kubernetes": "https://kubernetes.io/feed.xml",
    "aws": "https://aws.amazon.com/blogs/aws/feed/",
    "azure": "https://azure.microsoft.com/en-us/blog/feed/",
}

def fetch_feed(url: str) -> List[Dict]:
    """Fetch an RSS/Atom feed and return parsed entries."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        feed = feedparser.parse(response.text)
        logging.info("Getting Feeds")
        return feed.entries
    except Exception as e:
        logging.exception((f"⚠️ Failed to fetch {url}: {e}"))
        return []

def entries_to_documents(entries: List[Dict]) -> List[Document]:
    """Convert feed entries into LangChain Document objects."""
    docs = []
    for entry in entries:
        content = entry.get("summary", "") or entry.get("description", "")
        title = entry.get("title", "Untitled")
        link = entry.get("link", "")
        published = entry.get("published_parsed", None)

        # Convert published to datetime (if available)
        published_dt = None
        if published:
            published_dt = datetime(*published[:6])
            published_str = published_dt.isoformat()

        docs.append(
            Document(
                page_content=f"Title: {title}\n\nContent: {content}\n\nLink: {link}",
                metadata={
                    "source": link, 
                    "title": title, 
                    "published": published_str
                    }
            )
        )
    logging.info("Created Documents with Title, Summary, Link, and Published Date")
    return docs

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for better LLM/vector DB handling."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

def build_vector_db(documents: List[Document]) -> Chroma:
    """Create a Chroma vector DB from documents and persist it locally."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(CHROMA_DIR):
        logging.info("Removing old Chroma database...")
        import shutil
        shutil.rmtree(CHROMA_DIR)
    
    logging.info("Building new Chroma vector DB...")
    return Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_DIR)


def setup_chain() -> LLMChain:
    """Setup LangChain with Ollama LLM and prompt template."""
    llm = ChatOllama(
        base_url=LLM_MODEL_URL, 
        model=LLM_MODEL,
        temperature=0.7,
    )

    prompt = PromptTemplate(
        input_variables=["context", "links"],
        template="""
You are an expert cloud, Kubernetes, AI, and Technical blog writer.
Task:
1. Create a **short and catchy headding** for the linked post (max 10-12 words).
2. Rewrite the blog into a LinkedIn post(~1000 words).
3. Make it insightful, engaging, and useful for a professional audience.
4. Use bullet points, headings, and subheadings to organize the content.
5. Include a clear call-to-action (CTA) at the end of the post.
6. Make sure the post is visually appealing.
7. Add relevant emoji (but only tech/professional context).
8. Add relvent SEO friendly hastags.
9. At the bottom, include **all referenced blog links**.


Context from the Blogs:
{context}

Referenced Blog Links:
{links}

Format your response like this:
Heading: <your generated heading>

Post:
<your generated linked post>

Blog Links:
<list of blog links>
"""
    )

    return LLMChain(llm=llm, prompt=prompt)

def get_blog_entries_list(query: str) -> List[Dict]:
    """Get a list of tech-related blog links based on a query."""
    entries = []
    k8s_keywords = ["kubernetes", "k8s", "kube", "containers", "container"]
    aws_keywords = ["aws", "amazon", "ec2", "eks", "s3"]
    azure_keywords = ["azure", "microsoft"]
    hashicorp_keywords = ["hashicorp", "terraform", "vault"]

    if any(word in query.lower().strip() for word in k8s_keywords):
        print("url", BLOG_FEEDS_URL["kubernetes"])
        entries.extend(fetch_feed(BLOG_FEEDS_URL["kubernetes"]))
    elif any(word in query.lower().strip() for word in aws_keywords):
        print("url", BLOG_FEEDS_URL["aws"])
        entries.extend(fetch_feed(BLOG_FEEDS_URL["aws"]))
    elif any(word in query.lower().strip() for word in azure_keywords):
        print("url", BLOG_FEEDS_URL["azure"])
        entries.extend(fetch_feed(BLOG_FEEDS_URL["azure"]))
    elif any(word in query.lower().strip() for word in hashicorp_keywords):
        entries.extend(fetch_feed(BLOG_FEEDS_URL["hashicorp"]))
    else:
        for loc, url in BLOG_FEEDS_URL.items():
            entries.extend(fetch_feed(url))
    
    return entries

def generate_linked_post(query: str) -> str:
    """End-to-end pipeline: fetch blogs, store in DB, query, and summarize."""
    # Fetch blogs
    all_entries = get_blog_entries_list(query)

    # Convert → Chunk → Vector DB
    documents = entries_to_documents(all_entries)
    chunks = chunk_documents(documents)
    vector_db = build_vector_db(chunks)

    # Setup chain
    chain = setup_chain()

    latest_blog_fetch_trigger = ["latest", "recent"]

    if not query or any(word in query.lower().strip() for word in latest_blog_fetch_trigger) :
        latest_doc = max(documents, key=lambda d: d.metadata.get("published") or datetime.min)
        context = latest_doc.page_content
        links = latest_doc.metadata["source"]
        logging.info(f"Picked latest blog: {latest_doc.metadata['title']}")
    else:
        results = vector_db.similarity_search(query, k=4)
        if not results:
            return "No relevant content found to generate a LinkedIn post."
        context = "\n\n".join([doc.page_content for doc in results])
        links = "\n".join({doc.metadata["source"] for doc in results if doc.metadata.get("source")})
        logging.info(f"Generated post using topic search: {query}")

    return chain.run(context=context, links=links)

class LinkedPostRequest(BaseModel):
    query: str = Field(..., description="Topic to generate a linked post about.")

class LinkedPostResponse(BaseModel):
    status: Literal["success", "error"]
    message: str
    linked_post: str | None = None

def write_post_to_file(linked_in_post: str) -> bool:
    """Post Writor to File"""
    try:
        if linked_in_post:
            with open(LINKED_IN_POST_LOC + "/linked_in_post", 'w', encoding="utf-8") as f:
                f.write(linked_in_post)
            logging.info("LinkedIn Post has been written to file")
            return True
        logging.error("Response is Empty from Model")
        return False
    except Exception as e:
        logging.exception(e)
        return False


# @mcp.tool()
# def create_linked_post(req: LinkedPostRequest) -> LinkedPostResponse:
#     """
#     MCP Tool: Generates a linked blog post summary
#     by combining Kubernetes + AWS blog feeds.
#     """
#     try:
#         # trigger_words = ["linked", "post", "k8s", "aws", "kubernetes"]
#         # if any(word in req for word in trigger_words):
#         linked_post = generate_linked_post(req.query)
#         logging.info("Linked Post has been generated")
#         write_post_to_file(linked_post)

#         return LinkedPostResponse(
#             status="success",
#             message="Linked post generated successfully.",
#             linked_post=linked_post,
#         )
#     except Exception as e:
#         return LinkedPostResponse(
#             status="error",
#             message=f"Failed to generate linked post: {e}",
#             linked_post=None,
#         )
    
if __name__ == "__main__":
    # mcp.run(transport='stdio')
    query = "Write a linkedin post of latest updates k8s"
    linked_post_generate = generate_linked_post(query=query)
    write_post_to_file(linked_post_generate)
