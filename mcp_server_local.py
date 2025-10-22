from mcp.server.fastmcp import FastMCP
from utils.linkedin_post_creator import generate_linked_post, write_post_to_file
from utils.linkedin_post_creator import LinkedPostRequest, LinkedPostResponse


mcp = FastMCP(name="linkedin-post-creator")


@mcp.tool()
def create_linked_post(req: LinkedPostRequest) -> LinkedPostResponse:
    """
    MCP Tool: Generates a linked blog post summary
    by combining Kubernetes + AWS blog feeds.
    """
    try:
        linked_post = generate_linked_post(req.query)
        write_post_to_file(linked_post)

        return LinkedPostResponse(
            status="success",
            message="Linked post generated successfully.",
            linked_post=linked_post,
        )
    except Exception as e:
        return LinkedPostResponse(
            status="error",
            message=f"Failed to generate linked post: {e}",
            linked_post=None,
        )

if __name__ == "__main__":
    mcp.run(transport='stdio')