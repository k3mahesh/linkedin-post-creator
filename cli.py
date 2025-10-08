from utils.linkedin_post_creator import generate_linked_post
from utils.linkedin_post_creator import write_post_to_file


if __name__ == "__main__":
    query = "Write a linkedin post of latest updates k8s"
    linked_post_generate = generate_linked_post(query=query)
    write_post_to_file(linked_post_generate)