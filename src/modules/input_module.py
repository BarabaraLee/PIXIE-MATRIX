def get_user_input():
    """Get user input for story theme, author name, and optional guidance."""
    story_theme = input("Enter the story theme (e.g., rainforest cute animals): ")
    author_name = input("Enter the author name: ")
    guidance = input("Enter any additional guidance (optional): ")
    return story_theme, author_name, guidance
