def get_user_input():
    """Get user input for title, subtitle, author name, and optional guidance."""
    title = input("Enter the book title: ")
    subtitle = input("Enter the story theme (subtitle): ")
    author_name = input("Enter the author name: ")
    guidance = input("Enter any additional guidance (optional): ")
    return title, subtitle, author_name, guidance
