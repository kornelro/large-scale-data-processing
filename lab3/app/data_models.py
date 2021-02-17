from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class RedditSubmission:
    original_URL: str
    author_name: str
    subreddit_name: str
    subbreddit_display_name: str
    post_title: str
    post_title_embedding: List[np.array]
    post_text: str
    post_text_embedding: List[np.array]
    upvote_ratio: float
    up_votes_number: int
    comments_number: int
    nsfw: bool
    spoiler: bool
    original: bool
    distinguished: bool
    locked: bool
    fetch_time: float
