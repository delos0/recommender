"""
Music Recommendation System for Tune Duel Competition

This module implements the main recommendation function for the Tune Duel platform.
"""

from typing import List, Dict, Tuple, Any

def query(song_ratings: List[Dict[str, Any]], topk: int = 5) -> List[Tuple[str, str]]:
    """
    Generate recommendations based on user's song ratings.
    
    Args:
        song_ratings: List of dicts with 'song', 'rating', and 'spotify_id' keys
        topk: Number of recommendations to return
        
    Returns:
        List of (spotify_id, track_name) tuples
    """
    # TODO: Implement your recommendation algorithm here
    # This is a placeholder implementation
    
    # Simple fallback recommendations
    recommendations = [
        ("0yNttAVwMr39qyODHNIkrY", "Never Gonna Give You Up"),
        ("0VjIjW4GlUZAMYd2vXMi3b", "Blinding Lights"),
        ("7qiZfU4dY1lWllzX7mPBI3", "Shape of You"),
        ("2TIlqbIneP0ZY1O0EzYLlc", "Someone You Loved"),
        ("6UelLqGlWMcVH1E5c4H7lY", "Watermelon Sugar")
    ]
    
    return recommendations[:topk]

def test_recommender():
    """Test the recommender function."""
    test_ratings = [
        {'song': 'Test Song 1', 'rating': 5, 'spotify_id': 'test1'},
        {'song': 'Test Song 2', 'rating': 4, 'spotify_id': 'test2'}
    ]
    
    recommendations = query(test_ratings, topk=3)
    print(f"Generated {len(recommendations)} recommendations:")
    for i, (track_id, track_name) in enumerate(recommendations, 1):
        print(f"{i}. {track_name} (ID: {track_id})")

if __name__ == "__main__":
    test_recommender()
