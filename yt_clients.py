"""
YouTube Client Extractor Arguments for yt-dlp
"""

def hd_extractor_args(bgutil_http=None, bgutil_script=None):
    """
    Forces yt-dlp to use Android and Web clients to bypass strict blocking
    on datacenter IPs.
    """
    return {
        'youtube': {
            'player_client': ['android', 'web'],
            'player_skip': ['js', 'configs']
        }
    }

def fallback_extractor_args(bgutil_http=None, bgutil_script=None):
    """Fallback arguments if the primary HD extraction fails."""
    return {
        'youtube': {
            'player_client': ['ios', 'web']
        }
    }
