"""
YouTube Client Extractor Arguments for yt-dlp
"""

def hd_extractor_args(bgutil_http=None, bgutil_script=None):
    """
    Forces yt-dlp to use Android, iOS, and Web clients to bypass strict blocking
    on datacenter IPs. Integrates PO Token providers if configured.
    """
    yt_opts = {
        'player_client': ['android', 'ios', 'web'],
        'player_skip': ['js', 'configs']
    }
    
    # Inject Proof of Origin (PO) Token provider if environment supplies one
    if bgutil_http:
        yt_opts['po_token'] = [f"web+{bgutil_http}"]
    elif bgutil_script:
        yt_opts['po_token'] = [f"web+{bgutil_script}"]
        
    return {'youtube': yt_opts}

def fallback_extractor_args(bgutil_http=None, bgutil_script=None):
    """Fallback arguments utilizing different client hierarchies if primary fails."""
    yt_opts = {
        'player_client': ['ios', 'android', 'web']
    }
    
    if bgutil_http:
        yt_opts['po_token'] = [f"web+{bgutil_http}"]
    elif bgutil_script:
        yt_opts['po_token'] = [f"web+{bgutil_script}"]
        
    return {'youtube': yt_opts}

class NotASingleVideo(Exception):
    """Custom exception raised when a URL points to a playlist or unsupported entity."""
    pass

def youtube_non_video_reason(url):
    """
    Evaluates if a YouTube URL points to a non-video entity (like a channel or playlist).
    Returns a string reason if it is NOT a single video, otherwise returns None.
    """
    if 'playlist?list=' in url:
        return "a playlist"
    if '/c/' in url or '/channel/' in url or '/@' in url or '/user/' in url:
        return "a channel"
    return None
