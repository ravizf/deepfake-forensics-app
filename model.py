from forensics import analyze_media_file, infer_media_type


def analyze_file(filepath):
    media_type = infer_media_type(filepath)
    return analyze_media_file(filepath, media_type, "artifacts/heatmaps")
