# VideoScope user journeys

1. As an analyst, I upload a supported video and see its processing progress without blocking the interface.
2. As an analyst, I search in natural language and receive ranked moments with exact start/end timestamps and evidence from speech, OCR, objects, scenes, or visual retrieval.
3. As an analyst, I open a result and the player seeks to the exact moment without duplicating the source video.
4. As an editor, I collect multiple moments, adjust their boundaries, and export one playable MP4 montage.
5. As an operator, I can see which ML providers are ready, unavailable, or waiting for configuration; one unavailable provider does not break the whole pipeline.
6. As a sports analyst, I search for a made three-point, two-point, or free throw and see the temporal and Qwen evidence without an unverified player number being presented as fact.
7. As a local user, uploaded files and indexes stay on this machine; only explicitly configured Roboflow Serverless or InternVideo integrations receive selected frames, never the complete source video.
8. As an analyst, I can close and restart the application without losing queued indexing work or allowing an abandoned worker to overwrite a newer attempt.
9. As an analyst, I can request cancellation, see that a running provider call may finish before the next checkpoint, and retry a failed or cancelled job with the exact persisted source and plan.
