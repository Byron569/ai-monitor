"""统一识别结果收割逻辑。Pipeline 和 CameraPipeline 共用。"""


def process_recognition_harvest(
    worker, person_manager, recog_scheduler, frame_count, monitor=None
):
    results = worker.poll_results()
    for tid, name, sim, emb, qs, *rest in results:
        latency_ms = rest[0] if rest else 0.0
        if name != "Unknown" and emb is not None:
            person_manager.identify(tid, name, emb)
            recog_scheduler.mark_identified(tid, frame_count, name)
            person_manager.cache_embedding(tid, name, emb)
        elif emb is not None:
            cached_name, _ = person_manager.find_cached_identity(emb)
            if cached_name is not None:
                person_manager.identify(tid, cached_name, emb)
                recog_scheduler.mark_identified(tid, frame_count, cached_name)
                person_manager.cache_embedding(tid, cached_name, emb)
            else:
                recog_scheduler.mark_completed(tid, frame_count)
        if monitor is not None and latency_ms > 0:
            monitor.track_latency("recognize", latency_ms)
