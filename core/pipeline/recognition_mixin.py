"""统一识别结果收割逻辑。Pipeline 和 CameraPipeline 共用。"""


def process_recognition_harvest(
    worker, person_manager, recog_scheduler, frame_count
):
    """收割识别结果（非阻塞）—— Pipeline 和 CameraPipeline 公共逻辑。

    Args:
        worker: RecognitionWorker 实例
        person_manager: PersonManager 实例
        recog_scheduler: RecognitionScheduler 实例
        frame_count: 当前帧序号
    """
    results = worker.poll_results()
    for tid, name, sim, emb, qs in results:
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
        else:
            recog_scheduler.mark_completed(tid, frame_count)
