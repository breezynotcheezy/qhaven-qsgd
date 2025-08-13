"""
Async job manager, retry/failover logic, and batching for quantum jobs.
"""
import asyncio

class Scheduler:
    def __init__(self):
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
    def batch_jobs(self, jobs, max_parallel=2):
        # Schedule jobs in parallel, return when all done
        async def run_all():
            return await asyncio.gather(*(self.loop.run_in_executor(None, job) for job in jobs))
        return self.loop.run_until_complete(run_all())
    def retry(self, func, max_retries=2, backoff=2):
        # Idempotent retry, exponential backoff
        for i in range(max_retries+1):
            try:
                return func()
            except Exception as e:
                if i == max_retries:
                    raise
                import time
                time.sleep(backoff ** i)
