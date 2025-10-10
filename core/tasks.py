import asyncio
import threading
import time
import logging
from queue import Queue, Empty
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class Task:
    id: str
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    timeout: Optional[float] = None
    created_at: float = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.created_at is None:
            self.created_at = time.time()

class BackgroundTaskQueue:
    """Non-blocking task queue for camera operations"""
    
    def __init__(self, max_workers=5, max_queue_size=100):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.task_queue = Queue(maxsize=max_queue_size)
        self.tasks: Dict[str, Task] = {}
        self.workers = []
        self.running = False
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """Start background workers"""
        if self.running:
            return
            
        self.running = True
        
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker,
                name=f"CameraWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        self.logger.info(f"Started {self.max_workers} background workers")
    
    def stop(self):
        """Stop background workers"""
        self.running = False
        
        # Add poison pills to stop workers
        for _ in range(self.max_workers):
            try:
                self.task_queue.put(None, timeout=1)
            except:
                pass
                
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
            
        self.workers.clear()
        self.logger.info("Stopped background workers")
    
    def submit_task(self, func: Callable, *args, timeout: float = 30, **kwargs) -> str:
        """Submit a task to the queue"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            timeout=timeout
        )
        
        try:
            self.task_queue.put(task, timeout=1)
            self.tasks[task_id] = task
            self.logger.debug(f"Submitted task {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit task: {str(e)}")
            raise Exception("Task queue is full")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and result"""
        task = self.tasks.get(task_id)
        if not task:
            return None
            
        return {
            "id": task.id,
            "status": task.status.value,
            "created_at": task.created_at,
            "result": task.result,
            "error": task.error
        }
    
    def wait_for_task(self, task_id: str, timeout: float = 30) -> Optional[Dict[str, Any]]:
        """Wait for task completion"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task = self.tasks.get(task_id)
            if not task:
                return None
                
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT]:
                return self.get_task_status(task_id)
                
            time.sleep(0.1)
            
        return {"id": task_id, "status": "timeout", "error": "Wait timeout exceeded"}
    
    def _worker(self):
        """Background worker thread"""
        worker_name = threading.current_thread().name
        self.logger.debug(f"Worker {worker_name} started")
        
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                
                # Poison pill to stop worker
                if task is None:
                    break
                    
                self._execute_task(task)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {str(e)}")
                
        self.logger.debug(f"Worker {worker_name} stopped")
    
    def _execute_task(self, task: Task):
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        start_time = time.time()
        
        try:
            self.logger.debug(f"Executing task {task.id}")
            
            if task.timeout:
                # Execute with timeout using threading
                result_container = {}
                error_container = {}
                
                def target():
                    try:
                        result_container['result'] = task.func(*task.args, **task.kwargs)
                    except Exception as e:
                        error_container['error'] = str(e)
                
                thread = threading.Thread(target=target)
                thread.start()
                thread.join(timeout=task.timeout)
                
                if thread.is_alive():
                    task.status = TaskStatus.TIMEOUT
                    task.error = f"Task timed out after {task.timeout}s"
                    self.logger.warning(f"Task {task.id} timed out")
                    return
                
                if 'error' in error_container:
                    raise Exception(error_container['error'])
                    
                task.result = result_container.get('result')
            else:
                # Execute without timeout
                task.result = task.func(*task.args, **task.kwargs)
            
            task.status = TaskStatus.COMPLETED
            execution_time = time.time() - start_time
            self.logger.debug(f"Task {task.id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.logger.error(f"Task {task.id} failed: {str(e)}")
    
    def cleanup_old_tasks(self, max_age: float = 3600):
        """Remove old completed tasks"""
        current_time = time.time()
        to_remove = []
        
        for task_id, task in self.tasks.items():
            if (current_time - task.created_at > max_age and 
                task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT]):
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
            
        if to_remove:
            self.logger.debug(f"Cleaned up {len(to_remove)} old tasks")

class CameraTaskManager:
    """High-level manager for camera-related background tasks"""
    
    def __init__(self):
        self.task_queue = BackgroundTaskQueue(max_workers=3)
        self.task_queue.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True
        )
        self.cleanup_thread.start()
    
    def test_camera_connection_async(self, camera_id: int, timeout: float = 15) -> str:
        """Submit camera connection test as background task"""
        from .async_rtsp_manager import async_rtsp_manager
        
        def test_func():
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    async_rtsp_manager.test_camera_connection_async(camera_id)
                )
            finally:
                loop.close()
        
        return self.task_queue.submit_task(test_func, timeout=timeout)
    
    def get_camera_status_async(self, camera_id: int, timeout: float = 10) -> str:
        """Submit camera status check as background task"""
        from .async_rtsp_manager import async_rtsp_manager
        
        def status_func():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    async_rtsp_manager.get_camera_status_async(camera_id)
                )
            finally:
                loop.close()
        
        return self.task_queue.submit_task(status_func, timeout=timeout)
    
    def get_task_result(self, task_id: str, wait: bool = False, timeout: float = 30):
        """Get task result, optionally waiting for completion"""
        if wait:
            return self.task_queue.wait_for_task(task_id, timeout)
        else:
            return self.task_queue.get_task_status(task_id)
    
    def _cleanup_worker(self):
        """Periodic cleanup of old tasks"""
        while True:
            try:
                time.sleep(300)  # Clean up every 5 minutes
                self.task_queue.cleanup_old_tasks()
            except Exception as e:
                logger.error(f"Cleanup worker error: {str(e)}")

# Global task manager instance
camera_task_manager = CameraTaskManager()