"""
Logging configuration for the image captioning system.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import json
import structlog
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    enable_json_logging: bool = True,
    enable_console_logging: bool = True
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path (optional)
        log_dir: Directory for log files
        enable_json_logging: Enable structured JSON logging
        enable_console_logging: Enable console logging
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if enable_json_logging:
            console_formatter = structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(colors=True)
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_path = log_path / log_file
    else:
        file_path = log_path / f"image_captioning_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    if enable_json_logging:
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer()
        )
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Configure structlog
    if enable_json_logging:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


class RequestLogger:
    """Request logging utility."""
    
    def __init__(self, logger_name: str = "request_logger"):
        self.logger = structlog.get_logger(logger_name)
    
    def log_request(
        self,
        request_id: str,
        method: str,
        endpoint: str,
        user_id: Optional[str] = None,
        processing_time: Optional[float] = None,
        status_code: Optional[int] = None,
        error: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log API request."""
        log_data = {
            "request_id": request_id,
            "method": method,
            "endpoint": endpoint,
            "user_id": user_id,
            "processing_time": processing_time,
            "status_code": status_code,
            "error": error,
            **kwargs
        }
        
        if error:
            self.logger.error("API request failed", **log_data)
        else:
            self.logger.info("API request completed", **log_data)
    
    def log_model_inference(
        self,
        request_id: str,
        model_type: str,
        input_size: tuple,
        processing_time: float,
        success: bool,
        error: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log model inference."""
        log_data = {
            "request_id": request_id,
            "model_type": model_type,
            "input_size": input_size,
            "processing_time": processing_time,
            "success": success,
            "error": error,
            **kwargs
        }
        
        if error:
            self.logger.error("Model inference failed", **log_data)
        else:
            self.logger.info("Model inference completed", **log_data)
    
    def log_safety_check(
        self,
        request_id: str,
        check_type: str,
        result: dict,
        processing_time: float,
        **kwargs
    ) -> None:
        """Log safety check results."""
        log_data = {
            "request_id": request_id,
            "check_type": check_type,
            "result": result,
            "processing_time": processing_time,
            **kwargs
        }
        
        self.logger.info("Safety check completed", **log_data)


class MetricsLogger:
    """Metrics logging utility."""
    
    def __init__(self, logger_name: str = "metrics_logger"):
        self.logger = structlog.get_logger(logger_name)
    
    def log_model_metrics(
        self,
        model_type: str,
        metrics: dict,
        dataset: str,
        split: str,
        **kwargs
    ) -> None:
        """Log model evaluation metrics."""
        log_data = {
            "model_type": model_type,
            "metrics": metrics,
            "dataset": dataset,
            "split": split,
            **kwargs
        }
        
        self.logger.info("Model metrics", **log_data)
    
    def log_training_metrics(
        self,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float,
        **kwargs
    ) -> None:
        """Log training metrics."""
        log_data = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            **kwargs
        }
        
        self.logger.info("Training metrics", **log_data)
    
    def log_system_metrics(
        self,
        cpu_usage: float,
        memory_usage: float,
        gpu_usage: Optional[float] = None,
        gpu_memory: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log system metrics."""
        log_data = {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "gpu_usage": gpu_usage,
            "gpu_memory": gpu_memory,
            **kwargs
        }
        
        self.logger.info("System metrics", **log_data)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_exception(logger: structlog.BoundLogger, exception: Exception, context: dict = None) -> None:
    """Log an exception with context."""
    log_data = {
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "context": context or {}
    }
    
    logger.error("Exception occurred", **log_data, exc_info=True)


# Global logger instances
request_logger = RequestLogger()
metrics_logger = MetricsLogger()


if __name__ == "__main__":
    # Test logging configuration
    setup_logging(log_level="DEBUG")
    
    # Test structured logging
    logger = get_logger("test_logger")
    logger.info("Test message", extra_data={"key": "value"})
    
    # Test request logging
    request_logger.log_request(
        request_id="test-123",
        method="POST",
        endpoint="/v1/caption",
        processing_time=0.5,
        status_code=200
    )
    
    # Test metrics logging
    metrics_logger.log_model_metrics(
        model_type="baseline",
        metrics={"bleu_4": 0.25, "meteor": 0.30},
        dataset="coco",
        split="val"
    )
    
    print("Logging configuration test completed")
