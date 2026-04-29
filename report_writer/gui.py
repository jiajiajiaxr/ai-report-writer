import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import time
import sys
import os
import logging
from collections import deque


class LogEmitter:
    """Thread-safe log emitter that collects logs for GUI display"""
    def __init__(self, max_logs=500):
        self.logs = deque(maxlen=max_logs)
        self.listeners = []
        self._lock = threading.Lock()

    def add_log(self, level: str, message: str):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = {"time": timestamp, "level": level, "message": message}
        with self._lock:
            self.logs.append(log_entry)
        for listener in self.listeners:
            listener(log_entry)

    def info(self, message):
        self.add_log("INFO", message)

    def warning(self, message):
        self.add_log("WARN", message)

    def error(self, message):
        self.add_log("ERROR", message)

    def debug(self, message):
        self.add_log("DEBUG", message)

    def get_all_logs(self):
        with self._lock:
            return list(self.logs)

    def subscribe(self, listener):
        self.listeners.append(listener)

    def unsubscribe(self, listener):
        if listener in self.listeners:
            self.listeners.remove(listener)


# Global log emitter
log_emitter = LogEmitter()


class LogEmitterHandler(logging.Handler):
    """Custom logging handler that emits logs to the LogEmitter"""
    def emit(self, record):
        level = record.levelname
        message = record.getMessage()
        log_emitter.add_log(level, message)


def setup_logging():
    """Configure logging to emit to the GUI"""
    handler = LogEmitterHandler()
    handler.setLevel(logging.INFO)
    # Add handler to key loggers
    for logger_name in ["report_writer.service", "report_writer.writer"]:
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


class ReportGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("报告生成器")
        self.root.geometry("900x650")

        self.is_running = False
        self.generation_thread = None
        self.log_emitter = log_emitter

        self._setup_ui()
        self._subscribe_logs()

    def _setup_ui(self):
        # Project ID input
        input_frame = ttk.Frame(self.root, padding="5")
        input_frame.pack(fill=tk.X)

        ttk.Label(input_frame, text="项目ID:").pack(side=tk.LEFT)
        self.project_id_var = tk.StringVar(value="proj-0fccb435d0")
        ttk.Entry(input_frame, textvariable=self.project_id_var, width=30).pack(side=tk.LEFT, padx=5)

        ttk.Label(input_frame, text="目标字数:").pack(side=tk.LEFT, padx=(10, 0))
        self.target_words_var = tk.StringVar(value="15000")
        ttk.Entry(input_frame, textvariable=self.target_words_var, width=10).pack(side=tk.LEFT, padx=5)

        self.start_button = ttk.Button(input_frame, text="开始生成", command=self._on_start)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = ttk.Button(input_frame, text="停止", command=self._on_stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)

        # Progress bar
        progress_frame = ttk.Frame(self.root, padding="5")
        progress_frame.pack(fill=tk.X)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5)

        self.current_section_var = tk.StringVar(value="")
        self.section_label = ttk.Label(progress_frame, textvariable=self.current_section_var)
        self.section_label.pack()

        # Log display
        log_frame = ttk.Frame(self.root, padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(log_frame, text="日志输出:").pack(anchor=tk.W)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for colors
        self.log_text.tag_config("INFO", foreground="black")
        self.log_text.tag_config("WARN", foreground="orange")
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("DEBUG", foreground="gray")

        # Status bar
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, padding="2")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _subscribe_logs(self):
        self.log_emitter.subscribe(self._on_log)

    def _on_log(self, log_entry):
        def append_log():
            tag = log_entry["level"]
            self.log_text.insert(tk.END, f"[{log_entry['time']}] [{log_entry['level']}] {log_entry['message']}\n", tag)
            self.log_text.see(tk.END)

        self.root.after(0, append_log)

    def _on_start(self):
        if self.is_running:
            return

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.log_text.delete("1.0", tk.END)
        self.status_var.set("生成中...")

        self.generation_thread = threading.Thread(target=self._run_generation, daemon=True)
        self.generation_thread.start()

    def _on_stop(self):
        self.is_running = False
        self.status_var.set("已停止")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def _run_generation(self):
        try:
            project_id = self.project_id_var.get()
            target_words = int(self.target_words_var.get())

            self.log_emitter.info(f"开始生成报告: 项目={project_id}, 目标={target_words}字")

            import sys
            import os
            # Set up API key from key.env file
            key_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "key.env")
            if os.path.exists(key_env_path):
                with open(key_env_path) as f:
                    for line in f:
                        if '=' in line and not os.environ.get(line.split('=')[0].strip()):
                            k, v = line.strip().split('=', 1)
                            os.environ[k] = v
            elif not os.environ.get('MINIMAX_API_KEY'):
                # Try parent directory
                key_env_path2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "key.env")
                if os.path.exists(key_env_path2):
                    with open(key_env_path2) as f:
                        for line in f:
                            if '=' in line and not os.environ.get(line.split('=')[0].strip()):
                                k, v = line.strip().split('=', 1)
                                os.environ[k] = v

            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from report_writer.config import load_settings
            from report_writer.service import ReportWriterService

            settings = load_settings()
            service = ReportWriterService(settings)

            self.log_emitter.info("开始逐章节生成...")

            report = service.generate_report(project_id, target_words=target_words)

            report_id = report.report_id
            markdown_path = report.markdown_path
            # Compute word count from markdown file
            import re
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
            word_count = len(re.findall(r'\S+', content))
            files = [report.markdown_path, report.html_path, report.docx_path]
            self.log_emitter.info(f"报告生成完成: {report_id}")
            self.log_emitter.info(f"字数: {word_count}")
            self.log_emitter.info(f"文件: {[report.markdown_path, report.html_path, report.docx_path]}")

            self.root.after(0, lambda rid=report_id: self.status_var.set(f"完成: {rid}"))
            self.root.after(0, lambda: self.progress_var.set(100))

        except Exception as e:
            import traceback
            err_msg = str(e)
            self.log_emitter.error(f"生成失败: {err_msg}")
            self.log_emitter.debug(traceback.format_exc())
            self.root.after(0, lambda msg=err_msg: self.status_var.set(f"错误: {msg}"))
        finally:
            self.root.after(0, lambda: self._on_stop())


def main():
    setup_logging()
    root = tk.Tk()
    app = ReportGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
