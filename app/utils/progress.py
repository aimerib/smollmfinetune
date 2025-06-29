"""
Production-Grade Progress Indicators

Provides beautiful progress indicators, spinners, and progress tracking
for long-running operations in the Streamlit application.
"""

import streamlit as st
import time
from typing import Optional, Callable, Any, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading


@dataclass
class ProgressConfig:
    """Configuration for progress indicators"""
    show_percentage: bool = True
    show_eta: bool = True
    show_spinner: bool = True
    auto_close: bool = True
    close_delay: float = 1.0
    update_interval: float = 0.1


class ProgressTracker:
    """Advanced progress tracking with ETA calculation"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_update = self.start_time
        self.update_history = []
    
    def update(self, increment: int = 1):
        """Update progress and calculate ETA"""
        self.current += increment
        now = time.time()
        
        # Store update history for better ETA calculation
        self.update_history.append((now, self.current))
        
        # Keep only last 10 updates for ETA calculation
        if len(self.update_history) > 10:
            self.update_history.pop(0)
        
        self.last_update = now
    
    def get_progress(self) -> float:
        """Get current progress as percentage (0.0 to 1.0)"""
        return min(self.current / self.total, 1.0) if self.total > 0 else 0.0
    
    def get_eta(self) -> Optional[timedelta]:
        """Calculate estimated time remaining"""
        if len(self.update_history) < 2 or self.current == 0:
            return None
        
        # Calculate rate based on recent history
        recent_time, recent_progress = self.update_history[0]
        current_time, current_progress = self.update_history[-1]
        
        time_diff = current_time - recent_time
        progress_diff = current_progress - recent_progress
        
        if time_diff <= 0 or progress_diff <= 0:
            return None
        
        rate = progress_diff / time_diff  # items per second
        remaining_items = self.total - self.current
        
        if rate > 0:
            eta_seconds = remaining_items / rate
            return timedelta(seconds=eta_seconds)
        
        return None
    
    def format_eta(self) -> str:
        """Format ETA for display"""
        eta = self.get_eta()
        if eta is None:
            return "Calculating..."
        
        total_seconds = int(eta.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def is_complete(self) -> bool:
        """Check if progress is complete"""
        return self.current >= self.total


@contextmanager
def progress_spinner(message: str = "Processing...", success_message: str = "Complete!"):
    """Context manager for displaying a spinner during operations"""
    placeholder = st.empty()
    
    try:
        with placeholder.container():
            with st.spinner(message):
                yield
        
        # Show success message briefly
        placeholder.success(f"✅ {success_message}")
        time.sleep(1)
        placeholder.empty()
        
    except Exception as e:
        placeholder.error(f"❌ Operation failed: {str(e)}")
        time.sleep(2)
        placeholder.empty()
        raise


@contextmanager 
def progress_bar(total: int, description: str = "Processing", 
                config: Optional[ProgressConfig] = None):
    """Context manager for displaying a progress bar during operations"""
    if config is None:
        config = ProgressConfig()
    
    tracker = ProgressTracker(total, description)
    placeholder = st.empty()
    
    def update_display():
        progress = tracker.get_progress()
        
        with placeholder.container():
            # Main progress bar
            st.progress(progress)
            
            # Progress details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if config.show_percentage:
                    st.caption(f"Progress: {progress:.1%}")
            
            with col2:
                st.caption(f"Items: {tracker.current}/{tracker.total}")
            
            with col3:
                if config.show_eta and not tracker.is_complete():
                    st.caption(f"ETA: {tracker.format_eta()}")
                elif tracker.is_complete():
                    elapsed = time.time() - tracker.start_time
                    st.caption(f"Completed in {elapsed:.1f}s")
    
    try:
        update_display()
        yield tracker
        
        # Final update
        if not tracker.is_complete():
            tracker.current = tracker.total
        update_display()
        
        # Auto-close after delay
        if config.auto_close:
            time.sleep(config.close_delay)
            placeholder.empty()
            
    except Exception as e:
        placeholder.error(f"❌ {description} failed: {str(e)}")
        time.sleep(2)
        placeholder.empty()
        raise


class AsyncProgressTracker:
    """Thread-safe progress tracker for async operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.tracker = ProgressTracker(total, description)
        self.placeholder = st.empty()
        self.config = ProgressConfig()
        self.lock = threading.Lock()
        self.running = True
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def update(self, increment: int = 1):
        """Thread-safe progress update"""
        with self.lock:
            self.tracker.update(increment)
    
    def set_description(self, description: str):
        """Update the description"""
        with self.lock:
            self.tracker.description = description
    
    def finish(self, success_message: str = "Complete!"):
        """Finish progress tracking"""
        with self.lock:
            self.running = False
            self.tracker.current = self.tracker.total
        
        # Wait for update thread to finish
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        # Show final message
        self.placeholder.success(f"✅ {success_message}")
        time.sleep(1)
        self.placeholder.empty()
    
    def error(self, error_message: str):
        """Show error and stop tracking"""
        with self.lock:
            self.running = False
        
        self.placeholder.error(f"❌ {error_message}")
        time.sleep(2)
        self.placeholder.empty()
    
    def _update_loop(self):
        """Background thread for updating display"""
        while self.running:
            with self.lock:
                if not self.running:
                    break
                
                progress = self.tracker.get_progress()
                
                with self.placeholder.container():
                    st.progress(progress)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.caption(f"Progress: {progress:.1%}")
                    
                    with col2:
                        st.caption(f"Items: {self.tracker.current}/{self.tracker.total}")
                    
                    with col3:
                        if not self.tracker.is_complete():
                            st.caption(f"ETA: {self.tracker.format_eta()}")
            
            time.sleep(self.config.update_interval)


def show_loading_animation(message: str = "Loading...", duration: float = 2.0):
    """Show a beautiful loading animation"""
    placeholder = st.empty()
    
    # CSS for loading animation
    st.markdown("""
    <style>
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #e2e8f0;
        border-top: 4px solid #6366f1;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        color: #64748b;
        font-size: 1.1rem;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show loading animation
    placeholder.markdown(f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">{message}</div>
    </div>
    """, unsafe_allow_html=True)
    
    time.sleep(duration)
    placeholder.empty()


def step_progress(steps: list, current_step: int, description: str = ""):
    """Show a step-based progress indicator"""
    total_steps = len(steps)
    
    # Progress bar for overall completion
    progress = (current_step) / total_steps if total_steps > 0 else 0
    st.progress(progress, f"Step {current_step + 1} of {total_steps}")
    
    # Step indicators
    cols = st.columns(total_steps)
    for i, (col, step_name) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                st.success(f"✅ {step_name}")
            elif i == current_step:
                st.info(f"🔄 {step_name}")
            else:
                st.caption(f"⏳ {step_name}")
    
    if description:
        st.caption(description)


def progress_with_substeps(main_step: str, substeps: list, 
                          current_substep: int, main_progress: float = None):
    """Show progress with main step and substeps"""
    st.subheader(f"🔄 {main_step}")
    
    if main_progress is not None:
        st.progress(main_progress, f"Overall Progress: {main_progress:.1%}")
    
    # Substep progress
    substep_progress = current_substep / len(substeps) if substeps else 0
    st.progress(substep_progress, f"Current Step Progress")
    
    # Substep list
    for i, substep in enumerate(substeps):
        if i < current_substep:
            st.success(f"✅ {substep}")
        elif i == current_substep:
            st.info(f"🔄 {substep}")
        else:
            st.caption(f"⏳ {substep}")


# Convenience functions for common operations
def dataset_generation_progress(total_samples: int, description: str = "Generating dataset"):
    """Specialized progress tracker for dataset generation"""
    return progress_bar(
        total=total_samples,
        description=description,
        config=ProgressConfig(
            show_percentage=True,
            show_eta=True,
            auto_close=False  # Let the caller handle cleanup
        )
    )


def training_progress(total_steps: int, description: str = "Training model"):
    """Specialized progress tracker for model training"""
    return AsyncProgressTracker(total_steps, description)


def model_loading_progress(model_name: str):
    """Show progress for model loading"""
    return progress_spinner(
        message=f"Loading {model_name}...",
        success_message=f"{model_name} loaded successfully!"
    ) 