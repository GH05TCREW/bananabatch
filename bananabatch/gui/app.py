"""BananaBatch Streamlit GUI Application.

This module provides a web-based dashboard for BananaBatch,
built with Streamlit for easy local browser access.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from bananabatch.core.engine import BatchEditProcessor, BatchProcessor, FileManager
from bananabatch.core.models import (
    BatchProgress,
    EditJobConfig,
    EditType,
    ImageResult,
    JobConfig,
    JobStatus,
    ModelName,
)
from bananabatch.providers.gemini import GeminiProvider

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="BananaBatch",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")
    if "results" not in st.session_state:
        st.session_state.results = []
    if "edit_results" not in st.session_state:
        st.session_state.edit_results = []
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "progress" not in st.session_state:
        st.session_state.progress = BatchProgress()
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = None
    if "edit_output_dir" not in st.session_state:
        st.session_state.edit_output_dir = None


def render_sidebar() -> dict[str, Any]:
    """Render the sidebar configuration panel.

    Returns:
        Dictionary of configuration values.
    """
    st.sidebar.title("🍌 BananaBatch")
    st.sidebar.markdown("*Batch Image Generation*")
    st.sidebar.divider()

    # API Key configuration
    st.sidebar.subheader("API Configuration")
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Enter your Gemini API key or set GEMINI_API_KEY env var",
    )
    st.session_state.api_key = api_key

    st.sidebar.divider()

    # Model selection
    st.sidebar.subheader("Model Settings")
    model = st.sidebar.selectbox(
        "Model",
        options=[m.value for m in ModelName],
        index=0,
        help="Select the Gemini model for image generation",
    )

    # Concurrency settings
    max_workers = st.sidebar.slider(
        "Max Concurrent Workers",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of images to generate simultaneously",
    )

    st.sidebar.divider()

    # Output settings
    st.sidebar.subheader("Output Settings")
    output_dir = st.sidebar.text_input(
        "Output Directory",
        value="outputs",
        help="Directory to save generated images",
    )

    return {
        "api_key": api_key,
        "model": model,
        "max_workers": max_workers,
        "output_dir": output_dir,
    }


def render_input_section() -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Render the input data section.

    Returns:
        Tuple of (DataFrame of input data, prompt template string).
    """
    st.header("Input Configuration")

    col1, col2 = st.columns([2, 1])

    # File upload column
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Input File",
            type=["csv", "json"],
            help="Upload a CSV or JSON file with prompts",
        )

    # Template column - aligned at top
    with col2:
        template = st.text_area(
            "Jinja2 Template (optional)",
            placeholder="A photo of {{ item }} in {{ style }} style",
            help="Use {{ column_name }} to insert values from your data",
            height=68,
        )

    # Handle file upload results below the columns
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)

            st.success(f"Loaded {len(df)} records from {uploaded_file.name}")

            # Show preview
            with st.expander("Preview Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            return df, template if template else None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None, None

    return df, template if template else None


def render_manual_input() -> Optional[pd.DataFrame]:
    """Render manual prompt input section.

    Returns:
        DataFrame of manually entered prompts.
    """
    st.subheader("Or Enter Prompts Manually")

    prompts_text = st.text_area(
        "Enter prompts (one per line)",
        placeholder="A serene mountain landscape at sunset\nA cute robot playing guitar\nAbstract colorful geometric patterns",
        height=150,
    )

    if prompts_text.strip():
        prompts = [p.strip() for p in prompts_text.strip().split("\n") if p.strip()]
        if prompts:
            df = pd.DataFrame({"prompt": prompts})
            st.info(f"{len(prompts)} prompts ready")
            return df

    return None


def render_progress_section(progress: BatchProgress) -> None:
    """Render the progress section.

    Args:
        progress: Current batch progress.
    """
    st.header("Progress")

    # Progress bar
    if progress.total > 0:
        completed_total = progress.completed + progress.failed
        st.progress(
            completed_total / progress.total,
            text=f"Processing: {completed_total}/{progress.total}",
        )

    # Stats columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total", progress.total)
    with col2:
        st.metric("Completed", progress.completed)
    with col3:
        st.metric("Failed", progress.failed)
    with col4:
        st.metric("Success Rate", f"{progress.success_rate:.1f}%")


def render_results_section(results: list[ImageResult], output_dir: Optional[Path]) -> None:
    """Render the results section.

    Args:
        results: List of image results.
        output_dir: Path to output directory.
    """
    if not results:
        return

    st.header("Results")

    if output_dir:
        st.info(f"Output directory: `{output_dir}`")

    # Filter controls
    col1, col2 = st.columns([1, 3])
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All", "Completed", "Failed"],
        )

    # Filter results
    filtered_results = results
    if status_filter == "Completed":
        filtered_results = [r for r in results if r.status == JobStatus.COMPLETED]
    elif status_filter == "Failed":
        filtered_results = [r for r in results if r.status == JobStatus.FAILED]

    # Display results in a grid
    cols = st.columns(3)

    for i, result in enumerate(filtered_results):
        with cols[i % 3]:
            with st.container(border=True):
                if result.status == JobStatus.COMPLETED and result.output_path:
                    # Try to display the image
                    try:
                        if result.output_path.exists():
                            st.image(
                                str(result.output_path),
                                caption=result.output_path.name,
                                use_container_width=True,
                            )
                    except Exception:
                        st.info(f"{result.output_path.name}")

                    st.caption(f"{result.generation_time_ms:.0f}ms")
                else:
                    st.error(f"Failed: {result.error_message or 'Unknown error'}")

                # Show prompt in expander
                with st.expander("View Prompt"):
                    st.text(result.prompt_used[:200])


async def run_batch_generation(
    df: pd.DataFrame,
    config: dict[str, Any],
    template: Optional[str],
    progress_placeholder: Any,
    results_placeholder: Any,
) -> list[ImageResult]:
    """Run the batch generation process.

    Args:
        df: DataFrame with input data.
        config: Configuration dictionary.
        template: Optional prompt template.
        progress_placeholder: Streamlit placeholder for progress.
        results_placeholder: Streamlit placeholder for results.

    Returns:
        List of ImageResult objects.
    """
    # Save DataFrame to temporary file
    temp_file = Path("temp_input.csv")
    df.to_csv(temp_file, index=False)

    try:
        # Create job config
        job_config = JobConfig(
            input_file=temp_file,
            output_dir=Path(config["output_dir"]),
            prompt_template=template,
            model=ModelName(config["model"]),
            max_workers=config["max_workers"],
            api_key=config["api_key"],
        )

        # Initialize components
        provider = GeminiProvider(
            api_key=config["api_key"],
            default_model=config["model"],
        )
        file_manager = FileManager(job_config.output_dir)
        processor = BatchProcessor(
            provider=provider,
            config=job_config,
            file_manager=file_manager,
        )

        # Process and update progress
        results = []

        async for result in processor.process_stream():
            results.append(result)
            st.session_state.results = results
            st.session_state.progress = processor.progress
            st.session_state.output_dir = file_manager.output_dir

            # Update progress display
            with progress_placeholder.container():
                render_progress_section(processor.progress)

        return results

    finally:
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()


def render_edit_input_section() -> tuple[Optional[pd.DataFrame], Optional[str], list[Any]]:
    """Render the edit input data section.

    Returns:
        Tuple of (DataFrame of edit instructions, template string, uploaded images).
    """
    st.header("Edit Configuration")

    col1, col2 = st.columns([2, 1])

    # File upload column
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Edit Instructions",
            type=["csv", "json"],
            help="Upload a CSV or JSON file with edit instructions",
            key="edit_instructions_upload",
        )

    # Template column
    with col2:
        template = st.text_area(
            "Edit Prompt Template (optional)",
            placeholder="Add {{ defect_type }} damage to this product image",
            help="Use {{ column_name }} to insert values from your data",
            height=68,
            key="edit_template",
        )

    # Base images upload
    st.subheader("Base Images")
    uploaded_images = st.file_uploader(
        "Upload Base Images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Upload the images to be edited. Filenames should match the base_image column in your CSV.",
        key="base_images_upload",
    )

    if uploaded_images:
        st.info(f"Uploaded {len(uploaded_images)} images")

    # Handle file upload results
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)

            st.success(f"Loaded {len(df)} edit instructions from {uploaded_file.name}")

            # Show preview
            with st.expander("Preview Edit Instructions", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            return df, template if template else None, uploaded_images
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None, None, uploaded_images

    return df, template if template else None, uploaded_images


def render_edit_settings() -> dict[str, Any]:
    """Render edit-specific settings.

    Returns:
        Dictionary of edit settings.
    """
    st.subheader("Edit Settings")

    col1, col2 = st.columns(2)

    with col1:
        edit_type = st.selectbox(
            "Default Edit Type",
            options=[e.value for e in EditType],
            index=0,
            help="Default type of edit to apply",
        )

    with col2:
        strength = st.slider(
            "Edit Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="How strongly to apply the edit (0.0 = subtle, 1.0 = strong)",
        )

    return {
        "edit_type": edit_type,
        "strength": strength,
    }


async def run_batch_edit(
    df: pd.DataFrame,
    config: dict[str, Any],
    edit_settings: dict[str, Any],
    template: Optional[str],
    uploaded_images: list[Any],
    progress_placeholder: Any,
    results_placeholder: Any,
) -> list[ImageResult]:
    """Run the batch edit process.

    Args:
        df: DataFrame with edit instructions.
        config: Configuration dictionary.
        edit_settings: Edit-specific settings.
        template: Optional edit prompt template.
        uploaded_images: List of uploaded image files.
        progress_placeholder: Streamlit placeholder for progress.
        results_placeholder: Streamlit placeholder for results.

    Returns:
        List of ImageResult objects.
    """
    # Save uploaded images to temp directory
    temp_image_dir = Path("temp_images")
    temp_image_dir.mkdir(exist_ok=True)

    image_paths = {}
    for img in uploaded_images:
        img_path = temp_image_dir / img.name
        with open(img_path, "wb") as f:
            f.write(img.getbuffer())
        image_paths[img.name] = img_path

    # Save DataFrame to temporary file
    temp_file = Path("temp_edit_input.csv")
    df.to_csv(temp_file, index=False)

    try:
        # Create edit job config
        edit_config = EditJobConfig(
            input_file=temp_file,
            base_image_dir=temp_image_dir,
            output_dir=Path(config["output_dir"]),
            edit_template=template,
            default_edit_type=EditType(edit_settings["edit_type"]),
            default_strength=edit_settings["strength"],
            model=ModelName(config["model"]),
            max_workers=config["max_workers"],
            api_key=config["api_key"],
        )

        # Initialize components
        provider = GeminiProvider(
            api_key=config["api_key"],
            default_model=config["model"],
        )
        file_manager = FileManager(edit_config.output_dir)
        processor = BatchEditProcessor(
            provider=provider,
            config=edit_config,
            file_manager=file_manager,
        )

        # Process and update progress
        results = []

        async for result in processor.process_stream():
            results.append(result)
            st.session_state.edit_results = results
            st.session_state.progress = processor.progress
            st.session_state.edit_output_dir = file_manager.output_dir

            # Update progress display
            with progress_placeholder.container():
                render_progress_section(processor.progress)

        return results

    finally:
        # Clean up temp files
        if temp_file.exists():
            temp_file.unlink()
        # Clean up temp images
        for img_path in image_paths.values():
            if img_path.exists():
                img_path.unlink()
        if temp_image_dir.exists():
            try:
                temp_image_dir.rmdir()
            except OSError:
                pass  # Directory not empty


def main() -> None:
    """Main Streamlit application entry point."""
    init_session_state()

    # Render sidebar and get config
    config = render_sidebar()

    # Main content area
    st.title("🍌 BananaBatch")
    st.markdown("*Batch Image Generation & Editing with Nano Banana*")
    st.divider()

    # Check for API key
    if not config["api_key"]:
        st.warning("Please enter your Gemini API key in the sidebar to continue.")
        st.stop()

    # Create tabs for Generate and Edit modes
    tab_generate, tab_edit = st.tabs(["Generate", "Edit"])

    # ==================== GENERATE TAB ====================
    with tab_generate:
        # Input section
        df, template = render_input_section()
        manual_df = render_manual_input()

        # Use manual input if no file uploaded
        if df is None and manual_df is not None:
            df = manual_df

        st.divider()

        # Progress section placeholder
        progress_placeholder = st.empty()

        # Results section placeholder
        results_placeholder = st.empty()

        # Show existing results if any
        if st.session_state.results:
            with results_placeholder.container():
                render_results_section(
                    st.session_state.results,
                    st.session_state.output_dir,
                )

        # Generate button
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if df is not None:
                if st.button(
                    "Generate Images",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.is_running,
                    key="generate_btn",
                ):
                    st.session_state.is_running = True
                    st.session_state.results = []

                    try:
                        results = asyncio.run(
                            run_batch_generation(
                                df,
                                config,
                                template,
                                progress_placeholder,
                                results_placeholder,
                            )
                        )

                        st.session_state.results = results
                        st.success(f"Generation complete! Generated {len(results)} images.")

                    except Exception as e:
                        st.error(f"Error during generation: {e}")

                    finally:
                        st.session_state.is_running = False
                        st.rerun()

    # ==================== EDIT TAB ====================
    with tab_edit:
        # Edit input section
        edit_df, edit_template, uploaded_images = render_edit_input_section()

        # Edit settings
        edit_settings = render_edit_settings()

        st.divider()

        # Progress section placeholder
        edit_progress_placeholder = st.empty()

        # Results section placeholder
        edit_results_placeholder = st.empty()

        # Show existing edit results if any
        if st.session_state.edit_results:
            with edit_results_placeholder.container():
                render_results_section(
                    st.session_state.edit_results,
                    st.session_state.edit_output_dir,
                )

        # Edit button
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            can_edit = edit_df is not None and len(uploaded_images) > 0
            if not can_edit and edit_df is not None:
                st.warning("Please upload base images to edit.")

            if can_edit:
                if st.button(
                    "Edit Images",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.is_running,
                    key="edit_btn",
                ):
                    st.session_state.is_running = True
                    st.session_state.edit_results = []

                    try:
                        results = asyncio.run(
                            run_batch_edit(
                                edit_df,
                                config,
                                edit_settings,
                                edit_template,
                                uploaded_images,
                                edit_progress_placeholder,
                                edit_results_placeholder,
                            )
                        )

                        st.session_state.edit_results = results
                        st.success(f"Editing complete! Edited {len(results)} images.")

                    except Exception as e:
                        st.error(f"Error during editing: {e}")

                    finally:
                        st.session_state.is_running = False
                        st.rerun()

    # Footer
    st.divider()
    st.caption("BananaBatch v0.1.0 | Built with Streamlit & Google Gemini")


if __name__ == "__main__":
    main()
