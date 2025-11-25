# BananaBatch 🍌

Batch image generation and editing using Google Gemini.

## Installation

```bash
git clone https://github.com/GH05TCREW/bananabatch.git
cd bananabatch
pip install -e .
```

Set your API key:
```bash
export GEMINI_API_KEY=your_key_here  # or add to .env file
```

## Quick Start

### Generate Images

```bash
# From CSV with prompts
bananabatch generate -i prompts.csv

# With template variables
bananabatch generate -i products.csv -t "A photo of {{ product }} on {{ background }}"
```

### Edit Images

```bash
# Basic editing
bananabatch edit -i edits.csv -d ./images/

# Single image with multiple edits (e.g., defect generation)
bananabatch edit -i defects.csv --default-image product.jpg -t "Add a {{ defect }} at {{ location }}"
```

### GUI

```bash
streamlit run bananabatch/gui/app.py
```

## Input Formats

### CSV for Generation

```csv
prompt,filename
A mountain landscape,mountain
A robot playing guitar,robot
```

Or with template variables:
```csv
item,style
mountain,watercolor
robot,digital art
```

### CSV for Editing

```csv
base_image,edit_prompt,edit_type,strength
photo.jpg,Add warm lighting,transform,0.7
```

Or without `base_image` column when using `--default-image`:
```csv
defect_type,location
scratch,center
dent,corner
```

## CLI Reference

### Generate Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --input` | Required | CSV/JSON input file |
| `-m, --model` | `gemini-2.5-flash-image` | Model to use |
| `-o, --output` | `outputs/` | Output directory |
| `-t, --template` | None | Jinja2 template |
| `-w, --workers` | 5 | Concurrent workers |

### Edit Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --input` | Required | CSV/JSON with edit instructions |
| `-d, --images` | None | Base images directory |
| `-b, --default-image` | None | Default image for all rows |
| `-e, --type` | `transform` | Edit type |
| `-s, --strength` | 0.75 | Edit intensity (0-1) |
| `-w, --workers` | 2 | Concurrent workers |

### Edit Types

- `transform` - General modifications (lighting, effects)
- `style_transfer` - Artistic styles (painting, sketch)
- `inpaint` - Fill/modify regions
- `anomaly` - Synthetic defects for ML training
- `variation` - Create image variations

## Models

- `gemini-2.5-flash-image` - Fast (default)
- `gemini-3-pro-image-preview` - Higher quality

## Development

```bash
pip install -e ".[dev]"
pytest
black bananabatch && ruff check bananabatch --fix
```

## License

MIT