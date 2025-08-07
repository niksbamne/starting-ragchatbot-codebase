# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
- **Start server**: `./run.sh` or `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Install dependencies**: `uv sync`
- **Server runs on**: http://localhost:8000 (serves both API and web interface)
- **API documentation**: http://localhost:8000/docs

### Development Commands
- **Python management**: This project uses `uv` as the package manager
- **Environment setup**: Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials with the following architecture:

### Core Components

**RAG System (`rag_system.py`)**: Main orchestrator that coordinates all components
- Integrates document processing, vector storage, AI generation, and session management
- Handles course document ingestion from the `docs/` folder
- Manages tool-based search functionality

**Vector Storage (`vector_store.py`)**: ChromaDB-based storage with two collections
- `course_catalog`: Stores course metadata and titles for semantic course matching
- `course_content`: Stores chunked course content for content search
- Uses SentenceTransformer embeddings (`all-MiniLM-L6-v2`)

**Document Processing (`document_processor.py`)**: Structured course document parser
- Expects specific format: Course Title, Course Link, Course Instructor, then Lesson sections
- Creates sentence-based chunks with configurable overlap (800 chars, 100 char overlap)
- Extracts course metadata and lesson structure

**AI Generation (`ai_generator.py`)**: Anthropic Claude API integration
- Uses `claude-sonnet-4-20250514` model with tool calling capability
- Implements conversation context and tool-based search
- System prompt optimized for educational content with concise responses

**Search Tools (`search_tools.py`)**: Tool-based search interface
- `CourseSearchTool`: Provides semantic course name matching and lesson filtering
- `ToolManager`: Handles tool registration and execution for AI

**Session Management (`session_manager.py`)**: Conversation state management
- Maintains conversation history (configurable max: 2 exchanges)
- Session-based context for follow-up questions

### Data Models (`models.py`)
- `Course`: Title, instructor, lessons, course links
- `Lesson`: Lesson number, title, optional lesson link
- `CourseChunk`: Text chunks with course/lesson metadata for vector storage

### FastAPI Application (`app.py`)
- REST API endpoints: `/api/query` (chat), `/api/courses` (stats)
- Serves static frontend files from `frontend/` directory
- CORS enabled for development
- Auto-loads course documents from `docs/` folder on startup

### Frontend (`frontend/`)
- Vanilla JavaScript chat interface with sidebar
- Markdown rendering support via `marked.js`
- Course statistics display and suggested questions
- Session-based conversation tracking

## Configuration (`config.py`)

Key settings stored in `Config` dataclass:
- `ANTHROPIC_API_KEY`: Required for Claude API access
- `CHUNK_SIZE`: 800 characters for document chunks
- `CHUNK_OVERLAP`: 100 characters overlap between chunks
- `MAX_RESULTS`: 5 maximum search results returned
- `MAX_HISTORY`: 2 conversation exchanges remembered
- `CHROMA_PATH`: `./chroma_db` for persistent vector storage

## Course Document Format

Expected format for documents in `docs/` folder:
```
Course Title: [Title]
Course Link: [URL]
Course Instructor: [Name]

Lesson 1: [Lesson Title]
Lesson Link: [URL]
[Lesson content...]

Lesson 2: [Next Lesson Title]
[Content continues...]
```

## Key Integration Points

- **Tool-based search**: AI uses `search_course_content` tool instead of direct vector queries
- **Session continuity**: Frontend maintains session IDs for conversation context
- **Semantic course matching**: Vector search enables partial course name matching
- **Lesson filtering**: Search can be restricted to specific lessons within courses
- **Source tracking**: Search results include course/lesson attribution for UI display