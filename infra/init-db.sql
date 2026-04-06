-- Forecaster — initial database setup for local development
-- This script runs automatically when the PostgreSQL container is first created.

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- pgvector extension (for future agent memory / embeddings)
-- CREATE EXTENSION IF NOT EXISTS vector;
