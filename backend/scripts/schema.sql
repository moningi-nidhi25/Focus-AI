-- FocusAI Database Schema (PostgreSQL / Supabase)
-- Phase 2: Database and Multi-Tenant Architecture

-- Create UUID extension if not present
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('STUDENT', 'TEACHER', 'ADMIN')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Cohorts Table (Courses / Classes)
CREATE TABLE IF NOT EXISTS cohorts (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    teacher_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Cohort Memberships Table
CREATE TABLE IF NOT EXISTS cohort_memberships (
    cohort_id VARCHAR(36) NOT NULL REFERENCES cohorts(id) ON DELETE CASCADE,
    student_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (cohort_id, student_id)
);

-- 4. Study Sessions Table
CREATE TABLE IF NOT EXISTS study_sessions (
    id VARCHAR(36) PRIMARY KEY,
    student_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    cohort_id VARCHAR(36) REFERENCES cohorts(id) ON DELETE SET NULL,
    title VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed')),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    total_duration_seconds INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Study Contexts Table
CREATE TABLE IF NOT EXISTS study_contexts (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) UNIQUE NOT NULL REFERENCES study_sessions(id) ON DELETE CASCADE,
    topic VARCHAR(255) NOT NULL,
    learning_objectives TEXT,
    context_embedding TEXT, -- JSON {model, vector} (dense embedding models only)
    course_material TEXT, -- extracted course notes / documents (optional)
    keywords TEXT, -- JSON list of structured keywords
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. Content Analysis Table
CREATE TABLE IF NOT EXISTS content_analysis (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES study_sessions(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title VARCHAR(255),
    extracted_text TEXT,
    relevance_score FLOAT NOT NULL CHECK (relevance_score >= 0 AND relevance_score <= 100),
    decision VARCHAR(10) NOT NULL CHECK (decision IN ('ALLOW', 'WARN', 'BLOCK')),
    explanation TEXT,
    key_concepts TEXT, -- JSON list of matching concepts
    classification VARCHAR(30), -- RELEVANT | PARTIALLY_RELEVANT | DISTRACTING
    heatmap TEXT, -- JSON per-sentence relevance (Phase 6)
    signals TEXT, -- JSON individual relevance signals
    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. Activity Events Table
CREATE TABLE IF NOT EXISTS activity_events (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES study_sessions(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT -- JSON string payload
);

-- 8. Blocked Content Table
CREATE TABLE IF NOT EXISTS blocked_content (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES study_sessions(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title VARCHAR(255),
    reason TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 9. Focus Metrics Table
CREATE TABLE IF NOT EXISTS focus_metrics (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES study_sessions(id) ON DELETE CASCADE,
    focus_score FLOAT NOT NULL CHECK (focus_score >= 0 AND focus_score <= 100),
    productivity_score FLOAT NOT NULL CHECK (productivity_score >= 0 AND productivity_score <= 100),
    dwell_time_seconds INTEGER DEFAULT 0,
    tab_switch_count INTEGER DEFAULT 0,
    distraction_count INTEGER DEFAULT 0,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 10. Focus Predictions Table
CREATE TABLE IF NOT EXISTS focus_predictions (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES study_sessions(id) ON DELETE CASCADE,
    predicted_focus_score FLOAT NOT NULL,
    target_time_window VARCHAR(50),
    confidence FLOAT,
    curve_data TEXT, -- JSON string representing time-series prediction points
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for optimal querying
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_cohorts_teacher ON cohorts(teacher_id);
CREATE INDEX IF NOT EXISTS idx_study_sessions_student ON study_sessions(student_id);
CREATE INDEX IF NOT EXISTS idx_study_sessions_cohort ON study_sessions(cohort_id);
CREATE INDEX IF NOT EXISTS idx_content_analysis_session ON content_analysis(session_id);
CREATE INDEX IF NOT EXISTS idx_activity_events_session ON activity_events(session_id);
CREATE INDEX IF NOT EXISTS idx_focus_metrics_session ON focus_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_focus_predictions_session ON focus_predictions(session_id);

-- Existing database from Phase 2? The backend adds these columns automatically on startup
-- (init_db -> ensure_columns). To do it by hand instead:
-- ALTER TABLE study_contexts ADD COLUMN IF NOT EXISTS course_material TEXT, ADD COLUMN IF NOT EXISTS keywords TEXT;
-- ALTER TABLE content_analyses ADD COLUMN IF NOT EXISTS classification VARCHAR(30), ADD COLUMN IF NOT EXISTS heatmap TEXT, ADD COLUMN IF NOT EXISTS signals TEXT;
