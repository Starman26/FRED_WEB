-- Migration: diagnostic_history
-- Purpose: Store completed diagnostic sessions for learning and similarity search
-- Run in: Supabase SQL Editor or migration tool

-- Tabla para historial de diagnósticos
CREATE TABLE IF NOT EXISTS lab.diagnostic_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Context
    session_id TEXT,
    user_id TEXT,
    team_id UUID,

    -- Problem
    user_query TEXT NOT NULL,
    equipment_type TEXT,          -- 'plc', 'cobot', 'sensor', 'general'
    station_number INTEGER,
    severity TEXT,                -- 'critical', 'high', 'medium', 'low'

    -- Diagnosis
    diagnosis TEXT NOT NULL,      -- La respuesta/diagnóstico del agente
    root_cause TEXT,              -- Causa raíz identificada (si se determinó)
    confidence FLOAT,             -- 0.0-1.0

    -- Resolution
    actions_taken JSONB DEFAULT '[]',   -- [{tool, success, verified}]
    resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT,

    -- Learning
    tools_used JSONB DEFAULT '[]',      -- [tool_names]
    evidence_sources JSONB DEFAULT '[]', -- [{title, page, type}]
    lesson_learned TEXT,                 -- Reflexion output

    -- Metadata
    tokens_used INTEGER DEFAULT 0,
    duration_ms FLOAT,
    workers_used JSONB DEFAULT '[]',    -- [worker_names]

    -- Search
    embedding VECTOR(1536)              -- Para similarity search
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_diagnostic_history_team
    ON lab.diagnostic_history(team_id);
CREATE INDEX IF NOT EXISTS idx_diagnostic_history_station
    ON lab.diagnostic_history(station_number);
CREATE INDEX IF NOT EXISTS idx_diagnostic_history_equipment
    ON lab.diagnostic_history(equipment_type);
CREATE INDEX IF NOT EXISTS idx_diagnostic_history_created
    ON lab.diagnostic_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_diagnostic_history_embedding
    ON lab.diagnostic_history
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- RLS
ALTER TABLE lab.diagnostic_history ENABLE ROW LEVEL SECURITY;

-- Policy: team members can access their team's diagnostics
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'diagnostic_history' AND policyname = 'team_access'
    ) THEN
        CREATE POLICY "team_access" ON lab.diagnostic_history
            FOR ALL USING (team_id = current_setting('app.current_team_id')::UUID);
    END IF;
END $$;

-- Function for similarity search
CREATE OR REPLACE FUNCTION lab.match_diagnostics(
    query_embedding VECTOR(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 3,
    filter_team_id UUID DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    user_query TEXT,
    diagnosis TEXT,
    root_cause TEXT,
    severity TEXT,
    confidence FLOAT,
    actions_taken JSONB,
    lesson_learned TEXT,
    created_at TIMESTAMPTZ,
    similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
    SELECT
        d.id, d.user_query, d.diagnosis, d.root_cause,
        d.severity, d.confidence, d.actions_taken, d.lesson_learned,
        d.created_at,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM lab.diagnostic_history d
    WHERE
        d.embedding IS NOT NULL
        AND (filter_team_id IS NULL OR d.team_id = filter_team_id)
        AND 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
$$;
