-- =============================================================================
-- Code Intelligence Platform - Supabase Database Schema
-- =============================================================================
-- This schema should be run against your Supabase project to create the
-- required tables for the Code Intelligence Platform.
-- =============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Audit Logs Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    service VARCHAR(100) DEFAULT 'code-intelligence-platform',
    level VARCHAR(20) DEFAULT 'info',
    action VARCHAR(100) NOT NULL,
    user_id UUID,
    repository VARCHAR(255),
    file_path TEXT,
    details JSONB DEFAULT '{}',
    duration_ms INTEGER,
    status VARCHAR(20) DEFAULT 'success',
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for audit_logs
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_status ON audit_logs(status);
CREATE INDEX IF NOT EXISTS idx_audit_logs_repository ON audit_logs(repository);

-- =============================================================================
-- Analysis Results Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    analysis_type VARCHAR(50) NOT NULL, -- prd, qa, developer, pr_analysis
    repository VARCHAR(255),
    branch VARCHAR(100) DEFAULT 'main',
    commit_hash VARCHAR(64),
    trigger VARCHAR(50) DEFAULT 'manual', -- manual, webhook, scheduled
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, completed, failed
    summary JSONB DEFAULT '{}',
    findings_count INTEGER DEFAULT 0,
    duration_seconds DECIMAL(10, 2),
    agent_model VARCHAR(100),
    raw_results JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Indexes for analysis_results
CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_results_type ON analysis_results(analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_results_status ON analysis_results(status);
CREATE INDEX IF NOT EXISTS idx_analysis_results_repository ON analysis_results(repository);

-- =============================================================================
-- Analysis Findings Table (detailed findings from analysis)
-- =============================================================================
CREATE TABLE IF NOT EXISTS analysis_findings (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    analysis_id UUID REFERENCES analysis_results(id) ON DELETE CASCADE,
    finding_index INTEGER NOT NULL,
    type VARCHAR(50), -- risk, recommendation, issue, suggestion
    severity VARCHAR(20), -- critical, high, medium, low, info
    description TEXT NOT NULL,
    file_path TEXT,
    line_number INTEGER,
    suggestion TEXT,
    confidence DECIMAL(3, 2), -- 0.00 to 1.00
    metadata JSONB DEFAULT '{}'
);

-- Indexes for analysis_findings
CREATE INDEX IF NOT EXISTS idx_analysis_findings_analysis_id ON analysis_findings(analysis_id);
CREATE INDEX IF NOT EXISTS idx_analysis_findings_severity ON analysis_findings(severity);
CREATE INDEX IF NOT EXISTS idx_analysis_findings_type ON analysis_findings(type);

-- =============================================================================
-- Conversations Table (for tracking conversation sessions)
-- =============================================================================
CREATE TABLE IF NOT EXISTS conversations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    conversation_id VARCHAR(100) UNIQUE NOT NULL,
    user_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    agent_type VARCHAR(50), -- architect, developer, qa
    title VARCHAR(255),
    summary TEXT,
    message_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active', -- active, archived, deleted
    metadata JSONB DEFAULT '{}'
);

-- Indexes for conversations
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_agent_type ON conversations(agent_type);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC);

-- =============================================================================
-- Conversation Messages Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS conversation_messages (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    conversation_id VARCHAR(100) NOT NULL,
    message_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    role VARCHAR(20) NOT NULL, -- user, assistant, system
    content TEXT NOT NULL,
    agent_type VARCHAR(50),
    model VARCHAR(100),
    tokens_used INTEGER,
    context_files JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}'
);

-- Indexes for conversation_messages
CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_id ON conversation_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_created_at ON conversation_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_role ON conversation_messages(role);

-- =============================================================================
-- Chat Messages Table (simplified chat storage)
-- =============================================================================
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL, -- user, assistant
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for chat_messages
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);

-- =============================================================================
-- Ingestion Jobs Table (track data ingestion status)
-- =============================================================================
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    job_id VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, completed, failed
    repositories JSONB DEFAULT '[]',
    progress INTEGER DEFAULT 0, -- 0-100
    files_processed INTEGER DEFAULT 0,
    nodes_created INTEGER DEFAULT 0,
    relationships_created INTEGER DEFAULT 0,
    vectors_indexed INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for ingestion_jobs
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_job_id ON ingestion_jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_created_at ON ingestion_jobs(created_at DESC);

-- =============================================================================
-- Users Table (if not using Supabase Auth directly)
-- =============================================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    last_login TIMESTAMPTZ,
    role VARCHAR(20) DEFAULT 'user', -- user, admin, analyst
    preferences JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Index for users
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- =============================================================================
-- Row Level Security (RLS) Policies
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_findings ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE ingestion_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Service role has full access
CREATE POLICY "Service role has full access to audit_logs" 
    ON audit_logs FOR ALL 
    USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role has full access to analysis_results" 
    ON analysis_results FOR ALL 
    USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role has full access to analysis_findings" 
    ON analysis_findings FOR ALL 
    USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role has full access to conversations" 
    ON conversations FOR ALL 
    USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role has full access to conversation_messages" 
    ON conversation_messages FOR ALL 
    USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role has full access to chat_messages" 
    ON chat_messages FOR ALL 
    USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role has full access to ingestion_jobs" 
    ON ingestion_jobs FOR ALL 
    USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role has full access to users" 
    ON users FOR ALL 
    USING (auth.jwt() ->> 'role' = 'service_role');

-- Users can access their own data
CREATE POLICY "Users can view their own conversations" 
    ON conversations FOR SELECT 
    USING (user_id = auth.uid());

CREATE POLICY "Users can view their own analysis results" 
    ON analysis_results FOR SELECT 
    USING (EXISTS (
        SELECT 1 FROM conversations 
        WHERE conversations.user_id = auth.uid()
    ));

-- =============================================================================
-- Functions for updated_at timestamps
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to conversations table
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Cleanup function for old data
-- =============================================================================

CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 30)
RETURNS JSONB AS $$
DECLARE
    cutoff_date TIMESTAMPTZ;
    audit_count INTEGER;
    analysis_count INTEGER;
    message_count INTEGER;
BEGIN
    cutoff_date := NOW() - (days_to_keep || ' days')::INTERVAL;
    
    -- Delete old audit logs
    DELETE FROM audit_logs WHERE timestamp < cutoff_date;
    GET DIAGNOSTICS audit_count = ROW_COUNT;
    
    -- Delete old analysis results (cascade deletes findings)
    DELETE FROM analysis_results WHERE created_at < cutoff_date;
    GET DIAGNOSTICS analysis_count = ROW_COUNT;
    
    -- Delete old chat messages
    DELETE FROM chat_messages WHERE created_at < cutoff_date;
    GET DIAGNOSTICS message_count = ROW_COUNT;
    
    RETURN jsonb_build_object(
        'audit_logs_deleted', audit_count,
        'analysis_results_deleted', analysis_count,
        'chat_messages_deleted', message_count,
        'cutoff_date', cutoff_date
    );
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Sample RPC functions for common operations
-- =============================================================================

-- Get recent activity summary
CREATE OR REPLACE FUNCTION get_activity_summary(hours_back INTEGER DEFAULT 24)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'analysis_count', (SELECT COUNT(*) FROM analysis_results WHERE created_at > NOW() - (hours_back || ' hours')::INTERVAL),
        'conversation_count', (SELECT COUNT(*) FROM conversations WHERE created_at > NOW() - (hours_back || ' hours')::INTERVAL),
        'audit_log_count', (SELECT COUNT(*) FROM audit_logs WHERE timestamp > NOW() - (hours_back || ' hours')::INTERVAL),
        'ingestion_count', (SELECT COUNT(*) FROM ingestion_jobs WHERE created_at > NOW() - (hours_back || ' hours')::INTERVAL),
        'time_range_hours', hours_back
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;
